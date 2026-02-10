import os
import time
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
from g1_arm_IK import G1_29_ArmIK

# =========================
# Recording (NPZ) settings
# =========================
RECORD = False
TEST_MODE = True
CHECK_WORKSPACE = False
RECORD_RATE_HZ = 20
RECORD_DT = 1.0 / RECORD_RATE_HZ
record_frames = []

def pack_frame(ik, q):
    """
    Frame format (compatible with g1_high_level_controller.py):
      [ q | pL(3) | quatL(4) | pR(3) | quatR(4) ]
    where quat is (x, y, z, w) from scipy Rotation.as_quat().
    """
    L_SE3, R_SE3 = ik.forward_kinematics(q)

    pL = L_SE3.translation
    pR = R_SE3.translation

    quatL = Rotation.from_matrix(L_SE3.rotation).as_quat()
    quatR = Rotation.from_matrix(R_SE3.rotation).as_quat()

    return np.hstack([q, pL, quatL, pR, quatR])

def move_end_effectors_6dof_object_frame(
    ik,
    q_current: np.ndarray,
    motion_mat: np.ndarray,
    *,
    diameter: float = 0.40,
    rate_hz: float | None = None,
):
    """
    Execute an object-centric incremental motion_mat (T,6) by converting each step into
    coordinated left/right hand increments using object_to_hands(...), then applying
    the increments in each hand's local frame.

    motion_mat rows are:
        [dx, dy, dz, theta_x, theta_y, theta_z]
    - translations in meters
    - angles in degrees (XYZ Euler)

    Parameters
    ----------
    ik : your IK object
        Must provide:
          - ik.forward_kinematics(q) -> (L_SE3, R_SE3) as pin.SE3
          - ik.solve_ik(L_target_4x4, R_target_4x4, current_lr_arm_motor_q=q_current)

    q_current : (nq,) np.ndarray
        Current reduced-model joint configuration.

    motion_mat : (T,6) np.ndarray
        Object-centric incremental motions.

    diameter : float
        Object diameter used by object_to_hands() to compute left/right offsets.

    rate_hz : float | None
        If provided, sleeps 1/rate_hz seconds per step.

    Returns
    -------
    q_current : np.ndarray
        Final joint configuration.

    q_traj : (T, nq) np.ndarray
        Joint trajectory (one solved q per step).
    """
    motion_mat = np.asarray(motion_mat, dtype=float)
    if motion_mat.ndim != 2 or motion_mat.shape[1] != 6:
        raise ValueError(f"motion_mat must have shape (T, 6), got {motion_mat.shape}")

    q_current = np.asarray(q_current, dtype=float).copy()
    q_traj = []

    dt = None
    if rate_hz is not None and rate_hz > 0:
        dt = 1.0 / float(rate_hz)

    for mv in motion_mat:
        mv = np.asarray(mv, dtype=float).reshape(-1)
        if mv.size != 6:
            raise ValueError(f"Each row of motion_mat must have 6 elements, got {mv.shape}")

        # 1) Convert object motion -> per-hand increments
        #    object_to_hands should return (L_mv, R_mv), each shape (6,)
        L_mv, R_mv = object_to_hands(ik, q_current, mv, diameter=diameter)

        # 2) Current end-effector poses from FK
        L_SE3, R_SE3 = ik.forward_kinematics(q_current)

        # 3) Build incremental transforms for each hand (angles are degrees, XYZ)
        def mv_to_dT(mv_6):
            dx, dy, dz, thx, thy, thz = mv_6
            dR = Rotation.from_euler("xyz", [thx, thy, thz], degrees=True).as_matrix()
            dp = np.array([dx, dy, dz], dtype=float)
            return pin.SE3(dR, dp)

        dT_L = mv_to_dT(L_mv)
        dT_R = mv_to_dT(R_mv)

        # 4) HAND-FRAME application: apply increment in each hand's local frame
        L_target = L_SE3 * dT_L
        R_target = R_SE3 * dT_R

        # 5) Solve IK with both targets
        q_sol, _ = ik.solve_ik(
            L_target.homogeneous,
            R_target.homogeneous,
            current_lr_arm_motor_q=q_current
        )

        # 6) Check if hitting the workspace edge
        if CHECK_WORKSPACE:
            L_act, R_act = ik.forward_kinematics(q_sol)

            edge, reason = workspace_edge_detected(
                L_target, R_target,
                L_act, R_act,
                q_sol, q_current
            )

            if edge:
                print(f"[WARN] Workspace boundary reached ({reason})")
                break  # or scale motion / stop


        # ===== Recording: store one frame per solved step =====
        if RECORD:
            record_frames.append(pack_frame(ik, q_sol))

        q_current = q_sol
        q_traj.append(q_sol.copy())

        if dt is not None:
            time.sleep(dt)

    return q_current, np.vstack(q_traj) if len(q_traj) > 0 else np.empty((0, q_current.shape[0]))



def make_smooth_motion_mat(
    movement_1x6,
    *,
    T: int = 10,
    mode: str = "repeat",
    ramp: str = "linear",
):
    """
    Create a (T, 6) motion_mat from a single (1, 6) or (6,) movement command.

    movement_1x6: [dx, dy, dz, theta_x, theta_y, theta_z]
      - dx,dy,dz in meters
      - theta_* in degrees

    Parameters
    ----------
    T : int
        Number of timesteps in the output motion_mat.

    mode : {"repeat", "ramp"}
        - "repeat": split movement equally across T steps (all rows identical).
                    Example: (1,0,0,5,0,0), T=10 -> each row (0.1,0,0,0.5,0,0)
        - "ramp":   the per-step increments are scaled over time (smooth start/stop),
                    but the SUM over all rows still equals the requested movement.

    ramp : {"linear", "cosine"}
        Only used when mode="ramp".
        - "linear": weights increase linearly
        - "cosine": half-cosine easing (very smooth at start/end)

    Returns
    -------
    motion_mat : (T, 6) np.ndarray
        Sequence of incremental commands. Summing all rows yields the original movement.
    """
    mv = np.asarray(movement_1x6, dtype=float).reshape(-1)
    if mv.shape[0] != 6:
        raise ValueError(f"movement must have 6 elements, got shape {np.asarray(movement_1x6).shape}")

    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    if mode not in ("repeat", "ramp"):
        raise ValueError(f"mode must be 'repeat' or 'ramp', got {mode}")

    if ramp not in ("linear", "cosine"):
        raise ValueError(f"ramp must be 'linear' or 'cosine', got {ramp}")

    if mode == "repeat":
        step = mv / T
        return np.tile(step, (T, 1))

    # mode == "ramp"
    # Build nonnegative weights and normalize so sum(weights) == 1
    n = np.arange(1, T + 1, dtype=float)

    if ramp == "linear":
        w = n  # 1,2,...,T
    else:  # cosine easing: smooth start and end
        # weights: sin^2 profile over [0, pi]
        t = np.linspace(0.0, np.pi, T, dtype=float)
        w = np.sin(t) ** 2
        # In case T=1, sin(0)=0 -> handle gracefully
        if np.allclose(w.sum(), 0.0):
            w = np.ones(T, dtype=float)

    w = w / w.sum()
    motion_mat = w[:, None] * mv[None, :]
    return motion_mat


def save_recording_npz(
    frames,
    out_path="records/sim_rod_twist.npz",
    dt=RECORD_DT,
    note="cols=[q | pL(3) | quatL(4) | pR(3) | quatR(4)]"
):
    if len(frames) == 0:
        print("[INFO] No frames recorded; nothing to save.")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    traj = np.vstack(frames)

    np.savez_compressed(
        out_path,
        traj=traj,
        dt=dt,
        note=note
    )

    print(f"[INFO] Saved recording to {out_path}")
    print(f"[INFO] traj shape = {traj.shape} (frames, dims)")


def object_to_hands(ik, q_current: np.ndarray, mv, diameter: float = 0.40):
    """
    Convert an object-centric 6DoF motion mv = [dx, dy, dz, thx, thy, thz] (degrees)
    into left-hand and right-hand incremental motions, assuming the hands are on
    opposite sides of an object of given diameter.

    Returns:
        L_mv, R_mv : each shape (6,)
    """
    mv = np.asarray(mv, dtype=float).reshape(-1)
    if mv.size != 6:
        raise ValueError(f"mv must have 6 elements (shape (6,) or (1,6)), got {np.asarray(mv).shape}")

    q_current = np.asarray(q_current, dtype=float)

    L_SE3, R_SE3 = ik.forward_kinematics(q_current)

    # Current hand Euler angles (degrees), XYZ convention
    L_theta = Rotation.from_matrix(L_SE3.rotation).as_euler("xyz", degrees=True)
    R_theta = Rotation.from_matrix(R_SE3.rotation).as_euler("xyz", degrees=True)

    half = 0.5 * float(diameter)
    dx, dy, dz, thx, thy, thz = mv  # th* in degrees

    def hand_mv(theta, sign: float):
        """
        sign = +1 for right hand, -1 for left hand (mirrored offsets).
        """
        # Precompute radians used in your formulas
        yaw_sum = np.deg2rad(theta[2] + thz)
        yaw = np.deg2rad(theta[2])

        roll_sum = np.deg2rad(theta[0] + thx)
        roll = np.deg2rad(theta[0])

        dmx = dx + sign * half * (np.sin(yaw_sum) - np.sin(yaw))
        dmy = dy + sign * half * (
            (np.cos(roll_sum) - np.cos(roll)) +
            (np.cos(yaw_sum) - np.cos(yaw))
        )
        dmz = dz - sign * half * (np.sin(roll_sum) - np.sin(roll))

        return np.array([dmx, dmy, dmz, thx, thy, thz], dtype=float)

    R_mv = hand_mv(R_theta, sign=+1.0)
    L_mv = hand_mv(L_theta, sign=-1.0)

    return L_mv, R_mv


def workspace_edge_detected(
    L_target, R_target,
    L_actual, R_actual,
    q_sol, q_current,
    *,
    pos_tol=0.05,        # 5 cm
    joint_step_tol=0.2,  # rad
):
    pos_err_L = np.linalg.norm(
        L_target.translation - L_actual.translation
    )
    pos_err_R = np.linalg.norm(
        R_target.translation - R_actual.translation
    )

    dq = np.linalg.norm(q_sol - q_current)

    if pos_err_L > pos_tol or pos_err_R > pos_tol:
        return True, "pose_error"

    if dq > joint_step_tol:
        return True, "joint_jump"

    return False, "ok"



# mv = (1, 0, 0, 5, 0, 0)

# motion_mat = make_smooth_motion_mat(mv, T=10, mode="repeat")
# # -> each row is (0.1, 0, 0, 0.5, 0, 0)

# motion_mat2 = make_smooth_motion_mat(mv, T=20, mode="ramp", ramp="cosine")
# # -> per-step increments start small, get bigger, then small again
# #    (sum over rows still equals (1,0,0,5,0,0))

# # motion_mat: T steps of (dx, dy, dz, thx, thy, thz)
# motion_mat = np.array([
#     [ 0.01,  0.00,  0.00,   0.0,  0.0,  0.0],   # 1 cm forward (hand x)
#     [ 0.00,  0.01,  0.00,   0.0,  0.0,  0.0],   # 1 cm left    (hand y)
#     [ 0.00,  0.00,  0.01,   0.0,  0.0,  0.0],   # 1 cm up      (hand z)
#     [ 0.00,  0.00,  0.00,   5.0,  0.0,  0.0],   # +5 deg roll  (hand x)
#     [ 0.00,  0.00,  0.00,   0.0,  5.0,  0.0],   # +5 deg pitch (hand y)
#     [ 0.00,  0.00,  0.00,   0.0,  0.0,  5.0],   # +5 deg yaw   (hand z)
# ])



if __name__ == "__main__":

    ik = G1_29_ArmIK(Unit_Test=TEST_MODE, Visualization=TEST_MODE)

    mv = np.array([
        [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.3, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 45.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -45.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 45.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -45.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 30.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -30.0],
    ])

    mv_complex_seq = np.array([
        [ 0.12,  0.00,  0.06,   0.0,   0.0,   0.0],   # forward + up (reach out)
        [ 0.00,  0.10,  0.00,   0.0,   0.0,   0.0],   # right (lateral push)
        [ 0.00,  0.00,  0.00,  12.0,  -8.0,  18.0],   # roll + pitch + yaw (full 6DoF rotation)
        [ 0.06, -0.06,  0.00,  -8.0,   0.0, -12.0],   # diagonal translation + counter-rotation
        [-0.10,  0.00, -0.06,   0.0,  10.0,  25.0],   # retract + down + pitch + yaw (stress)
        [ 0.00, -0.12,  0.00, -10.0,   0.0, -20.0],   # left hard + roll + yaw back (reversal)
        [ 0.05,  0.00,  0.03,   8.0,   6.0,   0.0],   # small re-approach + roll/pitch
        [-0.03,  0.03,  0.00,   0.0,  -8.0,  10.0],   # small diagonal + pitch back + yaw
    ])


    motion_mat = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    for moving in mv_complex_seq:
        motion_mat = np.concatenate((motion_mat, make_smooth_motion_mat(moving, T=20, mode="ramp", ramp="cosine")))

    q_current = np.zeros(ik.reduced_robot.model.nq)

    q_current, q_traj = move_end_effectors_6dof_object_frame(
        ik,
        q_current,
        motion_mat,
        diameter=0.30,
        rate_hz=20,
    )
    q_current = np.zeros(ik.reduced_robot.model.nq)

    if RECORD:
        save_recording_npz(
            record_frames,
            out_path="records/sim_free_6DoF.npz",
            dt=RECORD_DT
        )

