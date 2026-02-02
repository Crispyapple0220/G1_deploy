import os
import time
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
from g1_arm_IK import G1_29_ArmIK

# =========================
# Recording (NPZ) settings
# =========================
RECORD = True
RECORD_RATE_HZ = 50
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

def move_right_hand_6dof_hand_frame(
    ik,
    q_current: np.ndarray,
    motion_mat: np.ndarray,
    *,
    rate_hz: float | None = None,
):
    """
    Move the right-hand end-effector using incremental 6DoF commands expressed in the *hand frame*.

    Each row of motion_mat is:
        [dx, dy, dz, theta_x, theta_y, theta_z]

    - Translations (dx,dy,dz): meters, expressed along the CURRENT hand axes.
    - Rotations (theta_*): degrees, XYZ Euler increments about the CURRENT hand axes.
    - Left hand is held fixed at its current pose each step (acts as an anchor),
      matching the dual-target IK signature in your g1_arm_IK.py.

    Parameters
    ----------
    ik : G1_29_ArmIK
        Must provide:
          - ik.forward_kinematics(q) -> (L_SE3, R_SE3) as pin.SE3
          - ik.solve_ik(L_target_4x4, R_target_4x4, current_lr_arm_motor_q=q_current)

    q_current : (nq,) np.ndarray
        Current reduced-model arm joint vector.

    motion_mat : (T, 6) np.ndarray
        Incremental motions per step.

    rate_hz : float | None
        If set, sleeps 1/rate_hz seconds per step (useful for real-time stepping).

    Returns
    -------
    q_current : np.ndarray
        Final joint configuration after executing all increments.

    q_traj : (T, nq) np.ndarray
        Joint trajectory (one solved q per input row).
    """
    motion_mat = np.asarray(motion_mat, dtype=float)
    if motion_mat.ndim != 2 or motion_mat.shape[1] != 6:
        raise ValueError(f"motion_mat must have shape (T, 6), got {motion_mat.shape}")

    q_current = np.asarray(q_current, dtype=float).copy()
    q_traj = []

    dt = None
    if rate_hz is not None and rate_hz > 0:
        dt = 1.0 / float(rate_hz)

    for dx, dy, dz, thx, thy, thz in motion_mat:
        # 1) Current end-effector poses from FK
        L_SE3, R_SE3 = ik.forward_kinematics(q_current)

        # 2) Build incremental transform dT from the 6DoF input
        #    Angles are DEGREES (as requested), XYZ Euler increments.
        dR = Rotation.from_euler("xyz", [thx, thy, thz], degrees=True).as_matrix()
        dp = np.array([dx, dy, dz], dtype=float)
        dT = pin.SE3(dR, dp)

        # 3) HAND-FRAME application (local/body-frame increment):
        #    Move/rotate relative to the CURRENT hand axes.
        R_target = R_SE3 * dT

        # 4) Keep left hand fixed (anchor)
        L_target = L_SE3

        # 5) Solve IK
        q_sol, _ = ik.solve_ik(
            L_target.homogeneous,
            R_target.homogeneous,
            current_lr_arm_motor_q=q_current
        )

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

    ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)

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
        [0.0, 0.0, 0.0, 0.0, 0.0, 45.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -45.0],
    ])

    motion_mat = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    for moving in mv:
        motion_mat = np.concatenate((motion_mat, make_smooth_motion_mat(moving, T=20, mode="ramp", ramp="cosine")))

    q_current = np.zeros(ik.reduced_robot.model.nq)


    q_current, q_traj = move_right_hand_6dof_hand_frame(
        ik,
        q_current,
        motion_mat,
        rate_hz=20,
    )
    q_current = np.zeros(ik.reduced_robot.model.nq)

    if RECORD:
        save_recording_npz(
            record_frames,
            out_path="records/sim_6DoF.npz",
            dt=RECORD_DT
        )

