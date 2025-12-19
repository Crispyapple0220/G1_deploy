import os
import time
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
from g1_arm_IK import G1_29_ArmIK

ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)

# q_current = get_current_arm_joints_from_robot()
q_current = np.zeros(ik.reduced_robot.model.nq)

moving1 = np.array([[0.20, 0, 0.10, 0], [-0.20, 0, -0.10, 0]])   # 20 cm forward, 10 cm upward, 45 degrees to the right


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


def steering_twist(theta_deg, radius=0.20):
    angle = np.deg2rad(theta_deg)

    dy = radius * (1 - np.cos(angle))
    dy_L =  dy
    dy_R =  dy

    dz = radius * np.sin(angle)
    dz_L =  dz
    dz_R = -dz

    dx_L = 0.0
    dx_R = 0.0

    twist_L =  theta_deg
    twist_R = -theta_deg

    moving_left  = np.array([dx_L, dy_L, dz_L, twist_L])
    moving_right = np.array([dx_R, dy_R, dz_R, twist_R])

    return moving_left, moving_right


# One shot moving
def move_rod(ik, q_current, moving, n_steps = 100, rate_hz = 25):

    L_SE3, R_SE3 = ik.forward_kinematics(q_current)
    L = L_SE3.translation.copy()
    R = R_SE3.translation.copy()
    L_rot = L_SE3.rotation.copy()
    R_rot = R_SE3.rotation.copy()

    axis = R - L
    radius = np.linalg.norm(axis) / 2
    moving_left, moving_right = steering_twist(moving[3], radius)
    movingx = moving.copy()
    movingx[3] = 0
    moving_left += movingx
    moving_right += movingx

    axis = axis / np.linalg.norm(axis)
    angle_left = np.deg2rad(moving_left[3])
    angle_right = np.deg2rad(moving_right[3])

    # Final rod pose (L', R')
    shift_left = moving_left[:3]
    shift_right = moving_right[:3]
    L_prime = L + shift_left
    R_prime = R + shift_right

    dt = 1.0 / rate_hz

    for i in range(n_steps + 1):
        s = i / n_steps   # 0 → 1
        twist_s_left = Rotation.from_rotvec(axis * (angle_left * s))
        twist_s_right = Rotation.from_rotvec(axis * (angle_right * s))

        L_pos_t = (1 - s) * L + s * L_prime
        R_pos_t = (1 - s) * R + s * R_prime
        L_rot_t = twist_s_left.as_matrix() @ L_rot
        R_rot_t = twist_s_right.as_matrix() @ R_rot
        L_target = pin.SE3(L_rot_t, L_pos_t)
        R_target = pin.SE3(R_rot_t, R_pos_t)

        q_sol, _ = ik.solve_ik(
            L_target.homogeneous,
            R_target.homogeneous,
            current_lr_arm_motor_q=q_current
        )


        # ===== Recording: store one frame per solved step =====
        if RECORD:
            record_frames.append(pack_frame(ik, q_sol))


        # send_to_robot(q_sol)

        q_current = q_sol

        time.sleep(dt)

    return q_current


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


if __name__ == "__main__":

    # Add a function to make sure the initial position is at zero position if needed

    for i in range(5):
        q_current = np.zeros(ik.reduced_robot.model.nq)
        q_current = move_rod(ik, q_current, np.array([0.10, 0, 0.10, -15]), rate_hz=RECORD_RATE_HZ)
        q_current = move_rod(ik, q_current, np.array([-0.10, 0, -0.10, 25]), rate_hz=RECORD_RATE_HZ)
        q_current = move_rod(ik, q_current, np.array([0, 0, 0, -10]), rate_hz=RECORD_RATE_HZ)

    if RECORD:
        save_recording_npz(
            record_frames,
            out_path="records/sim_rod_twist.npz",
            dt=RECORD_DT
        )
