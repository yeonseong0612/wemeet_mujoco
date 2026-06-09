import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.ik import get_joint_ids, solve_pose_ik


# ============================================================
# Rotation test mode
# ============================================================
# 0: XML site rotation 그대로 사용
# 1: target rotation에 local X축 기준 180도 추가
# 2: target rotation에 local Y축 기준 180도 추가
# 3: target rotation에 localZ축 기준 180도 추가
ROTATION_MODE = 1


def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return project_root


def reset_model(model, data, keyframe_id=0):
    mujoco.mj_resetData(model, data)

    if model.nkey > keyframe_id:
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

    mujoco.mj_forward(model, data)


def get_site_pose(model, data, site_name):
    site_id = model.site(site_name).id
    pos = data.site_xpos[site_id].copy()
    rot = data.site_xmat[site_id].reshape(3, 3).copy()
    return site_id, pos, rot


def print_matrix(name, mat):
    print(name)
    for row in mat:
        print("  ", row)


def print_model_info(model):
    print("Loaded model")
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("nbody:", model.nbody)
    print("njnt:", model.njnt)
    print("nsite:", model.nsite)

    print("\n==== Sites ====")
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        print(i, name)

    print("\n==== Actuators ====")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(i, name)


def rot_x_180():
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])


def rot_y_180():
    return np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])


def rot_z_180():
    return np.array([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


def apply_rotation_debug_mode(target_rot):
    """
    XML에서 가져온 target_rot을 테스트용으로 보정한다.

    ROTATION_MODE:
        0: no change
        1: local X 180 deg
        2: local Y 180 deg
        3: local Z 180 deg
    """
    if ROTATION_MODE == 0:
        print("[Rotation mode] use XML site rotation")
        return target_rot

    if ROTATION_MODE == 1:
        print("[Rotation mode] target_rot @ RotX(180)")
        return target_rot @ rot_x_180()

    if ROTATION_MODE == 2:
        print("[Rotation mode] target_rot @ RotY(180)")
        return target_rot @ rot_y_180()

    if ROTATION_MODE == 3:
        print("[Rotation mode] target_rot @ RotZ(180)")
        return target_rot @ rot_z_180()

    raise ValueError(f"Invalid ROTATION_MODE: {ROTATION_MODE}")


def get_charge_port_axis(model, data):
    """
    charge_port_center -> charge_port_axis_site 방향을 충전구 삽입 방향으로 정의한다.
    """
    center_id = model.site("charge_port_center").id
    axis_id = model.site("charge_port_axis_site").id

    center = data.site_xpos[center_id].copy()
    axis_point = data.site_xpos[axis_id].copy()

    axis = axis_point - center
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-9:
        raise RuntimeError("charge_port_axis_site is too close to charge_port_center.")

    axis = axis / axis_norm

    return center, axis_point, axis


def print_axis_alignment(model, data, ee_site_id, label):
    """
    현재 EE site의 local 축들과 charge_port_axis의 dot product를 출력한다.
    dot product가 1에 가까운 축이 충전구 삽입 방향과 같은 방향이다.
    """
    mujoco.mj_forward(model, data)

    port_center, port_axis_point, port_axis = get_charge_port_axis(model, data)

    ee_pos = data.site_xpos[ee_site_id].copy()
    ee_rot = data.site_xmat[ee_site_id].reshape(3, 3).copy()

    ee_x = ee_rot[:, 0]
    ee_y = ee_rot[:, 1]
    ee_z = ee_rot[:, 2]

    print(f"\n==== Axis Alignment Check: {label} ====")
    print("port_center    :", port_center)
    print("port_axis_point:", port_axis_point)
    print("port_axis      :", port_axis)
    print("ee_pos         :", ee_pos)

    print("EE +X dot port_axis:", np.dot(ee_x, port_axis))
    print("EE -X dot port_axis:", np.dot(-ee_x, port_axis))
    print("EE +Y dot port_axis:", np.dot(ee_y, port_axis))
    print("EE -Y dot port_axis:", np.dot(-ee_y, port_axis))
    print("EE +Z dot port_axis:", np.dot(ee_z, port_axis))
    print("EE -Z dot port_axis:", np.dot(-ee_z, port_axis))


def solve_ik_to_pose(
    model,
    data,
    ee_site_id,
    target_pos,
    target_rot,
    joint_ids,
    label,
    max_iters=500,
    step_size=0.3,
):
    print(f"\n==== Solve IK: {label} ====")
    print("Target pos:", target_pos)
    print_matrix("Target rot:", target_rot)

    q_target, final_pos, final_rot, final_pos_error, final_rot_error = solve_pose_ik(
        model=model,
        data=data,
        ee_site_id=ee_site_id,
        target_pos=target_pos,
        target_rot=target_rot,
        joint_ids=joint_ids,
        max_iters=max_iters,
        pos_tol=1e-4,
        rot_tol=1e-3,
        step_size=step_size,
        damping=1e-4,
        rot_weight=0.5,
        verbose=True,
    )

    print("q_target:", q_target)
    print("Final EE pos:", final_pos)
    print("Final pos error:", final_pos_error)
    print("Final rot error:", final_rot_error)
    print_matrix("Final EE rot:", final_rot)

    print_axis_alignment(model, data, ee_site_id, label)

    return q_target


def interpolate_q(q_from, q_to, alpha):
    alpha = np.clip(alpha, 0.0, 1.0)
    return (1.0 - alpha) * q_from + alpha * q_to


def main():
    project_root = get_project_root()
    xml_path = os.path.join(project_root, "models", "scene.xml")

    print("Loading:", xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print_model_info(model)

    ee_site_name = "attachment_site"
    approach_site_name = "charge_port_approach"
    insert_site_name = "charge_port_insert"

    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    ee_site_id = model.site(ee_site_name).id
    joint_ids = get_joint_ids(model, joint_names)
    qpos_ids = [model.jnt_qposadr[joint_id] for joint_id in joint_ids]

    if model.nu < len(joint_ids):
        raise RuntimeError(
            f"Actuator number is smaller than joint size. "
            f"model.nu={model.nu}, len(joint_ids)={len(joint_ids)}"
        )

    # ------------------------------------------------------------
    # 1. 초기 상태에서 충전구 GT site pose 읽기
    # ------------------------------------------------------------
    reset_model(model, data)

    _, ee_start_pos, ee_start_rot = get_site_pose(model, data, ee_site_name)
    _, approach_pos, approach_rot_xml = get_site_pose(model, data, approach_site_name)
    _, insert_pos, insert_rot_xml = get_site_pose(model, data, insert_site_name)

    approach_rot = apply_rotation_debug_mode(approach_rot_xml)
    insert_rot = apply_rotation_debug_mode(insert_rot_xml)

    print("\n==== Initial Info ====")
    print("EE site name       :", ee_site_name)
    print("Approach site name :", approach_site_name)
    print("Insert site name   :", insert_site_name)
    print("Start EE pos       :", ee_start_pos)
    print("Approach pos       :", approach_pos)
    print("Insert pos         :", insert_pos)
    print("Distance EE -> approach:", np.linalg.norm(approach_pos - ee_start_pos))
    print("Distance approach -> insert:", np.linalg.norm(insert_pos - approach_pos))

    print_matrix("\nStart EE rot:", ee_start_rot)
    print_matrix("\nApproach rot XML:", approach_rot_xml)
    print_matrix("\nApproach rot used:", approach_rot)
    print_matrix("\nInsert rot XML:", insert_rot_xml)
    print_matrix("\nInsert rot used:", insert_rot)

    print_axis_alignment(model, data, ee_site_id, "initial")

    # ------------------------------------------------------------
    # 2. 1차 목표: charge_port_approach pose로 이동
    # ------------------------------------------------------------
    q_approach = solve_ik_to_pose(
        model=model,
        data=data,
        ee_site_id=ee_site_id,
        target_pos=approach_pos,
        target_rot=approach_rot,
        joint_ids=joint_ids,
        label="charge_port_approach",
        max_iters=500,
        step_size=0.3,
    )

    # ------------------------------------------------------------
    # 3. 2차 목표: approach -> insert 방향으로 삽입
    # ------------------------------------------------------------
    num_insert_waypoints = 20
    q_insert_waypoints = []

    print("\n==== Build Insertion Waypoints ====")

    # 중요:
    # 여기서는 position은 approach -> insert로 보간하고,
    # rotation은 approach_rot을 계속 유지한다.
    # 즉, 같은 자세로 충전구 축 방향을 따라 삽입한다.
    for idx, alpha in enumerate(np.linspace(1.0 / num_insert_waypoints, 1.0, num_insert_waypoints)):
        waypoint_pos = interpolate_q(approach_pos, insert_pos, alpha)

        q_waypoint = solve_ik_to_pose(
            model=model,
            data=data,
            ee_site_id=ee_site_id,
            target_pos=waypoint_pos,
            target_rot=approach_rot,
            joint_ids=joint_ids,
            label=f"insert_waypoint_{idx + 1:02d}",
            max_iters=300,
            step_size=0.2,
        )

        q_insert_waypoints.append(q_waypoint)

    q_plan = [q_approach] + q_insert_waypoints

    print("\n==== Motion Plan ====")
    print("Number of q targets:", len(q_plan))
    print("1 approach target +", len(q_insert_waypoints), "insertion waypoints")

    # ------------------------------------------------------------
    # 4. Viewer에서 실제 actuator 제어
    # ------------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nViewer started.")
        print("Charging sequence:")
        print("  Phase 1: move to charge_port_approach")
        print("  Phase 2: move to charge_port_insert")
        print(f"  ROTATION_MODE = {ROTATION_MODE}")

        reset_model(model, data)

        q_start = data.qpos[qpos_ids].copy()

        approach_duration = 4.0
        insert_total_duration = 4.0
        insert_segment_duration = insert_total_duration / num_insert_waypoints

        segment_targets = q_plan
        segment_durations = [approach_duration] + [
            insert_segment_duration for _ in range(num_insert_waypoints)
        ]

        start_time = time.time()

        while viewer.is_running():
            elapsed = time.time() - start_time

            cumulative_time = 0.0
            q_cmd = segment_targets[-1]

            for seg_idx, duration in enumerate(segment_durations):
                seg_start = cumulative_time
                seg_end = cumulative_time + duration

                if seg_start <= elapsed < seg_end:
                    alpha = (elapsed - seg_start) / duration

                    if seg_idx == 0:
                        q_from = q_start
                    else:
                        q_from = segment_targets[seg_idx - 1]

                    q_to = segment_targets[seg_idx]
                    q_cmd = interpolate_q(q_from, q_to, alpha)
                    break

                cumulative_time = seg_end

            data.ctrl[: len(q_cmd)] = q_cmd

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(0.002)


if __name__ == "__main__":
    main()