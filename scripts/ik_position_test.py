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
    _, approach_pos, approach_rot = get_site_pose(model, data, approach_site_name)
    _, insert_pos, insert_rot = get_site_pose(model, data, insert_site_name)

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
    print_matrix("\nApproach rot:", approach_rot)
    print_matrix("\nInsert rot:", insert_rot)

    # ------------------------------------------------------------
    # 2. 1차 목표: 충전구 위쪽 approach pose로 이동
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
    # 3. 2차 목표: approach -> insert 방향으로 수직 삽입
    #    여기서는 직선 삽입을 위해 Cartesian waypoint를 여러 개 만든다.
    #    자세는 approach_rot을 유지한다.
    # ------------------------------------------------------------
    num_insert_waypoints = 20
    q_insert_waypoints = []

    print("\n==== Build Insertion Waypoints ====")

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
        print("  Phase 2: move downward to charge_port_insert")

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