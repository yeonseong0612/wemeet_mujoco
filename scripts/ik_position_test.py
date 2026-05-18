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

def print_matrix(name, mat):

    print(name)

    for row in mat:

        print("  ", row)


def main():
    project_root = get_project_root()
    xml_path = os.path.join(project_root, "models", "scene.xml")

    print("Loading:", xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("Loaded model")
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("nsite:", model.nsite)

    ee_site_name = "attachment_site"
    target_site_name = "ik_target"

    ee_site_id = model.site(ee_site_name).id
    target_site_id = model.site(target_site_name).id

    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    joint_ids = get_joint_ids(model, joint_names)

    # 초기화
    mujoco.mj_resetData(model, data)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    mujoco.mj_forward(model, data)

    start_pos = data.site_xpos[ee_site_id].copy()
    start_rot = data.site_xmat[ee_site_id].reshape(3, 3).copy()

    target_pos = data.site_xpos[target_site_id].copy()
    target_rot = data.site_xmat[target_site_id].reshape(3, 3).copy()

    print("\n==== IK Target Info ====")
    print("EE site name     :", ee_site_name)
    print("Target site name :", target_site_name)
    print("Start EE pos     :", start_pos)
    print("Target pos       :", target_pos)
    print("Initial pos error:", np.linalg.norm(target_pos - start_pos))

    print_matrix("\nStart EE rot:", start_rot)
    print_matrix("\nTarget rot:", target_rot)

    q_target, final_pos, final_rot, final_pos_error, final_rot_error = solve_pose_ik(
        model=model,
        data=data,
        ee_site_id=ee_site_id,
        target_pos=target_pos,
        target_rot=target_rot,
        joint_ids=joint_ids,
        max_iters=500,
        pos_tol=1e-4,
        rot_tol=1e-3,
        step_size=0.3,
        damping=1e-4,
        rot_weight=0.5,
    )

    print("\n==== IK Result ====")
    print("q_target:", q_target)
    print("Final EE pos:", final_pos)
    print("Final pos error:", final_pos_error)
    print("Final rot error:", final_rot_error)

    print_matrix("\nFinal EE rot:", final_rot)

    if model.nu < len(q_target):
        raise RuntimeError(
            f"Actuator number is smaller than target q size. "
            f"model.nu={model.nu}, len(q_target)={len(q_target)}"
        )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nViewer started.")
        print("The robot should move toward ik_target with target orientation.")

        # 실제 actuator로 따라가는지 보기 위해 다시 초기화
        mujoco.mj_resetData(model, data)

        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)

        mujoco.mj_forward(model, data)

        while viewer.is_running():
            data.ctrl[: len(q_target)] = q_target

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(0.002)


if __name__ == "__main__":
    main()