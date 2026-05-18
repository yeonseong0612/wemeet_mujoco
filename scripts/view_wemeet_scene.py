import os

import mujoco
import mujoco.viewer

def main():
    scripts_dir = os.path.dirname(os.path.dirname(__file__))
    proj_root = os.path.dirname(scripts_dir)
    xml_path = os.path.join(proj_root, "wemeet_mujoco", "models", "scene.xml")

    print("Loding: ", xml_path)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("Loaded wemeet_scene")
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
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()