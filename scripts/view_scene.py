import os
import time

import mujoco
import mujoco.viewer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

xml_path = os.path.join(PROJECT_ROOT, "models", "scene.xml")

print(f"[INFO] Loading XML: {xml_path}")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("[INFO] Model loaded successfully.")
print(f"[INFO] Number of bodies: {model.nbody}")
print(f"[INFO] Number of geoms: {model.ngeom}")
print(f"[INFO] Number of sites: {model.nsite}")

print("\n[INFO] Bodies:")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"  {i}: {name}")

print("\n[INFO] Sites:")
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"  {i}: {name}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25
    viewer.cam.lookat[:] = [0.0, 0.0, 0.8]

    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)