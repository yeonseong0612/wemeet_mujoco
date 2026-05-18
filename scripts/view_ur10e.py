import mujoco
import mujoco.viewer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

XML_PATH = (
    PROJECT_ROOT
    / "ext"
    / "mujoco_menagerie"
    / "universal_robots_ur10e"
    / "scene.xml"
)

print("Loading:", XML_PATH)

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

print("Loaded UR10e model")
print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)
print("nbody:", model.nbody)
print("njnt:", model.njnt)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()