import mujoco
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

XML_PATH = (
    PROJECT_ROOT
    / "ext"
    / "mujoco_menagerie"
    / "universal_robots_ur10e"
    / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

# keyframe이 있으면 첫 번째 keyframe으로 초기화
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
else:
    mujoco.mj_resetData(model, data)

mujoco.mj_forward(model, data)

site_id = model.site("attachment_site").id

print("attachment_site position:")
print(data.site_xpos[site_id])

print("\nattachment_site rotation matrix:")
print(data.site_xmat[site_id].reshape(3, 3))