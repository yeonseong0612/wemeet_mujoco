import mujoco

import cv2

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

XML_PATH = PROJECT_ROOT / "models" / "basic_scene.xml"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = mujoco.MjModel.from_xml_path(str(XML_PATH))

data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=480, width=640)

# 몇 step 진행

for _ in range(10):

    mujoco.mj_step(model, data)

# main_camera 시점으로 렌더링

renderer.update_scene(data, camera="main_camera")

rgb = renderer.render()

# OpenCV는 BGR 기준이라 RGB -> BGR 변환

rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

cv2.imwrite(str(OUTPUT_DIR / "main_camera.png"), rgb_bgr)

print("saved:", OUTPUT_DIR / "main_camera.png")