"""Port.obj 만 독립 렌더 — 소켓 입구(핀/구멍)가 body 어느 축을 향하는지 확정.
body 축: R=+X G=+Y B=+Z. site: center(녹), approach(주황), insert(파랑), axis(빨강구)."""
import os
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yskim/projects/wemeet_mujoco"
XML = f"""
<mujoco>
  <compiler angle="radian"/>
  <visual><global offwidth="640" offheight="520"/></visual>
  <asset>
    <mesh name="port" file="{ROOT}/models/assets/Port.obj" scale="0.001 0.001 0.001"/>
  </asset>
  <worldbody>
    <light pos="0.3 0.3 0.5" dir="-1 -1 -1.5"/>
    <light pos="-0.3 -0.3 0.5" dir="1 1 -1.5"/>
    <body name="port" pos="0 0 0">
      <geom type="mesh" mesh="port" rgba="0.35 0.35 0.35 1"/>
      <site name="bx" pos="0.06 0 0" size="0.05 0.003 0.003" type="box" rgba="1 0 0 1"/>
      <site name="by" pos="0 0.06 0" size="0.003 0.05 0.003" type="box" rgba="0 1 0 1"/>
      <site name="bz" pos="0 0 0.06" size="0.003 0.003 0.05" type="box" rgba="0 0 1 1"/>
      <!-- scene.xml 의 포트 site (body-local pos 동일) -->
      <site name="center"   pos="0 -0.015 0.0075"  size="0.012" rgba="0 1 0 1"/>
      <site name="approach" pos="0 -0.015 0.1575"  size="0.012" rgba="1 0.6 0 1"/>
      <site name="insert"   pos="0 -0.015 -0.0125" size="0.010" rgba="0 0.4 1 1"/>
      <site name="axis"     pos="0 -0.015 -0.0925" size="0.010" rgba="1 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
opt = mujoco.MjvOption(); opt.sitegroup[:] = 1

renderer = mujoco.Renderer(model, height=520, width=640)
fig, ax = plt.subplots(2, 2, figsize=(13, 11))
# +Z 쪽과 -Z 쪽을 각각 정면으로 보는 뷰 포함
views = [(0, 0), (180, 0), (90, 88), (90, -88)]
names = ["az0 (+X side)", "az180 (-X side)", "top +Z down", "bottom -Z up"]
for k, ((az, el), nm) in enumerate(zip(views, names)):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, -0.01, -0.03]
    cam.distance, cam.azimuth, cam.elevation = 0.4, az, el
    renderer.update_scene(data, camera=cam, scene_option=opt)
    a = ax[k // 2, k % 2]
    a.imshow(renderer.render()); a.axis("off")
    a.set_title(f"{nm}  (R+X G+Y B+Z; green=center, orange=approach)")
out = os.path.join(ROOT, "outputs/port_alone.png")
plt.tight_layout(); plt.savefig(out, dpi=110)
print("saved:", out)
