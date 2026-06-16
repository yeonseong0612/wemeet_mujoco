import os, mujoco
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "models", "scene.xml"))
print("nkey (keyframes):", m.nkey)
print("nu (actuators):", m.nu)
print("attachment_site id:", m.site("attachment_site").id)
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", e)
