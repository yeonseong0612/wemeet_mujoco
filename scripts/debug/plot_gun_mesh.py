"""충전건 메시를 custom_ee 프레임에서 3개 직교 투영으로 그려 총구 끝을 눈으로 확인."""
import os
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/yskim/projects/wemeet_mujoco"
MESH = os.path.join(ROOT, "models/robots/ur10e/assets/EE.obj")
OUT = os.path.join(ROOT, "outputs/gun_mesh_frames.png")


def main():
    mesh = trimesh.load(MESH, force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64) * 0.001
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # euler 1.5708 about X
    v = (Rx @ v.T).T + np.array([0.08, 0.1, -0.03])

    # 서브샘플
    idx = np.random.default_rng(0).choice(len(v), size=min(8000, len(v)), replace=False)
    p = v[idx]

    # 후보 끝점들
    cands = {
        "Ymin/-Z tube tip": np.array([0.0, -0.027, -0.073]),
        "Ymax body end":    np.array([0.0,  0.208,  0.012]),
        "old tip(0.25,0,0)": np.array([0.25, 0.0, 0.0]),
    }
    colors = {"Ymin/-Z tube tip": "red", "Ymax body end": "green",
              "old tip(0.25,0,0)": "magenta"}

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    planes = [("X", "Y", 0, 1), ("Y", "Z", 1, 2), ("X", "Z", 0, 2)]
    for k, (la, lb, ia, ib) in enumerate(planes):
        ax[k].scatter(p[:, ia], p[:, ib], s=1, c="0.6", alpha=0.4)
        for name, c in cands.items():
            ax[k].scatter(c[ia], c[ib], s=120, c=colors[name],
                          marker="*", label=name, zorder=5)
        ax[k].set_xlabel(f"custom_ee {la}")
        ax[k].set_ylabel(f"custom_ee {lb}")
        ax[k].set_title(f"{la}-{lb} projection")
        ax[k].axis("equal")
        ax[k].grid(True, alpha=0.3)
    ax[0].legend(fontsize=8, loc="upper right")

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=110)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
