"""최소 재현 스크립트 — gdb/faulthandler용."""
import sys, os, faulthandler
faulthandler.enable()

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../ext/FoundationPose/mycpp/build")
))

import numpy as np
import mycpp

# 가장 단순한 케이스: 2개의 float32 단위행렬, sym_tfs도 단위행렬 1개
poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
sym   = np.eye(4, dtype=np.float32)[np.newaxis]

print(f"poses shape={poses.shape} dtype={poses.dtype}")
print(f"sym   shape={sym.shape}   dtype={sym.dtype}")
print("calling cluster_poses...")
result = mycpp.cluster_poses(30, 99999, poses, sym)
print(f"result: {len(result)} poses")
