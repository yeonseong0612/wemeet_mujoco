"""
cluster_poses segfault 진단 스크립트.

의심 원인:
  1. dtype 불일치: rot_grid는 np.linalg.inv 결과(float64),
     cluster_poses는 vectorMatrix4f(=Matrix4f, float32)를 기대함.
  2. symmetry_tfs가 빈 벡터일 경우의 동작 확인.
  3. pybind11이 float64 배열을 Matrix4f로 변환할 때 no-copy path를 타면
     float64 포인터를 float32*로 재해석 → UB/segfault.

실행 방법:
  conda activate pytorch
  python scripts/debug/debug_cluster_poses.py

  또는 gdb로 실행:
  gdb -ex run -ex bt --args python scripts/debug/debug_cluster_poses.py
"""

import sys
import os
import faulthandler
import traceback

# faulthandler: SIGSEGV 발생 시 Python 레벨 스택트레이스 출력
faulthandler.enable()

# mycpp.so 경로 추가
MYCPP_BUILD = os.path.join(
    os.path.dirname(__file__), "../../ext/FoundationPose/mycpp/build"
)
sys.path.insert(0, os.path.abspath(MYCPP_BUILD))

import numpy as np

try:
    import mycpp
    print(f"[OK] mycpp imported from {mycpp.__file__}")
except ImportError as e:
    print(f"[FAIL] mycpp import error: {e}")
    sys.exit(1)

# cluster_poses 존재 확인
assert hasattr(mycpp, "cluster_poses"), "cluster_poses not found in mycpp"
print("[OK] cluster_poses attribute found")


# ─── 헬퍼 ────────────────────────────────────────────────────────────────────

def make_identity_grid(n: int, dtype) -> np.ndarray:
    """n×4×4 단위행렬 배열을 지정 dtype으로 반환."""
    grid = np.tile(np.eye(4, dtype=dtype), (n, 1, 1))
    # 각 행렬에 작은 회전/위치 오프셋을 줘서 완전히 동일하지 않게 함
    for i in range(n):
        angle = i * 0.01  # 라디안
        grid[i, 0, 0] = np.cos(angle)
        grid[i, 0, 1] = -np.sin(angle)
        grid[i, 1, 0] = np.sin(angle)
        grid[i, 1, 1] = np.cos(angle)
    return grid


def run_case(label: str, poses_in: np.ndarray, sym_tfs: np.ndarray):
    print(f"\n{'='*60}")
    print(f"Case: {label}")
    print(f"  poses_in  shape={poses_in.shape}  dtype={poses_in.dtype}  C-contig={poses_in.flags['C_CONTIGUOUS']}")
    print(f"  sym_tfs   shape={sym_tfs.shape}   dtype={sym_tfs.dtype}  C-contig={sym_tfs.flags['C_CONTIGUOUS']}")
    try:
        result = mycpp.cluster_poses(30, 99999, poses_in, sym_tfs)
        result_arr = np.asarray(result)
        print(f"  [OK] output shape={result_arr.shape}")
    except Exception as e:
        print(f"  [EXCEPTION] {type(e).__name__}: {e}")
        traceback.print_exc()
    # segfault가 나면 이 줄은 출력되지 않는다
    print(f"  (survived)")


# ─── 테스트 케이스 ────────────────────────────────────────────────────────────

# sym_tfs: 단위행렬 1개 (float32) — estimater.py 기본값과 동일
sym_f32 = np.eye(4, dtype=np.float32)[np.newaxis]   # (1,4,4) float32
sym_f64 = np.eye(4, dtype=np.float64)[np.newaxis]   # (1,4,4) float64
sym_empty_f32 = np.empty((0, 4, 4), dtype=np.float32)  # 빈 배열

# ── Case 1: float32 poses + float32 sym  (예상: OK) ──────────────────────────
run_case(
    "float32 poses + float32 sym (정상 케이스)",
    make_identity_grid(10, np.float32),
    sym_f32,
)

# ── Case 2: float64 poses + float32 sym  (의심 케이스) ───────────────────────
run_case(
    "float64 poses + float32 sym (rot_grid가 np.linalg.inv 결과인 실제 케이스)",
    make_identity_grid(10, np.float64),
    sym_f32,
)

# ── Case 3: float64 poses + float64 sym  ─────────────────────────────────────
run_case(
    "float64 poses + float64 sym",
    make_identity_grid(10, np.float64),
    sym_f64,
)

# ── Case 4: float32 poses + 빈 sym_tfs  ──────────────────────────────────────
# symmetry_tfs가 비어있으면 내부 루프가 돌지 않아 isnew=True가 유지됨
# → segfault 없이 모든 pose가 출력 되어야 함
run_case(
    "float32 poses + empty sym_tfs",
    make_identity_grid(10, np.float32),
    sym_empty_f32,
)

# ── Case 5: float64 poses + 빈 sym_tfs  ──────────────────────────────────────
run_case(
    "float64 poses + empty sym_tfs",
    make_identity_grid(10, np.float64),
    sym_empty_f32,
)

# ── Case 6: 실제 규모(252개) + float64  ──────────────────────────────────────
run_case(
    "252-pose float64 (실제 rot_grid 규모)",
    make_identity_grid(252, np.float64),
    sym_f32,
)

# ── Case 7: float32로 명시적 캐스팅 후 호출  ─────────────────────────────────
poses_f64 = make_identity_grid(252, np.float64)
poses_f32_cast = np.ascontiguousarray(poses_f64, dtype=np.float32)
run_case(
    "252-pose float64→float32 캐스팅 후 호출 (권장 fix 검증)",
    poses_f32_cast,
    sym_f32,
)

print("\n" + "="*60)
print("모든 케이스 완료 — segfault 없이 여기까지 출력됐다면")
print("dtype 변환으로 문제가 해결됨.")
