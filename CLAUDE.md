# wemeet_mujoco — 프로젝트 현황 & 작업 지침

---

## 1. 환경 & 주요 경로

| 항목 | 값 |
|---|---|
| 런타임 | WSL2 / **conda env `wemeet`** / Python 3.10 / PyTorch 2.4.0 / CUDA 12.1 |
| GPU | NVIDIA RTX 4060 Ti 8GB (sm_89) |
| 프로젝트 루트 | `/home/yskim/projects/wemeet_mujoco` |

**주요 파일**

| 역할 | 경로 |
|---|---|
| Scene | `models/scene.xml` |
| 로봇 | `models/robots/ur10e/ur10e.xml` |
| 충전구 mesh | `models/assets/Port.obj` (mm → m, scale 0.001) |
| 차체 mesh | `models/assets/2022_KIA_EV6.obj` |
| FoundationPose | `ext/FoundationPose/` (빌드 완료, weights 적재 완료) |
| IK | `src/ik.py` |
| 노이즈 유틸 | `src/depth_noise.py` ← **신규 생성 예정** |
| 스크립트 | `scripts/` |

**카메라 (scene.xml)**

| 이름 | 종류 | 용도 | 상태 |
|---|---|---|---|
| `overview_camera` | 고정 전체 조망 | 디버깅 뷰어 | 유지 |
| `cam_port` | 고정 외부(eye-to-hand) | ablation baseline / GT depth 비교 | **보존 (ablation용)** |
| `cam_eih` | EE 장착 (eye-in-hand) | FoundationPose 실운용 입력 | **신규 추가 예정** |
| `stereo_left` / `stereo_right` | 고정 스테레오 | stereo depth ablation | 보존 |

> ⚠️ `cam_port`는 삭제하지 않는다. ablation 비교(eye-to-hand vs eye-in-hand)에 사용.

**충전구 site (charge_port_frame body 로컬 좌표)**

| site | pos (object frame) | 용도 |
|---|---|---|
| `charge_port_center` | `0 -0.02 0` | 소켓 입구 중심(Port.obj 실측 결합면, mesh AABB Z [-0.115, +0.0087]) |
| `charge_port_axis_site` | `0 -0.02 -0.10` | 삽입축 기준점(소켓 안쪽 -Z 0.10 m) |
| `charge_port_approach` | `0 -0.02 0.15` | 접근 웨이포인트(소켓 바깥 +Z 0.15 m) |
| `charge_port_insert` | `0 -0.02 -0.02` | 착좌 타겟(소켓 안쪽 -Z 0.02 m) |

---

## 2. 좌표계 & 자세 변환 규약 (반드시 준수)

FoundationPose는 object(mesh) 프레임의 6DoF pose `ob_in_cam`(OpenCV 규약)을 출력한다.

```
target_in_cam  = ob_in_cam   @ target_in_obj   # target_in_obj = site 고정 오프셋(4×4)
target_in_base = cam_in_base @ target_in_cam
→ IK 입력 (src/ik.py: solve_pose_ik)
```

### Eye-in-Hand 전환 후 달라지는 점

- `cam_in_base`가 **고정값이 아니라 매 스텝 동적으로 계산**되어야 한다.

```python
# MuJoCo에서 실시간으로 cam_in_base 획득
cam_pos  = data.cam_xpos[cam_id]          # (3,)  world frame
cam_mat  = data.cam_xmat[cam_id].reshape(3,3)  # world frame
cam_in_base = pos_mat_to_T(cam_pos, cam_mat)   # (4,4)
```

- EE IK 대상 site: `charger_tip_site` — custom_ee 프레임에서 **+Y 방향 약 0.204 m** 돌출(pos `0 0.2041 0.012`, ur10e.xml). 커넥터 결합면 법선 = custom_ee +Y. (※ tip **site 로컬 프레임**은 xyaxes 로 +X=forward 로 정의됨 — 아래 R_TIP_IN_OBJ 의 "tip +X" 는 이 site 로컬축 기준.)
- `R_TIP_IN_OBJ`: tip(site 로컬) +X(총신 forward) → port **-Z**(소켓 안으로 삽입 방향), tip +Z(총구 up) → port +Y(위) — 변경 없음.

---

## 3. Eye-in-Hand 카메라 설정 🆕

### 카메라 사양 — Intel RealSense D435 기준

| 파라미터 | 값 | 비고 |
|---|---|---|
| 해상도 | 848 × 480 | depth/RGB 동일 |
| fovy | 42.5° | 수직 FOV (depth stream 기준) |
| 근거리 클립 | 0.1 m | near 0.1 m |
| 원거리 클립 | 3.0 m | far 3.0 m |
| 물리 무게 | 72 g | UR10e 페이로드 여유 충분 |

### scene.xml 추가 내용

```xml
<!-- custom_ee body 안에 삽입 -->
<body name="custom_ee" ...>
  ...
  <!-- Eye-in-Hand RGB-D 카메라: 충전건 위쪽, 소켓 방향을 내려다보는 각도 -->
  <camera name="cam_eih"
          pos="0 0.04 0.08"
          euler="1.9 0 0"
          fovy="42.5"/>
</body>
```

> **결정 사항:** `pos` / `euler` 값은 실제 EE geometry를 확인한 뒤 조정한다.
> Claude Code는 수정 전 `custom_ee` body의 현재 geometry를 `scene.xml`에서 먼저 보고하고,
> 위치·방향 후보 2~3개를 제안한 뒤 사용자 확인 후 적용한다.

### 초기 관찰 Pose 요건

Eye-in-Hand이므로 팔이 `charge_port_approach` 위치에 있을 때 `cam_eih` FOV 안에 충전구 전체가 들어와야 한다.

검증 방법:
1. 초기 q 설정 후 `cam_eih`로 렌더링
2. 충전구 AABB 전체가 이미지 내에 있는지 확인
3. FoundationPose 입력으로 사용 가능한 해상도인지 확인 (충전구 단축 ≥ 80px 권장)

### 뷰어 동시 표시

```python
# scripts/test_eih_view.py (신규 생성 예정)
rgb_overview = render_rgb(model, data, cam="overview_camera", W=640, H=480)
rgb_eih      = render_rgb(model, data, cam="cam_eih",         W=848, H=480)
depth_eih    = render_depth(model, data, cam="cam_eih",       W=848, H=480)

# 좌: 전체 조망 / 우: EiH RGB / 우하: EiH depth 컬러맵
panel = make_debug_panel(rgb_overview, rgb_eih, depth_eih)
cv2.imshow("EiH Debug", panel)
```

---

## 4. Depth 소스 방침

| 경로 | 상태 | 용도 |
|---|---|---|
| `cam_eih` GT depth + 노이즈 모델 | **신규 운용 예정** | FoundationPose 실운용 입력 |
| `cam_port` GT depth (노이즈 없음) | **ablation baseline** | eye-to-hand vs eye-in-hand 비교 |
| `stereo_left`/`stereo_right` (FastACV 등) | 보존 | RGB-D vs stereo ablation |

> **스테레오 경로는 삭제하지 않는다.** `test_stereo_depth.py`를 ablation baseline으로 유지.

---

## 5. 노이즈 모델 — Intel RealSense D435 기준 🆕

`src/depth_noise.py` 에 구현한다. 모든 노이즈는 **파이프라인 외부에서 주입**하여
GT depth와 noisy depth를 독립적으로 보관할 수 있게 한다.

### 노이즈 구성 요소

#### 5-1. Gaussian Depth Noise (거리 비례)
D435 실측 기반: σ ≈ 0.001 + 0.0005 × d (단위: m)

```python
def gaussian_depth_noise(depth: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    depth: (H, W) float32, metric meter
    scale: 노이즈 강도 배율 (1.0 = D435 기준)
    """
    sigma = (0.001 + 0.0005 * depth) * scale
    noise = np.random.normal(0, sigma, depth.shape).astype(np.float32)
    return np.clip(depth + noise, 0, None)
```

#### 5-2. Depth Hole (반사 / 검정 플라스틱)
충전구처럼 dark/reflective한 표면에서 depth 무효값 발생을 모사.

```python
def apply_depth_holes(depth: np.ndarray,
                      seg_mask: np.ndarray,
                      hole_rate: float = 0.20,
                      edge_thickness: int = 3) -> np.ndarray:
    """
    seg_mask: 충전구 영역 binary mask (H, W) bool
    hole_rate: 충전구 영역 내 무효값 비율 (D435 dark surface 기준 0.15~0.30)
    edge_thickness: 물체 경계 depth mixing 두께 (px)
    """
    depth_out = depth.copy()
    # 1) 충전구 영역 내 랜덤 hole
    port_pixels = np.where(seg_mask)
    n_holes = int(len(port_pixels[0]) * hole_rate)
    idx = np.random.choice(len(port_pixels[0]), n_holes, replace=False)
    depth_out[port_pixels[0][idx], port_pixels[1][idx]] = 0.0
    # 2) 경계 edge noise
    edge = cv2.dilate(seg_mask.astype(np.uint8), np.ones((edge_thickness, edge_thickness))) \
         - seg_mask.astype(np.uint8)
    depth_out[edge.astype(bool)] *= np.random.uniform(0.5, 1.0,
                                    depth_out[edge.astype(bool)].shape)
    return depth_out
```

#### 5-3. Quantization
D435 depth 해상도: 1mm (0.001 m)

```python
def quantize_depth(depth: np.ndarray, resolution: float = 0.001) -> np.ndarray:
    return (np.round(depth / resolution) * resolution).astype(np.float32)
```

#### 5-4. RGB Noise (선택)

```python
def apply_rgb_noise(rgb: np.ndarray,
                    gaussian_std: float = 5.0,
                    brightness_jitter: float = 0.1) -> np.ndarray:
    noisy = rgb.astype(np.float32)
    noisy += np.random.normal(0, gaussian_std, noisy.shape)
    noisy *= (1.0 + np.random.uniform(-brightness_jitter, brightness_jitter))
    return np.clip(noisy, 0, 255).astype(np.uint8)
```

### 노이즈 레벨 프리셋 (ablation용)

```python
NOISE_PRESETS = {
    "none": {
        "depth_gaussian_scale": 0.0,
        "hole_rate":            0.00,
        "quantize":             False,
        "rgb_gaussian_std":     0.0,
    },
    "low": {
        "depth_gaussian_scale": 0.5,   # D435 절반
        "hole_rate":            0.05,
        "quantize":             True,
        "rgb_gaussian_std":     3.0,
    },
    "medium": {                         # D435 실측 기준
        "depth_gaussian_scale": 1.0,
        "hole_rate":            0.20,
        "quantize":             True,
        "rgb_gaussian_std":     5.0,
    },
    "high": {
        "depth_gaussian_scale": 2.0,   # D435 열화/원거리
        "hole_rate":            0.35,
        "quantize":             True,
        "rgb_gaussian_std":     10.0,
    },
}
```

> **seg_mask 획득**: MuJoCo segmentation renderer(`mj_render` + `mjCAT_GEOM`)로
> 충전구 geom id에 해당하는 픽셀 mask를 추출한다.
> Claude Code는 `scene.xml`에서 충전구 geom 이름을 먼저 확인하고 보고한다.

---

## 6. 평가 확장 계획 🆕

### 평가 매트릭스 (논문 Table 구성)

| 조건 | 카메라 | 노이즈 | N | 목적 |
|---|---|---|---|---|
| A (기존 baseline) | eye-to-hand (`cam_port`) | none | 50 | 이전 결과 재현 |
| B | eye-in-hand (`cam_eih`) | none | 50 | EiH 구조 효과 |
| C | eye-in-hand | low | 50 | 노이즈 민감도 |
| D | eye-in-hand | medium (D435 기준) | 50 | 실환경 근사 |
| E | eye-in-hand | high | 50 | 강건성 한계 |

### 실행 명령 (계획)

```bash
conda activate wemeet

# 조건 A: eye-to-hand, no noise
python scripts/test_integrated_pipeline.py \
  --n 50 --seed 42 --camera cam_port --noise none --no-gui \
  --out outputs/eval_A_eth_none.json

# 조건 D: eye-in-hand, D435 medium noise
python scripts/test_integrated_pipeline.py \
  --n 50 --seed 42 --camera cam_eih --noise medium --no-gui \
  --out outputs/eval_D_eih_medium.json
```

> ⚠️ `--camera` / `--noise` 인자는 `test_integrated_pipeline.py` 수정 시 함께 추가한다.

---

## 7. 완료된 작업 ✅

### 시뮬레이션 파이프라인 (완료)

| # | 작업 | 산출물 |
|---|---|---|
| A | charge_port_center 정의 조사·확정 | scene.xml 코멘트, `scripts/debug/` 분석 스크립트 |
| B | RGB-D 시각화 검증 스크립트 | `scripts/test_rgbd_view.py` → `outputs/rgbd_view_test.png` |
| C | depth 소스 일원화 확인 | `test_foundation_pose.py`가 이미 GT depth 사용 (변경 없음) |
| D | FoundationPose end-to-end 실행 | `outputs/foundation_pose_inputs.png`, `foundation_pose_result.png` |
| E | FP 추정 pose → 삽입 타겟 → IK 연결 | `scripts/test_integrated_pipeline.py` |
| F | GT vs FP 정확도 평가 (N=10 pose sweep) | `outputs/integrated_eval.json`, `integrated_eval.png` |

**현재 달성 정확도 (N=10, seed=0, GT depth, eye-to-hand `cam_port`)**

| 지표 | mean | max |
|---|---|---|
| `ob_in_cam` 위치 오차 | **0.94 mm** | 2.46 mm |
| `ob_in_cam` 회전 오차 | **0.82°** | 2.51° |
| insert landing 위치 오차 | **1.68 mm** | 4.70 mm |
| insert landing 회전 오차 | **0.81°** | 2.48° |
| approach landing 위치 오차 | **3.80 mm** | 11.49 mm |

> FoundationPose refine 단계가 stochastic하여 재실행마다 수치가 변동한다.
> 회귀 비교 시 같은 머신·같은 실행에서 나온 값끼리만 비교할 것.

---

## 8. 남은 과제 🔲

### Phase 1-B: Eye-in-Hand + 노이즈 (현재 진행)

- [ ] **[EiH-1]** `scene.xml` — `custom_ee` body geometry 조사 & 보고 (수정 전)
- [ ] **[EiH-2]** `cam_eih` 카메라 추가 (충전건 위쪽, D435 파라미터)
- [ ] **[EiH-3]** 초기 관찰 pose 설정 — `cam_eih` FOV 안에 충전구 전체 포함 확인
- [ ] **[EiH-4]** `scripts/test_eih_view.py` — overview + EiH RGB + EiH depth 3분할 뷰어
- [ ] **[EiH-5]** `test_integrated_pipeline.py` — 동적 `cam_in_base` 계산으로 전환
- [ ] **[Noise-1]** `src/depth_noise.py` 구현 (섹션 5 기준)
- [ ] **[Noise-2]** MuJoCo segmentation renderer로 충전구 seg_mask 추출 확인
- [ ] **[Noise-3]** `test_integrated_pipeline.py` — `--camera` / `--noise` 인자 추가
- [ ] **[Eval-1]** 평가 매트릭스 A~E 전체 실행 (N=50, seed=42)
- [ ] **[Eval-2]** 결과 비교표 생성 스크립트 `scripts/compare_eval.py`

### Phase 1-A: 기존 과제

- [ ] **RGB-D vs 스테레오 depth ablation**: `test_stereo_depth.py` 결과와 정량 비교 → 논문 표
- [ ] **충전구 자세 범위 확장**: `MAX_TILT_DEG` 10° → 15~20° 로 FP 강건성 재평가
- [ ] **site 좌표 갱신 후 회귀 재실행**: `charge_port_center` 재정의 → 정확도 표 갱신
- [ ] **view_scene.py** GUI 뷰어 동작 확인 (WSL2 디스플레이)

---

## 9. 미결 결정 사항

| ID | 질문 | 결정 |
|---|---|---|
| D1 | eye-in-hand 카메라 위치/방향 | 충전건 위쪽. 정확한 pos/euler는 EiH-1 보고 후 확정 |
| D2 | 스테레오/FastACV 경로 | **보존(ablation용)** |
| D3 | 타겟 RGB-D 센서 | **Intel RealSense D435** (섹션 5 파라미터 적용) |
| D4 | depth hole 적용 | **적용** (seg_mask 기반, hole_rate=0.20 medium 기준) |

---

## 10. 스크립트 실행 방법

```bash
conda activate wemeet

# RGB-D 시각화 검증
python scripts/test_rgbd_view.py
# → outputs/rgbd_view_test.png

# EiH 뷰어 (신규)
python scripts/test_eih_view.py
# → outputs/eih_view_test.png (3분할: overview / EiH RGB / EiH depth)

# FoundationPose 단일 추정 (CUDA 필요)
python scripts/test_foundation_pose.py
# → outputs/foundation_pose_inputs.png, foundation_pose_result.png

# 통합 파이프라인 평가
python scripts/test_integrated_pipeline.py --n 10 --no-gui
# → outputs/integrated_eval.json, integrated_eval.png

# 평가 매트릭스 (노이즈 레벨 지정)
python scripts/test_integrated_pipeline.py \
  --n 50 --seed 42 --camera cam_eih --noise medium --no-gui \
  --out outputs/eval_D_eih_medium.json

# 결과 비교표 (신규)
python scripts/compare_eval.py \
  outputs/eval_A_eth_none.json \
  outputs/eval_D_eih_medium.json
```

---

## 11. 설계 원칙 (코드 수정 시 준수)

1. `scene.xml`의 `<compiler angle="radian"/>` — euler/quat 값을 radian으로 작성.
2. depth 렌더는 `enable_depth_rendering()` — 이미 metric meter. 수동 near/far 변환 금지.
3. FP 추정 pose이든 GT pose이든 **동일한 `target_in_obj` 오프셋**을 사용.
4. `solve_position_ik` 선행 → `solve_pose_ik` 후속(2단계 IK)으로 수렴 안정성 확보.
5. `charge_port_center`: 소켓 입구(커넥터 돌출부 앞면) 중심 — mesh com/centroid 아님.
6. `cam_in_base`는 eye-in-hand 전환 후 **반드시 동적으로 계산** — 하드코딩 금지.
7. 노이즈는 `src/depth_noise.py` 함수를 통해서만 주입 — GT depth 원본은 항상 보존.
8. `scene.xml` 구조 변경 전 반드시 해당 body/geom 현황을 먼저 보고하고 사용자 확인 후 수정.