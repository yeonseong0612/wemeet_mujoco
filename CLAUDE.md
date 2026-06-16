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
| 스크립트 | `scripts/` |

**카메라 (scene.xml)**

- `overview_camera` — 전체 조망
- `cam_port` — 충전구 정면, FoundationPose/RGB-D 입력 전용
- `stereo_left` / `stereo_right` — baseline 0.12 m (비교 baseline 보존)

**충전구 site (charge_port_frame body 로컬 좌표)**

| site | pos (object frame) | 용도 |
|---|---|---|
| `charge_port_center` | `0 -0.02 0.115` | 소켓 입구 중심(커넥터 돌출부 앞면) |
| `charge_port_axis_site` | `0 -0.02 0.015` | 삽입축 기준점(소켓 안쪽 -Z 0.10 m) |
| `charge_port_approach` | `0 -0.02 0.265` | 접근 웨이포인트(소켓 바깥 +Z 0.15 m) |
| `charge_port_insert` | `0 -0.02 0.095` | 착좌 타겟(소켓 안쪽 -Z 0.02 m) |

---

## 2. 좌표계 & 자세 변환 규약 (반드시 준수)

FoundationPose는 object(mesh) 프레임의 6DoF pose `ob_in_cam`(OpenCV 규약)을 출력한다.

```
target_in_cam  = ob_in_cam   @ target_in_obj   # target_in_obj = site 고정 오프셋(4×4)
target_in_base = cam_in_base @ target_in_cam
→ IK 입력 (src/ik.py: solve_pose_ik)
```

- **`cam_in_base`**: 시뮬 = scene.xml cam_port 배치 GT. 실환경 = hand-eye calibration 필요(Phase 2).
- **K**: fovy + 해상도에서 계산(`intrinsics_from_fovy`). sim/real 교체는 config 레이어에서.
- EE IK 대상 site: `charger_tip_site` (충전건 tip, custom_ee +X 방향 0.25 m 돌출).
- `R_TIP_IN_OBJ`: tip +X(총신 forward) → port +Z(삽입 방향), tip +Z(총구 up) → port +Y(위).

---

## 3. Depth 소스 방침

| 경로 | 상태 | 용도 |
|---|---|---|
| `cam_port` GT depth (`enable_depth_rendering()`) | **운용 중** | FoundationPose 입력 (Phase 1) |
| `stereo_left`/`stereo_right` (FastACV 등) | 보존 | RGB-D vs 스테레오 ablation (논문 비교) |

> **스테레오 경로는 삭제하지 않는다.** `test_stereo_depth.py`를 ablation baseline으로 유지.

---

## 4. 완료된 작업 ✅

### Phase 1 — 시뮬레이션 파이프라인 (완료)

| # | 작업 | 산출물 |
|---|---|---|
| A | charge_port_center 정의 조사·확정 | scene.xml 코멘트, `scripts/debug/` 분석 스크립트 |
| B | RGB-D 시각화 검증 스크립트 | `scripts/test_rgbd_view.py` → `outputs/rgbd_view_test.png` |
| C | depth 소스 일원화 확인 | `test_foundation_pose.py`가 이미 GT depth 사용 (변경 없음) |
| D | FoundationPose end-to-end 실행 | `outputs/foundation_pose_inputs.png`, `foundation_pose_result.png` |
| E | FP 추정 pose → 삽입 타겟 → IK 연결 | `scripts/test_integrated_pipeline.py` |
| F | GT vs FP 정확도 평가 (N=10 pose sweep) | `outputs/integrated_eval.json`, `integrated_eval.png` |

**현재 달성 정확도 (N=10, seed=0, GT depth 사용)**

| 지표 | mean | max |
|---|---|---|
| `ob_in_cam` 위치 오차 | **0.78 mm** | 1.08 mm |
| `ob_in_cam` 회전 오차 | **0.58°** | 1.12° |
| insert landing 위치 오차 | **0.88 mm** | 1.67 mm |
| insert landing 회전 오차 | **0.59°** | 1.08° |
| approach landing 위치 오차 | **2.14 mm** | 3.71 mm |

---

## 5. 남은 과제 🔲

### Phase 1 추가 검증 (시뮬)

- [ ] **N 증가 평가**: `--n 50 --seed 42` 등으로 통계 강건성 확인
- [ ] **RGB-D vs 스테레오 depth ablation**: `test_stereo_depth.py` 결과와 정량 비교 → 논문 표 작성
- [ ] **충전구 자세 범위 확장**: `MAX_TILT_DEG` 를 15-20° 로 올려 FP 강건성 재평가
- [ ] **view_scene.py** — GUI 뷰어 동작 확인 (WSL2 디스플레이 설정 필요 시)

### Phase 2 — 실환경 전환 (미착수)

- [ ] **D3. 실환경 segmentation mask 획득**: SAM2 또는 색상 기반 초기 mask → FoundationPose `register()`
- [ ] **D4. 실환경 RGB-D 카메라 선정**: RealSense D435 / L515 등 기종 확정 → 내부 K, depth 특성 문서화
- [ ] **Hand-eye calibration**: `cam_in_base` 실측 (eye-in-hand 또는 eye-to-hand)
- [ ] **실환경 K calibration**: `intrinsics_from_fovy` 대체, 실카메라 내부 파라미터 로드
- [ ] **실환경 end-to-end 파이프라인**: Phase 1 스크립트를 실카메라 입력으로 교체·검증

---

## 6. 미결 결정 사항 (사용자 확인 필요)

| ID | 질문 | 권장 |
|---|---|---|
| D2 | 스테레오/FastACV 경로: 제거 vs 보존? | **보존(ablation용)** |
| D3 | 실환경 초기 mask 획득 방법? | SAM2 또는 깊이 임계 기반 |
| D4 | 실환경 RGB-D 카메라 기종? | 미확정 |

---

## 7. 스크립트 실행 방법

```bash
conda activate wemeet

# RGB-D 시각화 검증
python scripts/test_rgbd_view.py
# → outputs/rgbd_view_test.png

# FoundationPose 단일 추정 (CUDA 필요)
python scripts/test_foundation_pose.py
# → outputs/foundation_pose_inputs.png, foundation_pose_result.png

# 통합 파이프라인 평가 (FP → IK, N=10)
python scripts/test_integrated_pipeline.py --n 10 --no-gui
# → outputs/integrated_eval.json, integrated_eval.png

# GUI 포함 실행 (WSL2 X11 필요)
python scripts/test_integrated_pipeline.py --n 5
```

---

## 8. 설계 원칙 (코드 수정 시 준수)

1. `scene.xml`의 `<compiler angle="radian"/>` 때문에 euler/quat 값을 radian으로 작성.
2. depth 렌더는 `enable_depth_rendering()` — 이미 metric meter. 수동 near/far 변환 금지.
3. FP 추정 pose이든 GT pose이든 **동일한 `target_in_obj` 오프셋**을 사용.
4. `solve_position_ik` 선행 → `solve_pose_ik` 후속(2단계 IK)으로 수렴 안정성 확보.
5. `charge_port_center`: 소켓 입구(커넥터 돌출부 앞면) 중심 — mesh com/centroid 아님.
