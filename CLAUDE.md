# wemeet_mujoco — 프로젝트 현황 & 작업 지침

EV 충전 로봇의 MuJoCo 시뮬레이션. RGB-D(FoundationPose)로 충전구 6DoF pose를
추정하고, IK로 UR10e 로봇 팔을 충전 커넥터 삽입 위치까지 이동시킨다.

---

## 1. 환경

| 항목 | 값 |
|---|---|
| 런타임 | WSL2 / **conda env `wemeet`** / Python 3.10 / PyTorch 2.4.0 / CUDA 12.1 |
| GPU | NVIDIA RTX 4060 Ti 8GB (sm_89) |
| 프로젝트 루트 | `/home/yskim/projects/wemeet_mujoco` |

```bash
conda activate wemeet
```

---

## 2. 폴더 구조

```
wemeet_mujoco/
├── models/                      MuJoCo 모델
│   ├── scene.xml                메인 씬: 충전구 body/site, 카메라 4종, include ur10e.xml
│   ├── assets/                  씬 레벨 mesh — Port.obj(충전구), 2022_KIA_EV6.obj(차체)
│   └── robots/ur10e/
│       ├── ur10e.xml            UR10e 로봇 + custom_ee(충전건) body + cam_eih
│       └── assets/              로봇/EE mesh (base, shoulder, ..., EE.obj)
├── src/
│   ├── ik.py                    position/pose IK (damped least squares)
│   └── depth_noise.py           RealSense D435 depth/RGB 노이즈 모델
├── scripts/                     실행 스크립트 (아래 §6 참조)
│   └── debug/                   1회성 분석/디버그 스크립트 (mesh 분석, pose 진단 등)
├── ext/
│   ├── FoundationPose/          6DoF pose estimator (빌드 완료, weights 적재됨)
│   ├── fastacv.py               FastACVNet+ 스테레오 depth 모델 wrapper
│   └── mujoco_menagerie/        참고용 MuJoCo 모델 모음
├── checkpoints/                 FastACVNet+ 사전학습 가중치 (stereo ablation용)
├── outputs/                     스크립트 산출물(이미지/JSON) — git 미추적(.gitignore)
└── requirements.txt
```

---

## 3. scene.xml 구성

### 카메라

| 이름 | 종류 | 용도 |
|---|---|---|
| `overview_camera` | 고정 전체 조망 | 디버깅 뷰어 |
| `cam_port` | 고정 외부(eye-to-hand) | ablation baseline, GT depth(노이즈 없음) |
| `cam_eih` | EE 장착(eye-in-hand), `custom_ee` body 내부 (`models/robots/ur10e/ur10e.xml`) | FoundationPose 실운용 입력 — RealSense D435 사양(해상도 848×480, fovy 42.5°) |
| `stereo_left` / `stereo_right` | 고정 스테레오(baseline 0.12 m) | FastACVNet+ stereo depth ablation |

`cam_eih`는 `custom_ee` 좌표 `pos="0 0.04 0.11" euler="1.30 0 0"` — 총신(+Y) 위쪽에서 전방을 ~16° 하향으로 내려다본다.

### 충전구 site (`charge_port_frame` body 로컬 좌표)

| site | pos (object frame) | 용도 |
|---|---|---|
| `charge_port_center` | `0 -0.02 0` | 소켓 입구 중심 (Port.obj 실측 결합면, mesh AABB Z `[-0.115, +0.0087]`) |
| `charge_port_axis_site` | `0 -0.02 -0.10` | 삽입축 기준점(소켓 안쪽 -Z 0.10 m) |
| `charge_port_approach` | `0 -0.02 0.15` | 접근 웨이포인트(소켓 바깥 +Z 0.15 m) |
| `charge_port_insert` | `0 -0.02 -0.02` | 착좌 타겟(소켓 안쪽 -Z 0.02 m) |

네 site 모두 `rgba` **alpha=0** — 렌더 시각 마커는 숨김 처리됨.
충전건 tip(`charger_tip_site`)은 `models/robots/ur10e/ur10e.xml`의 `custom_ee` body 안에 있으며,
custom_ee 프레임 **+Y 방향 약 0.204 m** 위치(`pos="0 0.2041 0.012"`).

---

## 4. 좌표계 & 자세 변환 규약 (반드시 준수)

FoundationPose는 object(mesh) 프레임의 6DoF pose `ob_in_cam`(OpenCV 규약)을 출력한다.

```
target_in_cam   = ob_in_cam   @ target_in_obj   # target_in_obj = site 고정 오프셋(4×4)
target_in_world = cam_in_world @ target_in_cam
→ IK 입력 (src/ik.py: solve_pose_ik)
```

- `cam_in_world`는 **호출 시점마다 동적으로 계산**한다(`pos_mat_to_T(data.cam_xpos[cam_id], data.cam_xmat[cam_id])`).
- eye-in-hand로 캡처할 때는 **영상 렌더와 카메라 pose 획득을 같은 로봇 자세에서** 수행해야 한다.
- EE IK 대상 site: `charger_tip_site`. `R_TIP_IN_OBJ`: tip +X → port -Z(삽입 방향), tip +Z → port +Y(위).

---

## 5. Depth 소스 & 노이즈 모델

| 경로 | 용도 |
|---|---|
| `cam_eih` GT depth + 노이즈 모델 | FoundationPose 실운용 입력 시뮬레이션 (eye-in-hand) |
| `cam_port` GT depth (노이즈 없음) | ablation baseline (eye-to-hand) |
| `stereo_left`/`stereo_right` + FastACVNet+ | RGB-D vs 스테레오 depth ablation — 보존 |

`NOISE_PRESETS = {"none", "low", "medium", "high"}` (medium = D435 실측 기준).
모든 노이즈는 GT depth 원본을 보존하며 새 배열을 반환한다.

---

## 6. 스크립트

| 스크립트 | 역할 |
|---|---|
| `scripts/test_rgbd_view.py` | overview + cam_port RGB/GT depth 시각화 |
| `scripts/test_eih_view.py` | overview + cam_eih RGB/depth 3분할, 관찰 pose FOV 검증 |
| `scripts/test_seg_mask.py` | seg_mask 추출 + 노이즈 주입 데모 |
| `scripts/test_foundation_pose.py` | GT depth → FoundationPose 단일 pose 추정 |
| `scripts/test_integrated_pipeline.py` | FP → IK 통합 파이프라인 + 정확도 평가 |
| `scripts/compare_eval.py` | 평가 결과 JSON 비교표/그래프 |
| `scripts/test_stereo_depth.py` | 스테레오 + FastACVNet+ depth 추정 (ablation) |
| `scripts/test_collision.py` | 충전건↔충전구 충돌 판정 테스트 (4케이스 삽입 궤적 sweep) |

### `test_integrated_pipeline.py` 주요 인자

```bash
python scripts/test_integrated_pipeline.py \
  --n 50 --seed 42 \
  --camera cam_port|cam_eih \
  --noise none|low|medium|high \
  --no-gui \
  --out outputs/result.json
```

---

## 7. 현재 정확도 (N=50, seed=42, mean)

| run | 카메라 | 노이즈 | ob_in_cam trans (mm) | ob_in_cam rot (°) | insert landing (mm) |
|---|---|---|---|---|---|
| A | cam_port | none | 0.94 | 0.84 | 1.00 |
| B | cam_eih | none | 0.77 | 0.70 | 0.58 |
| C | cam_eih | low | 0.90 | 0.77 | 0.74 |
| D | cam_eih | medium | 0.94 | 0.76 | 0.75 |
| E | cam_eih | high | 1.01 | 0.70 | 0.80 |

---

## 8. 충돌 테스트 계획 🆕

### 배경 & 방침

현재 충전건(`ee_visual`) / 충전구(`charge_port_visual`)는 모두 **visual-only** (contype=0).
실제 contact physics가 없어 팔이 충전구를 뚫고 지나가도 감지하지 못한다.

Port.obj는 소켓 구멍이 있는 오목한(concave) 형태라 mesh 그대로 collision geom으로
쓰면 MuJoCo의 convex hull 변환으로 구멍이 막혀 삽입 판정이 불가능해진다.
→ **primitive(cylinder/box)로 소켓 형상을 근사**한다.

### Collision Geom 설계

#### 충전구 쪽 (scene.xml 또는 Port body 안)

소켓 입구를 cylinder로 근사:

```xml
<!-- 소켓 입구 테두리 — 충전건이 삽입 전 먼저 닿는 면 -->
<geom name="port_socket_rim"
      type="cylinder"
      size="0.028 0.005"        <!-- 반지름 2.8cm, 두께 5mm -->
      pos="0 -0.02 0"           <!-- charge_port_center와 동일 위치 -->
      contype="2" conaffinity="2"
      rgba="0 0.5 1 0.3"        <!-- 디버깅용 반투명 파랑 -->
      group="2"/>               <!-- 기본 렌더에서 숨김, 디버그 시 group 표시 -->

<!-- 소켓 내부 통로 — 삽입 성공 시 충전건이 통과하는 공간(충돌 없어야 함) -->
<!-- 내부는 geom 없음: 입구 rim만 막고 안쪽은 뚫린 것처럼 처리 -->
```

#### 충전건 tip 쪽 (ur10e.xml custom_ee body 안)

```xml
<!-- 충전건 tip collision — charger_tip_site 위치 기준 -->
<geom name="ee_tip_collision"
      type="cylinder"
      size="0.020 0.015"        <!-- 반지름 2.0cm, 길이 3cm (tip 돌출부) -->
      pos="0 0.2041 0.012"      <!-- charger_tip_site와 동일 -->
      euler="1.5708 0 0"        <!-- +Y 방향 정렬 (총신 방향) -->
      contype="2" conaffinity="2"
      rgba="1 0.3 0 0.3"        <!-- 디버깅용 반투명 주황 -->
      group="2"/>
```

> **contype/conaffinity = 2**: 로봇 링크(contype=1)·바닥(contype=1)과는 충돌하지 않고,
> 충전건↔충전구 쌍끼리만 충돌 감지. 기존 동작 무영향.

> ⚠️ **실제 size 값은 Port.obj / EE.obj mesh 치수를 먼저 확인 후 결정한다.**
> Claude Code는 수정 전 두 mesh의 AABB를 보고하고 size 후보를 제안한 뒤 사용자 확인 후 적용.

### 테스트 시나리오 (`scripts/test_collision.py`)

| 케이스 | 설명 | 기대 결과 |
|---|---|---|
| 정렬 삽입 | GT pose 기반 IK → insert target까지 이동 | ncon=0 (접촉 없이 통과) |
| 오정렬 삽입 (5°) | 고의 오프셋 5° tilt 후 삽입 시도 | ncon>0, force 발생 |
| 오정렬 삽입 (10°) | 고의 오프셋 10° tilt | ncon>0, force 더 큼 |
| FP 추정 삽입 | FP 추정 pose → IK → 삽입 | 정렬과 유사 (ncon=0 또는 최소) |

### 측정 지표

```python
# 삽입 완료 시점 contact 수 & force
n_contact = data.ncon
for i in range(data.ncon):
    c = data.contact[i]
    if involves(c, "ee_tip_collision", "port_socket_rim"):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        # force[0:3] = 힘 (N), force[3:6] = 토크 (N·m)
```

### 판정 기준

| 지표 | 성공 기준 | 실패 기준 |
|---|---|---|
| `data.ncon` (충전건↔충전구) | 0 (삽입 성공) | > 0 (간섭 발생) |
| `hit_steps` (삽입 궤적 sweep 중 접촉 스텝) | 0/40 | > 0/40 |
| insert landing 위치 오차 | ≤ 2 mm | > 5 mm |

> ⚠️ force 절대값은 kinematic IK 모드에서 비물리적 — ncon/hit_steps만 신뢰.

### 산출물

- `outputs/collision_test_results.json` — 케이스별 ncon / force / landing 오차
- `outputs/collision_test_vis.png` — 4케이스 overview 스크린샷 (contact 표시)

---

## 9. 남은 과제 🔲

### Phase 1-C: 충돌 테스트 ✅ 완료

- [x] **[Col-1]** Port.obj / EE.obj AABB 조사 — socket X반폭 ≈30mm, tip plug 실효반지름 ≈21mm
- [x] **[Col-2]** `scene.xml` 충전구에 소켓 입구 rim 추가 — solid 1개 대신 **8기둥 ring** `port_rim_0~7`
      (입구평면 z=0, 반지름 0.028m 원주, 각 cylinder `size="0.007 0.005"`). solid disk는 정렬삽입도
      막아 ncon=0 불가 → 중앙이 뚫린 ring으로 근사(안쪽반지름 21mm, tip r=20mm와 1mm 여유).
- [x] **[Col-3]** `ur10e.xml` custom_ee에 `ee_tip_collision` cylinder (`size="0.020 0.015"`, +Y정렬) 추가
- [x] **[Col-4]** `contype/conaffinity=2` 검증 — 비트마스크상 로봇링크/바닥(class1)과 무충돌, rim↔tip만 신규
- [x] **[Col-5]** `scripts/test_collision.py` — 4케이스. **단일 deep-insert 정적평가는 plug가 rim을 이미
      통과해 tilt도 무접촉** → approach→insert **궤적 sweep**(40스텝)으로 진입 중 접촉을 집계하도록 구현.
- [x] **[Col-6]** 결과 보고 — `outputs/collision_test_results.json` + `collision_test_vis.png`

#### 결과 (seed=42, cam_port, port pos≈(0.11,0.81,0.672))

| 케이스 | max ncon | hit steps | peak F | landing | 판정 |
|---|---|---|---|---|---|
| 1 정렬 | **0** | 0/40 | 0 N | 0.02 mm | PASS (통과) |
| 2 5° tilt | 7 | 6/40 | 416 N | 0.02 mm | CONTACT |
| 3 10° tilt | 9 | 8/40 | 324 N | 0.02 mm | CONTACT |
| 4 FP 추정 | **0** | 0/40 | 0 N | 0.79 mm | PASS (통과) |

> ⚠️ **force 절대값은 비물리적**(kinematic IK가 충돌을 무시하고 plug를 관통시켜 솔버가 침투깊이×강성으로
> 산출). 신뢰 가능한 판별자는 **ncon / hit_steps**: 정렬·FP=0, tilt>0 으로 깨끗이 분리됨.
> peak F가 5°>10° 인 건 단일스텝 침투의 노이즈 — 궤적 누적 Fsum은 10°(7508) > 5°(1450)로 단조.
> 정밀 force가 필요하면 위치제어 quasi-static 삽입 시뮬레이션(actuator+mj_step)으로 후속 확장 가능.

### Phase 1-A: 기존 과제

- [ ] **[Tilt-1]** `MAX_TILT_DEG` 10° → 15~20°로 확장 후 조건 B/D 기준 재평가
- [ ] **[Paper-1]** 논문 Table 작성 — 조건 A~E 비교표 (§7 수치 기준)
- [ ] **[Paper-2]** "depth noise affects translation only" 분석 서술
- [ ] **RGB-D vs 스테레오 depth ablation** — `test_stereo_depth.py` 결과와 정량 비교

---

## 10. 설계 원칙 (코드 수정 시 반드시 준수)

1. `scene.xml`의 `<compiler angle="radian"/>` — euler/quat 값을 radian으로 작성.
2. depth 렌더는 `enable_depth_rendering()` — 이미 metric meter. 수동 near/far 변환 금지.
3. FP 추정 pose이든 GT pose이든 **동일한 `target_in_obj` 오프셋**을 사용.
4. `solve_position_ik` 선행 → `solve_pose_ik` 후속(2단계 IK)으로 수렴 안정성 확보.
5. `charge_port_center`: 소켓 입구(커넥터 돌출부 앞면) 중심 — mesh com/centroid 아님.
6. `cam_in_world`는 **호출 시점마다 동적으로 계산** — 캐시/하드코딩 금지.
7. 노이즈는 `src/depth_noise.py` 함수를 통해서만 주입 — GT depth 원본은 항상 보존.
8. 충전구 site는 alpha=0으로 숨겨져 있다 — 디버깅 시만 임시로 alpha 올리고 원복.
9. `scene.xml`/`ur10e.xml` 구조 변경 전 반드시 해당 body/geom 현황을 먼저 보고하고 사용자 확인 후 수정.
10. `outputs/`는 `.gitignore` 대상 — 산출물은 커밋하지 않는다.
11. collision geom은 `contype=2 conaffinity=2`로 충전건↔충전구 쌍끼리만 감지 — 기존 로봇 동작에 영향 없게 한다.