# wemeet_mujoco — 프로젝트 현황 & 작업 지침

EV 충전 로봇의 MuJoCo 시뮬레이션. RGB-D(FoundationPose)로 충전구 6DoF pose를
추정하고, IK로 UR10e 로봇 팔을 충전 커넥터 삽입 위치까지 이동시킨다.

현재 시뮬레이션은 두 버전으로 운영된다 (§3 참조):

- **V1 — 충전구 단독**: 인식·제어 파이프라인 정확도와 충돌 검증을 처음 확립한 베이스라인. **완료.**
- **V2 — 차량 결합(Hummer EV)**: 충전구가 실제 차체에 결합된 형태로, 차체 occlusion·로봇 도달성 등
  실차에 가까운 조건을 반영한 현재 메인 작업. **진행 중.**

V1에서 검증한 인식–제어–충돌 로직(좌표 변환 규약, IK 2단계, depth 노이즈, collision geom 설계)은
V2에서도 그대로 재사용한다. V2에서 새로 부딪힌 문제(차체에 의한 가려짐, body 중첩에 따른 좌표
변환, 로봇 도달 높이)만 별도로 해결했다.

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
│   ├── scene.xml                메인 씬: 현재 V2(차량 결합) 구성이 기본 — §3.2 참조
│   ├── assets/                  씬 레벨 mesh
│   │   ├── hummer_body.obj          V2 — Hummer EV 차체(외장)
│   │   ├── hummer_charge_port.obj   V2 — 차체에 실측 결합된 충전구(현재 사용 중)
│   │   ├── 2024_gmc_hummer_ev_suv_w_inlet.glb  V2 — 차체 원본 소스(glb), obj는 여기서 export
│   │   ├── Port.obj                 V1 — 충전구 단독 mesh(보존, V2엔 미사용)
│   │   └── 2022_KIA_EV6.obj, Untitled.obj  과거 검토용, 현재 씬에서 미사용(레거시)
│   └── robots/ur10e/
│       ├── ur10e.xml            UR10e 로봇 + custom_ee(충전건) body + cam_eih + ee_tip_collision
│       └── assets/              로봇/EE mesh (base, shoulder, ..., EE.obj)
├── src/
│   ├── ik.py                    position/pose IK (damped least squares)
│   └── depth_noise.py           RealSense D435 depth/RGB 노이즈 모델
├── scripts/                     실행 스크립트 (아래 §6 참조)
│   └── debug/                   1회성 분석/디버그 스크립트 — 대부분 V1 Port.obj 기하 조사용
│                                 (analyze_port_mesh.py, find_socket_face.py, render_port_alone.py 등)
├── docs/
│   └── WeMeet_결과보고서.md     중간 보고서 초안 — **V1(KIA EV6) 기준으로 작성, V2 반영 전** (별도 작업 필요)
├── ext/
│   ├── FoundationPose/          6DoF pose estimator (빌드 완료, weights 적재됨)
│   ├── fastacv.py               FastACVNet+ 스테레오 depth 모델 wrapper
│   └── mujoco_menagerie/        참고용 MuJoCo 모델 모음
├── checkpoints/                 FastACVNet+ 사전학습 가중치 (stereo ablation용)
├── outputs/                     스크립트 산출물(이미지/JSON) — git 미추적(.gitignore)
│   ├── only_port/                   V1(충전구 단독) 평가·충돌테스트 결과 아카이브
│   └── final/                       V2(차량 결합) 정렬 검증 스크린샷
└── requirements.txt
```

---

## 3. 시뮬레이션 버전: V1(충전구 단독) ↔ V2(차량 결합)

### 3.1 V1 — 충전구 단독 (완료)

충전구(`Port.obj`)가 `charge_port_frame` body로 world에 직접 떠 있는 가장 단순한 구성. 인식–제어
파이프라인의 정확도(§7)와 충돌 검증 로직(§8) 자체를 처음 확립하기 위한 베이스라인으로 사용했다.
결과는 `outputs/only_port/`에 보존되어 있다.

### 3.2 V2 — 차량 결합(Hummer EV) (진행 중, 현재 메인)

충전구가 실제 차량(Hummer EV) 차체에 결합된 위치·자세 그대로 시뮬레이션에 들어간 구성. `scene.xml`의
`hummer_vehicle` body 아래에 `charge_port_frame`이 자식으로 중첩되어 있다. V1 대비 세 가지 현실적
제약이 새로 생겼고, 각각 다음과 같이 풀었다.

1. **충전구가 차체 패널에 가려 카메라에서 안 보이는 문제** — 차체에 매립된 충전구는 로봇이 접근하는
   각도에 따라 차체 외판에 시야가 가려질 수 있다. 단순히 "포트가 보이는지"만 판정하던 기존 가시성
   기준(픽셀 수 임계값)으로는 차체에 살짝 가려진 흐릿한 관측까지 통과시켜 평가를 왜곡했다.
   → 가시 픽셀 수 임계값을 대폭 상향(800 → 30,000px)하여 "차체에 가리지 않은 정면에 가까운 관측"만
   평가에 채택하도록 했다(경험적으로 양호한 뷰는 65k~72k px, 가려진 뷰는 20k px 미만으로 뚜렷이 갈림).

2. **충전구 body가 차량 body 아래 중첩되며 world 좌표를 직접 못 쓰는 문제** — V1에서는 충전구 pose를
   world 좌표로 바로 지정할 수 있었지만, V2에서는 `charge_port_frame`의 부모가 `hummer_vehicle`이라
   MuJoCo가 body_pos/body_quat를 항상 "부모 frame 기준"으로 해석한다. world 좌표를 그대로 대입하면
   차량이 움직인 것처럼 잘못 배치된다.
   → 매 호출 시점에 부모(`hummer_vehicle`)의 world pose를 읽어, world 좌표를 부모 기준 로컬 좌표로
   역변환한 뒤 대입하는 방식으로 해결했다. (참고: §4의 "동적 계산" 원칙과 같은 결의 문제.)

3. **충전구가 차체 결합 높이(약 1.17m)로 올라가며 로봇이 닿지 않는 문제** — UR10e를 바닥에 그대로
   두면 V2의 충전구 높이까지 팔이 닿지 않는다.
   → 로봇 베이스를 받침대(pedestal)로 0.45m 들어올렸다(`ur10e.xml`의 `base` body `pos="0 0 0.45"`,
   `scene.xml`의 `robot_pedestal`은 순수 시각적 받침대).

> 차체-충전구 mesh 정렬은 적용 전에 배치/회전 스크린샷으로 시각 검증했다
> (`outputs/final/overview_placement_check.png`, `overview_rotation_check.png`, `port_roll_check.png`).

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
- **V2 전용**: `charge_port_frame`이 `hummer_vehicle`의 자식으로 중첩되어 있어, 충전구의 world pose를
  설정/조회할 때는 부모(`hummer_vehicle`) world pose를 먼저 동적으로 읽고 그 기준으로 변환해야 한다
  (§3.2의 문제 2). V1(충전구가 world 최상위)에서는 이 변환이 불필요했다.

### 충전구 site (`charge_port_frame` body 로컬 좌표, V1/V2 공통 정의)

| site | pos (object frame) | 용도 |
|---|---|---|
| `charge_port_center` | 소켓 입구 중심(mesh 원점=입구 중심으로 export, V1/V2 mesh가 달라 표현값은 다름) | 소켓 입구 중심 |
| `charge_port_axis_site` | 입구 안쪽 -Z 0.10 m | 삽입축 기준점 |
| `charge_port_approach` | 입구 바깥 +Z 0.15 m | 접근 웨이포인트 |
| `charge_port_insert` | 입구 안쪽 -Z 0.02 m | 착좌 타겟 |

네 site 모두 `rgba` **alpha=0** — 렌더 시각 마커는 숨김 처리됨.
충전건 tip(`charger_tip_site`)은 `models/robots/ur10e/ur10e.xml`의 `custom_ee` body 안에 있으며,
custom_ee 프레임 **+Y 방향 약 0.204 m** 위치(`pos="0 0.2041 0.012"`).

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
| `scripts/test_integrated_pipeline.py` | FP → IK 통합 파이프라인 + 정확도 평가 (현재 V2 mesh/좌표 기준) |
| `scripts/compare_eval.py` | 평가 결과 JSON 비교표/그래프 |
| `scripts/test_stereo_depth.py` | 스테레오 + FastACVNet+ depth 추정 (ablation) |
| `scripts/test_collision.py` | 충전건↔충전구 충돌 판정 테스트 (4케이스 삽입 궤적 sweep, 현재 V2 mesh/좌표 기준) |

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

## 7. 인식·제어 정확도 평가

### 7.1 V1 결과 (완료) — N=50, seed=42, mean

카메라 종류(외부 고정 vs eye-in-hand)와 depth 노이즈 수준을 조합한 5개 조건(A~E)으로, 인식부터
삽입까지 전 과정의 정확도를 정량화했다.

| run | 카메라 | 노이즈 | ob_in_cam trans (mm) | ob_in_cam rot (°) | insert landing (mm) |
|---|---|---|---|---|---|
| A | cam_port | none | 0.94 | 0.84 | 1.00 |
| B | cam_eih | none | 0.77 | 0.70 | 0.58 |
| C | cam_eih | low | 0.90 | 0.77 | 0.74 |
| D | cam_eih | medium | 0.94 | 0.76 | 0.75 |
| E | cam_eih | high | 1.01 | 0.70 | 0.80 |

핵심 발견:
- **eye-in-hand 전환으로 삽입 정확도 42% 향상** (A 1.00mm → B 0.58mm). 충전구에 가까이서 인식할수록
  정확도가 올라간다는 것을 확인했다.
- **실환경 노이즈(D435 medium)에서도 서브밀리미터 정확도 유지** (0.75mm).
- **회전 오차는 노이즈 수준과 무관하게 0.70~0.84°로 일정** — depth 노이즈는 거리(병진) 추정에만
  영향을 주고, 자세(회전) 추정은 주로 RGB 형상 정보에 의존하기 때문으로 해석된다.

### 7.2 V2 평가 — 미실시 (남은 과제, §9)

차체 결합 후 occlusion 등 새 제약이 생겨, 위 A~E 매트릭스를 V2 조건으로 다시 돌려 동일하게
서브밀리미터 정확도가 유지되는지 확인이 필요하다. 아직 N=50 규모로 재실행되지 않았다.

---

## 8. 충돌 검증

충전건과 충전구가 모두 visual-only(contype=0)라 실제 contact physics가 없어, 팔이 충전구를 뚫고
지나가도 감지하지 못하는 문제가 있었다. 또한 충전구(Port) mesh는 소켓 구멍이 있는 오목한 형태라,
mesh 그대로 collision geom으로 쓰면 MuJoCo가 convex hull로 변환하면서 구멍이 막혀 삽입 판정 자체가
불가능해진다.

**해결**: 소켓 형상을 mesh가 아닌 8개의 작은 원통(cylinder)으로 이루어진 고리(ring)로 근사했다.
중앙이 비어 있어 정렬된 커넥터는 고리 사이 구멍을 통과하고, 어긋난 커넥터만 기둥에 걸리도록
설계했다(고리 안쪽 반지름 21mm, 커넥터 실효 반지름 20mm로 1mm 여유). 충돌 감지는
`contype/conaffinity=2`로 충전건 tip ↔ 충전구 고리 쌍에만 한정해, 기존 로봇·바닥 충돌 동작에는
영향이 없게 분리했다. 판정은 단일 시점 정적 평가로는 plug가 이미 고리를 통과해 버려 오정렬도
무접촉으로 나오는 문제가 있어, **접근→삽입 40스텝 궤적 전체를 따라가며 접촉 발생 스텝 수
(`hit_steps`)를 누적 집계**하는 방식으로 바꿔 풀었다.

> ⚠️ force 절대값은 kinematic IK 모드에서 비물리적(솔버가 충돌을 무시하고 plug를 침투시킨 뒤
> 침투깊이×강성으로 역산)이라 신뢰할 수 없다. 판별에는 **ncon / hit_steps**만 사용한다.

### 8.1 V1 결과 (완료) — `outputs/only_port/`, seed=42, cam_port

| 케이스 | max ncon | hit steps | peak F | landing | 판정 |
|---|---|---|---|---|---|
| 1 정렬 | **0** | 0/40 | 0 N | 0.02 mm | PASS (통과) |
| 2 5° tilt | 7 | 6/40 | 416 N | 0.02 mm | CONTACT |
| 3 10° tilt | 9 | 8/40 | 324 N | 0.02 mm | CONTACT |
| 4 FP 추정 | **0** | 0/40 | 0 N | 0.79 mm | PASS (통과) |

정렬·FP 추정 삽입은 무접촉으로 통과했고, 의도적 오정렬(5°/10°)에서만 간섭이 깨끗이 갈려
충돌 검증 체계가 의도대로 동작함을 확인했다.

### 8.2 V2 결과 (완료, 이슈 발견) — `outputs/collision_test_results.json`, seed=42, cam_port

| 케이스 | max ncon | hit steps | peak F | landing | tilt 측정값 | 판정 |
|---|---|---|---|---|---|---|
| 1 정렬 | 0 | 0/40 | 0 N | 0.02 mm | 0.04° | PASS |
| 2 5° tilt | 6 | 6/40 | 146 N | 0.02 mm | 4.99° | CONTACT |
| 3 10° tilt | 10 | 8/40 | 333 N | 0.02 mm | 9.98° | CONTACT |
| 4 FP 추정 | 14 | **9/40** | 712 N | **4.13 mm** | 2.57° | **CONTACT** |

정렬·오정렬 케이스는 V1과 동일하게 의도대로 갈렸으나, **케이스 4(FP 추정 삽입)가 V1에서는 무접촉
통과(0.79mm)였던 것과 달리 V2에서는 접촉이 발생(4.13mm, 9/40 스텝)**했다. 차체 결합으로 인한
occlusion 탓에 FP 입력 관측 품질이 V1보다 떨어져 추정 오차가 커진 것으로 보이나, 원인은 아직
확정하지 못했다 — §9 남은 과제로 이어짐.

---

## 9. 남은 과제 🔲

### V2(차량 결합) — 진행 중

- [ ] **[V2-Eval]** V2(차량 결합) 조건으로 A~E 정확도 매트릭스(N=50) 재실행 — V1과 동일하게
      서브밀리미터 정확도가 유지되는지 확인.
- [ ] **[V2-Col]** §8.2에서 발견된 "FP 추정 삽입이 V2에서 접촉 발생" 이슈 원인 분석 — occlusion으로
      인한 포즈 추정 오차 확대인지, 좌표 변환(부모 body 중첩) 쪽 문제인지 구분 필요.
- [ ] 미커밋 상태인 `scene.xml`의 `hummer_vehicle` 대체 quat 실험 주석 정리(사용자 작업 중, 별도 진행).

### Phase 1 공통 — 기존 과제

- [ ] **[Tilt-1]** `MAX_TILT_DEG` 10° → 15~20°로 확장 후 조건 B/D 기준 재평가
- [ ] **[Paper-1]** 논문 Table 작성 — 조건 A~E 비교표(§7) + V1/V2 비교
- [ ] **[Paper-2]** "depth noise affects translation only" 분석 서술
- [ ] **RGB-D vs 스테레오 depth ablation** — `test_stereo_depth.py` 결과와 정량 비교
- [ ] `docs/WeMeet_결과보고서.md` — 현재 V1(KIA EV6) 기준 초안을 V1/V2 구분 반영해 갱신 (별도 작업)

### 완료된 과제

- [x] Phase 1-C 충돌 테스트(V1, Col-1~6) — §8.1
- [x] eye-in-hand 카메라(`cam_eih`) + depth 노이즈 모델 — §5
- [x] V2(차량 결합) 씬 구성 + occlusion/좌표변환/로봇 도달 문제 해결 — §3.2
- [x] V2 충돌 테스트(4케이스) 1차 실행 — §8.2 (단, FP 케이스 이슈는 미해결)

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
12. **(V2)** body가 다른 body의 자식으로 중첩된 경우(`charge_port_frame` ← `hummer_vehicle`), world
    pose를 직접 대입하지 말고 부모 frame 기준 로컬 좌표로 변환 후 대입한다 — 원칙 6(동적 계산)과 같은 결.
13. **(V2)** 차체에 결합된 부품의 가시성 판정은 "보이는지 여부"가 아니라 "차체에 가리지 않고 정면에
    가깝게 보이는지"까지 기준으로 잡는다 — occlusion이 있는 환경에서는 단순 픽셀수 임계값을 충분히
    높게 잡아야 한다.
