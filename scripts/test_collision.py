#!/usr/bin/env python3
"""[Col-5/6] 충전건 tip ↔ 충전구 소켓 rim 충돌 테스트 (CLAUDE.md §8).

collision geom (CLAUDE.md §8, contype=2/conaffinity=2/group=2):
    - 충전구: port_rim_0 ~ port_rim_7  (소켓 입구 반지름 0.028m 원주 8기둥, scene.xml)
    - 충전건: ee_tip_collision          (tip 실효반지름 0.020m cylinder, ur10e.xml)

4 케이스 (모두 동일한 고정 충전구 포즈에서, charger_tip_site 를 insert 타겟으로 IK):
    1) 정렬 삽입       : GT pose 기반, R_aligned 그대로            → ncon=0 기대
    2) 오정렬 5° tilt  : R_aligned 를 tip 로컬축으로 5° 기울여 삽입  → ncon>0 기대
    3) 오정렬 10° tilt : 10° 기울여 삽입                            → ncon 더 큼
    4) FP 추정 삽입    : cam_port RGB-D → FoundationPose → IK 삽입  → 정렬과 유사

측정 (port_rim_* ↔ ee_tip_collision 쌍만):
    - data.ncon (해당 쌍 카운트)
    - contact force 크기 (mj_contactForce, 해당 쌍 합/최대)
    - insert landing 위치 오차 (charger_tip_site 실제 위치 vs GT insert 위치, mm)

산출물:
    - outputs/collision_test_results.json
    - outputs/collision_test_vis.png  (4케이스 overview 스크린샷, contact 표시)

Usage:
    conda activate wemeet
    python scripts/test_collision.py
"""
import os
import sys
import json

import numpy as np
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)

# 기존 통합 파이프라인의 좌표/IK/렌더/FP 헬퍼를 그대로 재사용 (CLAUDE.md §2 규약 일관성).
import test_integrated_pipeline as pipe
from src.ik import solve_position_ik, solve_pose_ik

OUTPUT_DIR = pipe.OUTPUT_DIR
JSON_OUT = os.path.join(OUTPUT_DIR, "collision_test_results.json")
PNG_OUT = os.path.join(OUTPUT_DIR, "collision_test_vis.png")

CAMERA = "cam_port"          # FP 입력: 외부 고정(GT depth, 노이즈 없음) = 베이스라인 A
OVERVIEW_CAM = "overview_camera"
RIM_GEOMS = [f"port_rim_{i}" for i in range(8)]
TIP_GEOM = "ee_tip_collision"
INSERT_SITE = "charge_port_insert"
SEED = 42
TILT_CASES_DEG = [5.0, 10.0]
EST_REFINE_ITER = pipe.EST_REFINE_ITER


def collect_pair_contacts(model, data, tip_id, rim_ids):
    """data.contact 중 (ee_tip_collision ↔ port_rim_*) 쌍만 골라 force 측정.

    Returns: dict(n_contact, force_sum_N, force_max_N, per_contact[...]).
    mj_contactForce 는 contact 좌표계 6D wrench → 선형 force[:3] 크기를 사용.
    """
    rim_set = set(rim_ids)
    n, fsum, fmax, detail = 0, 0.0, 0.0, []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        is_pair = (g1 == tip_id and g2 in rim_set) or (g2 == tip_id and g1 in rim_set)
        if not is_pair:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        fmag = float(np.linalg.norm(wrench[:3]))
        rim_g = g2 if g1 == tip_id else g1
        rim_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, rim_g)
        n += 1
        fsum += fmag
        fmax = max(fmax, fmag)
        detail.append({"rim": rim_name, "dist_mm": float(c.dist * 1e3),
                       "force_N": fmag})
    return {"n_contact": n, "force_sum_N": fsum, "force_max_N": fmax,
            "contacts": detail}


def step_ik(model, data, ids, target_pos, target_rot, reset=False):
    """charger_tip_site 를 타겟에 정렬하는 1 스텝 IK (CLAUDE.md §10-4 2단계).

    reset=False 면 직전 qpos 에서 warm-start (sweep 중 연속 이동).
    solve_pose_ik 내부 마지막에 mj_forward → data.ncon 유효 → 그대로 contact 판독.
    """
    if reset:
        pipe.reset_robot(model, data)
    solve_position_ik(model, data, ids["ee"], target_pos,
                      ids["joints"], max_iters=200, verbose=False)
    _, fpos, frot, pe, re = solve_pose_ik(
        model, data, ids["ee"], target_pos, target_rot,
        ids["joints"], max_iters=300, pos_tol=1e-4, rot_tol=1e-3,
        step_size=0.3, damping=1e-4, rot_weight=1.5, verbose=False)
    return fpos, frot, pe, re


def sweep_insertion(model, data, ids, tip_id, rim_ids,
                    start_pos, end_pos, target_rot, n_steps=40):
    """approach→insert 직선 삽입 궤적을 따라가며 케이스 자세(target_rot)를 유지.

    각 스텝마다 IK→mj_forward→contact 판독. 정렬이면 plug 가 rim 구멍을 통과(무접촉),
    기울어졌으면 plug 선단이 진입 중 rim 기둥에 걸린다(접촉). 궤적 전체에서 집계한다.

    Returns: dict(max_ncon, peak_force_N, sum_force_N, n_contact_steps,
                  final_pos, final_rot, vis_step, vis_ncon, all_contacts).
    """
    pipe.reset_robot(model, data)
    max_ncon, peak_F, sum_F, n_steps_hit = 0, 0.0, 0.0, 0
    vis_step, vis_ncon, vis_F = n_steps - 1, 0, 0.0
    fpos = frot = None
    all_contacts = []
    for k in range(n_steps):
        a = k / (n_steps - 1)
        tpos = (1 - a) * start_pos + a * end_pos
        fpos, frot, pe, re = step_ik(model, data, ids, tpos, target_rot, reset=(k == 0))
        con = collect_pair_contacts(model, data, tip_id, rim_ids)
        if con["n_contact"] > 0:
            n_steps_hit += 1
            sum_F += con["force_sum_N"]
            for c in con["contacts"]:
                c["step"] = k
                all_contacts.append(c)
        max_ncon = max(max_ncon, con["n_contact"])
        peak_F = max(peak_F, con["force_max_N"])
        # 스크린샷: 가장 접촉이 큰(또는 없으면 마지막) 스텝을 대표로.
        if con["n_contact"] > vis_ncon or (con["n_contact"] == vis_ncon and con["force_max_N"] > vis_F):
            vis_step, vis_ncon, vis_F = k, con["n_contact"], con["force_max_N"]
    return {"max_ncon": max_ncon, "peak_force_N": peak_F, "sum_force_N": sum_F,
            "n_contact_steps": n_steps_hit, "final_pos": fpos, "final_rot": frot,
            "vis_step": vis_step, "all_contacts": all_contacts}


def make_focus_camera(lookat, distance=0.32, azimuth=110.0, elevation=-18.0):
    """삽입 영역(소켓 입구)을 가깝게 잡는 free 카메라."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def render_view(renderer, model, data, cam, opt):
    renderer.update_scene(data, camera=cam, scene_option=opt)
    return renderer.render().copy()


def save_vis(images, summaries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Collision test — ee_tip_collision vs port_rim_* (group2 + contacts)",
                 fontsize=14)
    for ax, img, s in zip(axes.ravel(), images, summaries):
        ax.imshow(img)
        ax.set_title(s, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    cwd0 = os.getcwd()
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    img_w, img_h = pipe.CAM_RES.get(CAMERA, (pipe.IMG_W, pipe.IMG_H))
    model = mujoco.MjModel.from_xml_path(pipe.XML_PATH)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, img_w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, img_h)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=img_h, width=img_w)

    ids = {
        "cam": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA),
        "geom": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, pipe.PORT_GEOM),
        "body": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pipe.PORT_BODY),
        "ee": model.site(pipe.EE_SITE).id,
        "joints": pipe.get_joint_ids(model, pipe.JOINT_NAMES),
        "targets": {s: model.site(s).id for s in pipe.TARGET_SITES},
    }
    ids["insert"] = ids["targets"][INSERT_SITE]
    ids["obs"] = model.site(pipe.OBS_SITE).id
    ids["noise"] = "none"
    ids["eye_in_hand"] = bool(model.cam_bodyid[ids["cam"]] != 0)

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TIP_GEOM)
    rim_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in RIM_GEOMS]
    ids["approach"] = ids["targets"]["charge_port_approach"]
    K = pipe.intrinsics_from_fovy(model, ids["cam"], img_w, img_h)
    insert_offset = pipe.target_in_obj(model, ids["insert"])
    approach_offset = pipe.target_in_obj(model, ids["approach"])

    # collision geom(group2) + contact 표시용 렌더 옵션
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    for g in range(3):
        opt.geomgroup[g] = 1
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    print(f"[INFO] camera={CAMERA} tip_geom_id={tip_id} rim_geom_ids={rim_ids}")

    # 1) 고정 충전구 포즈 1개 (도달가능 + 가시) 샘플 — 4 케이스 공통.
    print("[INFO] 도달가능/가시 충전구 포즈 샘플 ...")
    pos, quat = pipe.sample_reachable_pose(model, data, renderer, ids, rng)
    pipe.set_port_pose(model, data, ids["body"], pos, quat)

    # GT 기준값: approach(삽입 시작)·insert(착지) world 위치와 정렬 자세.
    R_world_obj = data.xmat[ids["body"]].reshape(3, 3).copy()
    gt_insert_pos = data.site_xpos[ids["insert"]].copy()
    gt_approach_pos = data.site_xpos[ids["approach"]].copy()
    R_aligned = R_world_obj @ pipe.R_TIP_IN_OBJ
    # 스크린샷용 free 카메라 — 소켓 입구 중심을 가깝게.
    center_id = model.site("charge_port_center").id
    focus_cam = make_focus_camera(data.site_xpos[center_id].copy())
    print(f"[INFO] port pos={pos.round(3)}  gt_insert_pos={gt_insert_pos.round(4)}")

    # 2) FoundationPose 추정(케이스 4용) — 빌드 후 cwd=FP_DIR 에서 register.
    print("[INFO] FoundationPose 초기화 + 추정 ...")
    est = pipe.build_estimator()
    pipe.position_for_capture(model, data, ids)          # cam_port → home
    rgb, depth, mask = pipe.render_rgbd_mask(renderer, model, data, ids["cam"], ids["geom"])
    Twc_cv = pipe.cam_in_world_cv(model, data, ids["cam"])
    fp_ob = est.register(K=K.astype(np.float64), rgb=rgb,
                         depth=depth.astype(np.float32), ob_mask=mask.astype(bool),
                         iteration=EST_REFINE_ITER)
    fp_insert = Twc_cv @ fp_ob @ insert_offset
    fp_approach = Twc_cv @ fp_ob @ approach_offset
    os.chdir(cwd0)
    pipe.reset_robot(model, data)

    # 3) 4 케이스 정의: (label, start_pos, end_pos, target_rot)
    #    삽입 궤적 approach→insert 를 따라가며 케이스 자세를 유지한다(삽입 시도).
    cases = [("1_aligned", gt_approach_pos, gt_insert_pos, R_aligned)]
    for d in TILT_CASES_DEG:
        # tip 로컬 Y축(삽입 forward=local X 에 수직)으로 d° 기울임 → 삽입축 오정렬.
        R_tilt = R_aligned @ pipe.rotvec_to_R(np.array([0.0, np.deg2rad(d), 0.0]))
        cases.append((f"{int(d)}_tilt", gt_approach_pos, gt_insert_pos, R_tilt))
    cases.append(("4_fp", fp_approach[:3, 3].copy(), fp_insert[:3, 3].copy(),
                  fp_insert[:3, :3].copy()))

    # 4) 케이스 실행(삽입 궤적 sweep) + 측정 + 스크린샷
    results, images, summaries = [], [], []
    for label, spos, epos, trot in cases:
        sw = sweep_insertion(model, data, ids, tip_id, rim_ids, spos, epos, trot)
        landing_mm = float(np.linalg.norm(sw["final_pos"] - gt_insert_pos) * 1e3)
        tilt_deg = pipe.rot_err_deg(sw["final_rot"], R_aligned)

        rec = {
            "case": label,
            "n_contact": sw["max_ncon"],
            "force_sum_N": sw["sum_force_N"],
            "force_max_N": sw["peak_force_N"],
            "n_contact_steps": sw["n_contact_steps"],
            "landing_pos_err_mm": landing_mm,
            "tip_tilt_vs_aligned_deg": tilt_deg,
            "contacts": sw["all_contacts"][:20],   # 대표 일부만 저장
        }
        results.append(rec)

        # 스크린샷: 대표 스텝(최대 접촉, 없으면 최종)을 재현해 렌더.
        a = sw["vis_step"] / 39.0
        vpos = (1 - a) * spos + a * epos
        step_ik(model, data, ids, vpos, trot, reset=True)
        images.append(render_view(renderer, model, data, focus_cam, opt))
        summaries.append(
            f"case {label}: max_ncon={sw['max_ncon']}  "
            f"Fpeak={sw['peak_force_N']:.2f}N  hit_steps={sw['n_contact_steps']}/40\n"
            f"landing={landing_mm:.2f}mm  tilt={tilt_deg:.1f}deg")
        print(f"[CASE {label:9s}] max_ncon={sw['max_ncon']:2d}  "
              f"Fpeak={sw['peak_force_N']:7.3f}N  Fsum={sw['sum_force_N']:8.3f}N  "
              f"hit={sw['n_contact_steps']:2d}/40  landing={landing_mm:6.2f}mm  "
              f"tilt={tilt_deg:5.2f}°")

    # 판정 (CLAUDE.md §8 기준): 성공 ncon=0 & F<1N & landing≤2mm / 간섭 ncon>0 or F>5N or landing>5mm
    def verdict(r):
        if r["n_contact"] == 0 and r["force_max_N"] < 1.0 and r["landing_pos_err_mm"] <= 2.0:
            return "PASS(삽입성공)"
        if r["n_contact"] > 0 or r["force_max_N"] > 5.0 or r["landing_pos_err_mm"] > 5.0:
            return "CONTACT(간섭감지)"
        return "MARGINAL"
    for r in results:
        r["verdict"] = verdict(r)

    out = {
        "config": {
            "camera": CAMERA, "seed": SEED, "tilt_cases_deg": TILT_CASES_DEG,
            "refine_iter": EST_REFINE_ITER,
            "tip_geom": TIP_GEOM, "rim_geoms": RIM_GEOMS,
            "port_pos": pos.tolist(), "port_quat": quat.tolist(),
            "gt_insert_pos": gt_insert_pos.tolist(),
            "criteria": {"pass": "ncon==0 & Fmax<1N & landing<=2mm",
                         "fail": "ncon>0 | Fmax>5N | landing>5mm"},
        },
        "cases": results,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(JSON_OUT, "w") as f:
        json.dump(out, f, indent=2)
    save_vis(images, summaries, PNG_OUT)

    print("\n===== Collision test summary =====")
    print(f"  {'case':10s} {'ncon':>5s} {'Fmax(N)':>9s} {'landing(mm)':>12s} "
          f"{'tilt(deg)':>10s}  verdict")
    for r in results:
        print(f"  {r['case']:10s} {r['n_contact']:5d} {r['force_max_N']:9.3f} "
              f"{r['landing_pos_err_mm']:12.2f} {r['tip_tilt_vs_aligned_deg']:10.2f}"
              f"  {r['verdict']}")
    print(f"\n[INFO] 저장: {JSON_OUT}\n[INFO] 저장: {PNG_OUT}")


if __name__ == "__main__":
    main()
