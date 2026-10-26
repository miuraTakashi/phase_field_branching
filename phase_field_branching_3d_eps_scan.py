"""
phase_field_branching_3d の ε 走査（a>0 の枝の蛇行は表面張力で減るか）

予測（phase_field_branching_3d.md「表面張力と蛇行」）:
  - 中心線の曲げは面積をほぼ変えず、先端の向きは σ_1 = 0 で中立 → ε を上げても蛇行（半径単位）は減らない。
  - 円柱側面の静止条件 c(v_cyl(ρ) - 1/2 - a) = d_u/ρ で太さ ρ が決まり、ε↑ で細くなる。
  - 先端速度 V_tip = Ṙ_sphere(ρ) が ε ≈ 0.25 で負になり、伸長が止まる。

実験:
  A) 球＋摂動から ε ∈ {0.20, 0.21, 0.22, 0.23}（a=0.06）
  B) ε=0.20 で枝を育てた状態から ε ∈ {0.20, 0.24, 0.26, 0.28} に切り替えて継続
評価: 3D 骨格化 → 枝セグメントごとに中心線を追跡し、屈曲度 τ、曲率 κρ、持続長 ℓ_p/ρ、動径整列 |t·r̂|。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, optimize, special
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree

import phase_field_branching_3d as m3

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RAW = HERE / "results_3d_raw"

A_CTRL = 0.06
EPS_SPHERE = (0.20, 0.21, 0.22, 0.23)
EPS_QUENCH = (0.20, 0.24, 0.26, 0.28)
QUENCH_FROM_STEP = 5000
QUENCH_STEPS = 6000


# ---------------------------------------------------------------------------
# 理論: 円柱の静止半径と先端速度
# ---------------------------------------------------------------------------


def cylinder_interface_v(rho: float, d: float) -> float:
    x = rho / math.sqrt(d)
    p = special.i1(x) / special.i0(x)
    s = special.k1(x) / special.k0(x)
    return s / (p + s)


def tube_theory(eps: float, d: float, a: float) -> dict:
    du, c, ell = m3.sharp_constants(eps, d)

    def side(r):
        return c * (cylinder_interface_v(r, d) - 0.5 - a) - du / r

    rs = np.linspace(0.05, 5.0, 500)
    vals = np.array([side(r) for r in rs])
    roots = [optimize.brentq(side, rs[i], rs[i + 1]) for i in range(len(rs) - 1) if vals[i] * vals[i + 1] < 0]
    rho = float(max(roots)) if roots else float("nan")
    v_tip = m3.sphere_dRdt(rho, eps, d, a) if roots else float("nan")
    return {
        "eps": eps,
        "Gamma": m3.instability_number(eps),
        "rho_cyl": rho,
        "rho_local": (c * ell / 4 - du) / (c * a),
        "v_tip": v_tip,
        "interface_width": 4 * math.sqrt(2 * du),
        "dRdt_sphere_R0": m3.sphere_dRdt(1.0, eps, d, a),
    }


# ---------------------------------------------------------------------------
# 中心線解析
# ---------------------------------------------------------------------------


def _ordered_path(pts: np.ndarray) -> np.ndarray:
    """26 近傍で連結した骨格点群の最長測地線（2 回の最短路探索）。"""
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=1.75, output_type="ndarray")
    if len(pairs) == 0:
        return pts[:1]
    w = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
    n = len(pts)
    g = coo_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])), shape=(n, n)).tocsr()
    d0 = shortest_path(g, indices=0, directed=False)
    d0[~np.isfinite(d0)] = -1
    i = int(np.argmax(d0))
    di, pred = shortest_path(g, indices=i, directed=False, return_predecessors=True)
    di[~np.isfinite(di)] = -1
    j = int(np.argmax(di))
    path = [j]
    while path[-1] != i and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return pts[path[::-1]]


def _resample(path: np.ndarray, h: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 2 * h:
        return path[:1]
    q = np.arange(0.0, s[-1], h)
    return np.stack([np.interp(q, s, path[:, k]) for k in range(3)], axis=1)


def centerline_stats(u: np.ndarray, region: np.ndarray, dx: float, center: np.ndarray) -> dict:
    from skimage.morphology import skeletonize

    m1 = ndimage.gaussian_filter(u.astype(np.float32), 0.8, mode="wrap") > 0.5
    lab, n_comp = ndimage.label(m1)
    if n_comp == 0:
        return {"n_comp": 0}
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n_comp + 1))
    dist = ndimage.distance_transform_edt(m1)
    skel = skeletonize(m1).astype(bool)
    rho_vox = float(np.mean(dist[skel & region])) if np.any(skel & region) else float("nan")
    rho = rho_vox * dx

    nb = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="wrap") - 1
    junction = skel & (nb >= 3)
    junction = ndimage.binary_dilation(junction, iterations=1) & skel
    segs = skel & ~junction
    slab, n_seg = ndimage.label(segs, structure=np.ones((3, 3, 3)))
    objs = ndimage.find_objects(slab)

    min_len = max(3.0 * rho_vox, 8.0)
    h = 1.0
    smooth = max(1.0, 0.5 * rho_vox)
    lags = np.arange(0, 120)
    csum = np.zeros(len(lags))
    ccount = np.zeros(len(lags))
    taus, kappas, radial, lengths = [], [], [], []
    for k, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = slab[sl] == k
        if sub.sum() < min_len:
            continue
        pts = np.argwhere(sub).astype(np.float64) + np.array([s.start for s in sl])
        mid = pts.mean(axis=0).astype(int)
        if not region[tuple(mid)]:
            continue
        path = _ordered_path(pts)
        path = _resample(path, h)
        if len(path) < min_len:
            continue
        path = ndimage.gaussian_filter1d(path, smooth, axis=0, mode="nearest")
        t = np.gradient(path, h, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        L = (len(path) - 1) * h
        chord = np.linalg.norm(path[-1] - path[0])
        trim = int(math.ceil(2 * smooth))
        core = slice(trim, len(path) - trim) if len(path) > 2 * trim + 3 else slice(None)
        kap = np.linalg.norm(np.gradient(t, h, axis=0), axis=1)[core]
        rhat = path - center
        rhat /= np.maximum(np.linalg.norm(rhat, axis=1, keepdims=True), 1e-12)
        taus.append(L / max(chord, 1e-9))
        kappas.append(float(np.median(kap)) * rho_vox)
        radial.append(float(np.mean(np.abs(np.sum(t * rhat, axis=1)))))
        lengths.append(L / rho_vox)
        for li, lag in enumerate(lags):
            if lag >= len(t):
                break
            dots = np.sum(t[: len(t) - lag] * t[lag:], axis=1)
            csum[li] += dots.sum()
            ccount[li] += len(dots)

    corr = np.where(ccount > 0, csum / np.maximum(ccount, 1), np.nan)
    lag_rho = lags * h / rho_vox if rho_vox > 0 else lags * np.nan
    ok = np.isfinite(corr) & (corr > 0.25) & (lags > 0) & (ccount > 200)
    if ok.sum() >= 3:
        slope = np.polyfit(lag_rho[ok], np.log(corr[ok]), 1)[0]
        lp_rho = float(-1.0 / slope) if slope < 0 else float("inf")
    else:
        lp_rho = float("nan")

    small = sizes < (4 / 3) * math.pi * (2.5 * max(rho_vox, 1.0)) ** 3
    return {
        "n_comp": int(n_comp),
        "frac_vol_droplets": float(sizes[small].sum() / sizes.sum()),
        "rho_measured": rho,
        "n_segments": len(taus),
        "tortuosity_median": float(np.median(taus)) if taus else float("nan"),
        "kappa_rho_median": float(np.median(kappas)) if kappas else float("nan"),
        "persistence_over_rho": lp_rho,
        "radial_alignment": float(np.mean(radial)) if radial else float("nan"),
        "segment_len_over_rho_median": float(np.median(lengths)) if lengths else float("nan"),
        "skeleton_len": float(skel.sum()) * dx,
        "corr_lag_rho": lag_rho,
        "corr": corr,
    }


def grid_center(p: m3.Params3D) -> np.ndarray:
    return np.full(3, p.n / 2.0 - 1.0)


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


def run_quench(base: m3.Params3D, backend: str) -> tuple[dict, list[dict]]:
    p0 = replace(base, eps=0.20)
    s0 = m3.Solver(p0, backend)
    s0.step(QUENCH_FROM_STEP)
    s0.sync()
    uv0 = s0.uv.clone()
    pre = {"u": s0.u_host(), "p": p0, "label": f"quench pre (eps=0.20, t={QUENCH_FROM_STEP * p0.dt:.0f})"}
    del s0
    out = []
    for eps in EPS_QUENCH:
        p = replace(base, eps=eps)
        s = m3.Solver(p, backend)
        s.uv = uv0.clone()
        t0 = time.perf_counter()
        vol = []
        for _ in range(QUENCH_STEPS // 500):
            s.step(500)
            vol.append(float((s.u_host() > 0.5).sum()) * p.dx**3)
        s.sync()
        u = s.u_host()
        print(f"  [quench eps={eps:.2f}] {QUENCH_STEPS} steps, {time.perf_counter() - t0:.1f}s, vol {vol[0]:.1f}→{vol[-1]:.1f}", flush=True)
        out.append({"u": u, "p": p, "label": f"quench → eps={eps:.2f}", "vol": np.array(vol)})
        del s
    return pre, out


def analyze_state(u: np.ndarray, p: m3.Params3D) -> dict:
    probe = m3.InterfaceProbe(p)
    region, _, _ = m3.shell_region(p, probe.radii(u))
    return centerline_stats(u, region, p.dx, grid_center(p))


def plot_all(theory: list[dict], sphere: list[dict], pre: dict, quench: list[dict], out: Path) -> None:
    ncol = max(len(sphere), len(quench) + 1)
    fig = plt.figure(figsize=(4.2 * ncol, 13))
    gs = fig.add_gridspec(3, ncol, height_ratios=[1.1, 1.1, 1.0])
    for i, r in enumerate(sphere):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        st = r["stats"]
        if st.get("n_comp", 0) > 0:
            v, f = m3.isosurface(r["u"], r["p"].L, stride=2, cut=False)
            m3.draw_surface(ax, v, f, extent=0.5 * r["p"].L * 0.95)
        ax.set_title(
            f"sphere start, ε={r['p'].eps:.2f}, t={r['final_step'] * r['p'].dt:.0f}\n"
            f"τ={st.get('tortuosity_median', np.nan):.2f}  ℓp/ρ={st.get('persistence_over_rho', np.nan):.1f}  ρ={st.get('rho_measured', np.nan):.2f}",
            fontsize=9,
        )
        ax.set_axis_off()
    for i, r in enumerate([pre] + quench):
        ax = fig.add_subplot(gs[1, i], projection="3d")
        st = r["stats"]
        if st.get("n_comp", 0) > 0:
            v, f = m3.isosurface(r["u"], r["p"].L, stride=2, cut=False)
            m3.draw_surface(ax, v, f, extent=0.5 * r["p"].L * 0.95, base=(0.45, 0.70, 0.90) if i else (0.93, 0.72, 0.25))
        ax.set_title(
            f"{r['label']}\nτ={st.get('tortuosity_median', np.nan):.2f}  ℓp/ρ={st.get('persistence_over_rho', np.nan):.1f}  "
            f"ρ={st.get('rho_measured', np.nan):.2f}  drop={st.get('frac_vol_droplets', np.nan):.2f}",
            fontsize=9,
        )
        ax.set_axis_off()

    eps_t = [t["eps"] for t in theory]
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(eps_t, [t["rho_cyl"] for t in theory], "k-", label="ρ theory (cylinder side at rest)")
    ax.plot(eps_t, [t["interface_width"] for t in theory], "k:", label="interface width 4√(2d_u)")
    ax.plot([r["p"].eps for r in sphere], [r["stats"].get("rho_measured", np.nan) for r in sphere], "o", c="C1", label="measured, sphere start")
    ax.plot([r["p"].eps for r in quench], [r["stats"].get("rho_measured", np.nan) for r in quench], "s", c="C0", label="measured, quench")
    ax.set_xlabel("ε")
    ax.set_ylabel("tube radius ρ")
    ax.legend(fontsize=7)
    ax.set_title("tube radius")

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(eps_t, [t["v_tip"] for t in theory], "k-", label="V_tip theory = Ṙ_sphere(ρ)")
    ax.axhline(0, c="gray", lw=0.6)
    ax.plot([r["p"].eps for r in sphere], [r["v_front"] for r in sphere], "o", c="C1", label="measured dR_env/dt (sphere start)")
    ax.plot([r["p"].eps for r in quench], [r["dskel_dt_per_tip"] for r in quench], "s", c="C0", label="quench: skeleton growth rate / tip")
    ax.set_xlabel("ε")
    ax.set_ylabel("tip speed")
    ax.legend(fontsize=7)
    ax.set_title("tip advance stops near ε≈0.25")

    ax = fig.add_subplot(gs[2, 2])
    for r in sphere:
        st = r["stats"]
        if "corr" in st:
            ax.plot(st["corr_lag_rho"], st["corr"], c=plt.cm.Oranges(0.4 + 0.6 * (r["p"].eps - 0.2) / 0.03), label=f"sphere ε={r['p'].eps:.2f}")
    for r in [pre] + quench:
        st = r["stats"]
        if "corr" in st:
            ax.plot(st["corr_lag_rho"], st["corr"], "--", c=plt.cm.Blues(0.35 + 0.65 * (r["p"].eps - 0.2) / 0.08), label=r["label"])
    ax.set_xlim(0, 25)
    ax.set_ylim(-0.2, 1.0)
    ax.axhline(0, c="gray", lw=0.6)
    ax.set_xlabel("arc length / ρ")
    ax.set_ylabel("⟨t(s)·t(s+Δ)⟩")
    ax.legend(fontsize=6)
    ax.set_title("tangent correlation (centerline)")

    if ncol > 3:
        ax = fig.add_subplot(gs[2, 3])
        rows = [(f"S {r['p'].eps:.2f}", r["stats"]) for r in sphere] + [("Q pre", pre["stats"])] + [(f"Q {r['p'].eps:.2f}", r["stats"]) for r in quench]
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [s.get("tortuosity_median", np.nan) for _, s in rows], 0.4, label="tortuosity τ (median)")
        ax.bar(x + 0.2, [s.get("kappa_rho_median", np.nan) * 10 for _, s in rows], 0.4, label="10 × κρ (median)")
        ax.set_xticks(x, [n for n, _ in rows], rotation=60, fontsize=7)
        ax.legend(fontsize=7)
        ax.set_title("meander metrics (S: sphere start, Q: quench)")
    fig.suptitle(f"3D branching phase field, a={A_CTRL}: does stronger surface tension (larger ε) straighten branches?", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--backend", default="torch_cuda")
    ap_.add_argument("--n", type=int, default=256)
    ap_.add_argument("--steps", type=int, default=12000)
    args = ap_.parse_args()
    RESULTS.mkdir(exist_ok=True)
    base = m3.Params3D(n=args.n, a=A_CTRL, n_steps=args.steps)

    theory = [tube_theory(e, base.d, A_CTRL) for e in np.round(np.arange(0.20, 0.2851, 0.01), 3)]
    for t in theory:
        print("theory", {k: round(v, 4) for k, v in t.items()})

    sphere = []
    for eps in EPS_SPHERE:
        p = replace(base, eps=eps)
        res = m3.run_case(p, f"epsscan_sphere_eps{eps:.2f}", args.backend)
        h = res["hist"]
        half = h["t"] >= 0.5 * h["t"][-1]
        v_front = float(np.polyfit(h["t"][half], h["R_env"][half], 1)[0]) if half.sum() > 3 else float("nan")
        st = analyze_state(res["u"], p) if np.any(res["u"] > 0.5) else {"n_comp": 0}
        sphere.append({"u": res["u"], "p": p, "final_step": res["final_step"], "v_front": v_front, "stats": st})
        print(f"  sphere eps={eps:.2f}: v_front={v_front:.5f}", {k: v for k, v in st.items() if not isinstance(v, np.ndarray)}, flush=True)

    pre, quench = run_quench(base, args.backend)
    pre["stats"] = analyze_state(pre["u"], pre["p"])
    for r in quench:
        r["stats"] = analyze_state(r["u"], r["p"]) if np.any(r["u"] > 0.5) else {"n_comp": 0}
        dl = r["stats"].get("skeleton_len", 0.0) - pre["stats"]["skeleton_len"]
        n_tips = max(1, r["stats"].get("n_segments", 1))
        r["dskel_dt_per_tip"] = dl / (QUENCH_STEPS * r["p"].dt) / n_tips
        print(f"  {r['label']}:", {k: v for k, v in r["stats"].items() if not isinstance(v, np.ndarray)}, flush=True)

    plot_all(theory, sphere, pre, quench, RESULTS / "phase_field_branching_3d_eps_scan.png")

    def clean(st):
        return {k: v for k, v in st.items() if not isinstance(v, np.ndarray)}

    summary = {
        "a": A_CTRL,
        "theory": theory,
        "sphere_start": [{"eps": r["p"].eps, "final_step": r["final_step"], "v_front": r["v_front"], **clean(r["stats"])} for r in sphere],
        "quench_pre": {"eps": 0.20, "step": QUENCH_FROM_STEP, **clean(pre["stats"])},
        "quench": [{"eps": r["p"].eps, "steps": QUENCH_STEPS, "dskel_dt_per_tip": r["dskel_dt_per_tip"], "vol": r["vol"].tolist(), **clean(r["stats"])} for r in quench],
    }
    (RESULTS / "phase_field_branching_3d_eps_scan.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("done")


if __name__ == "__main__":
    main()
