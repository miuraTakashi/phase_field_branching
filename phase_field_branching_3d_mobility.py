"""
phase_field_branching_3d + 粗視化密度場による移動度ゲート（外縁でのみ界面が動く）

  w = (1 - ℓ_w² Δ)^{-1} u          （w_t = D_w Δw + κ_w(u - w) の準静的極限、ℓ_w = √(D_w/κ_w)）
  u_t = M(w) [d_u Δu + u(1-u)(u+v-1-a)],   M(w) = ½[1 - tanh((w - w_c)/δ)]
  v_t = d Δv + 1 - u - v

数値: 2D/3D 版と同じ半陰的スペクトル法で u* を求め、u^{n+1} = u^n + M(w^n)(u* - u^n)。
w は u* のスペクトルから 1 回の irfftn で得る（追加コストは FFT 1 本）。

予測（phase_field_branching_3d.md「移動度ゲート」）:
  - ∇M が外向き → 先端の向きに復元力（緩和長 ~ ℓ_w）→ 動径整列・持続長の増加
  - 内部凍結 → とぐろ巻き・側方出芽・融合ループの抑制
  - 二分岐には先端半球の l=2 不安定（σ_2 > 0 > σ_3）が必要 → a ≈ 0.03 付近
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
from scipy import ndimage

import phase_field_branching_3d as m3
import phase_field_branching_3d_eps_scan as es

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RAW = HERE / "results_3d_raw"
PREFIX = "phase_field_branching_3d_mobility"

A_VALUES = (0.03, 0.04, 0.06)
ELL_W_VALUES = (None, 1.5, 2.5)
WC_FACTOR = 0.8
LAMELLA_PERIOD = 2.5


class GatedSolver:
    def __init__(self, p: m3.Params3D, ell_w: float | None, w_c: float, delta: float, device: str = "cuda"):
        import torch

        self.torch = torch
        self.p = p
        self.ell_w = ell_w
        self.w_c = w_c
        self.delta = delta
        n = p.n
        lam = m3.laplacian_symbol_rfft(n, p.dx)
        mult = np.stack([1.0 / (1.0 + p.dt * p.du * lam), 1.0 / (1.0 + p.dt * p.d * lam)])
        self.mult = torch.as_tensor(mult, dtype=torch.float32, device=device)
        u0, v0 = m3.initial_condition(p)
        self.uv = torch.as_tensor(np.stack([u0, v0]), dtype=torch.float32, device=device)
        if ell_w is not None:
            self.wmult = torch.as_tensor(1.0 / (1.0 + ell_w**2 * lam), dtype=torch.float32, device=device)
            self.w = torch.fft.irfftn(torch.fft.rfftn(self.uv[0]) * self.wmult, s=(n, n, n))
        else:
            self.w = None

    def mobility(self):
        if self.w is None:
            return None
        return 0.5 * (1.0 - self.torch.tanh((self.w - self.w_c) / self.delta))

    def step(self, k: int = 1) -> None:
        torch = self.torch
        p = self.p
        n = p.n
        for _ in range(k):
            u = self.uv[0]
            v = self.uv[1]
            f = u + p.dt * (u * (1.0 - u) * (u + v - 1.0 - p.a))
            g = v + p.dt * (1.0 - u - v)
            spec = torch.fft.rfftn(torch.stack((f, g)), dim=(1, 2, 3))
            spec *= self.mult
            new = torch.fft.irfftn(spec, s=(n, n, n), dim=(1, 2, 3))
            if self.w is not None:
                m = 0.5 * (1.0 - torch.tanh((self.w - self.w_c) / self.delta))
                new[0] = u + m * (new[0] - u)
                self.w = torch.fft.irfftn(spec[0] * self.wmult, s=(n, n, n))
            self.uv = new

    def u_host(self) -> np.ndarray:
        return self.uv[0].float().cpu().numpy()

    def w_host(self) -> np.ndarray | None:
        return None if self.w is None else self.w.float().cpu().numpy()

    def m_host(self) -> np.ndarray | None:
        m = self.mobility()
        return None if m is None else m.float().cpu().numpy()


# ---------------------------------------------------------------------------
# 理論
# ---------------------------------------------------------------------------


def tip_theory(a: float, eps: float = 0.2, d: float = 0.1) -> dict:
    t = es.tube_theory(eps, d, a)
    rho = t["rho_cyl"]
    _, rel = m3.sphere_sigma_l(np.arange(2, 5), rho, eps, d, a)
    phi_lam = m3.lamella_fraction(a, LAMELLA_PERIOD, d)
    return {
        "a": a,
        "rho_cyl": rho,
        "v_tip": t["v_tip"],
        "sigma_rel_l2": float(rel[0]),
        "sigma_rel_l3": float(rel[1]),
        "sigma_rel_l4": float(rel[2]),
        "phi_lamella": phi_lam,
        "w_c": WC_FACTOR * phi_lam,
    }


# ---------------------------------------------------------------------------
# 分岐点の分類（Y: 対称二分岐 / T: 側枝）とトポロジー
# ---------------------------------------------------------------------------


def junction_stats(u: np.ndarray, region: np.ndarray, dx: float, center: np.ndarray) -> dict:
    from skimage.measure import euler_number
    from skimage.morphology import skeletonize

    m1 = ndimage.gaussian_filter(u.astype(np.float32), 0.8, mode="wrap") > 0.5
    if not m1.any():
        return {}
    dist = ndimage.distance_transform_edt(m1)
    skel = skeletonize(m1).astype(bool)
    rho = float(np.mean(dist[skel & region])) if np.any(skel & region) else 3.0
    nb = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="wrap") - 1
    jvox = skel & (nb >= 3)
    jlab, n_j = ndimage.label(jvox, structure=np.ones((3, 3, 3)))
    segs = skel & ~ndimage.binary_dilation(jvox, iterations=1)
    slab, _ = ndimage.label(segs, structure=np.ones((3, 3, 3)))
    ends = skel & (nb == 1) & region

    r_lo, r_hi = 1.0 * rho, 3.5 * rho
    w = int(math.ceil(r_hi)) + 2
    n_y = n_t = n_other = n_high = 0
    angles = []
    cents = ndimage.center_of_mass(jvox, jlab, index=np.arange(1, n_j + 1)) if n_j else []
    for c in cents:
        c = np.asarray(c)
        ci = tuple(c.astype(int))
        if not region[ci]:
            continue
        sl = tuple(slice(max(0, int(c[k]) - w), min(u.shape[k], int(c[k]) + w + 1)) for k in range(3))
        sub = slab[sl]
        off = np.array([s.start for s in sl])
        pts = np.argwhere(sub > 0)
        if len(pts) == 0:
            continue
        labs = sub[tuple(pts.T)]
        pts = pts + off
        dd = np.linalg.norm(pts - c, axis=1)
        ring = (dd >= r_lo) & (dd <= r_hi)
        dirs = []
        for lb in np.unique(labs[ring]):
            q = pts[ring & (labs == lb)]
            if len(q) < 2:
                continue
            vec = q.mean(axis=0) - c
            nv = np.linalg.norm(vec)
            if nv > 0:
                dirs.append(vec / nv)
        if len(dirs) < 3:
            continue
        if len(dirs) > 3:
            n_high += 1
            continue
        dirs = np.asarray(dirs)
        rhat = c - center
        rhat /= max(np.linalg.norm(rhat), 1e-9)
        ip = int(np.argmin(dirs @ rhat))
        e = -dirs[ip]
        kids = np.delete(dirs, ip, axis=0)
        th = np.degrees(np.arccos(np.clip(kids @ e, -1, 1)))
        t1, t2 = sorted(th)
        angles.append((t1, t2))
        if t1 >= 15 and t2 <= 80 and (t2 - t1) <= 25:
            n_y += 1
        elif t1 < 20 and t2 >= 45:
            n_t += 1
        else:
            n_other += 1

    chi = euler_number(m1, connectivity=3)
    lab, b0 = ndimage.label(m1)
    _, n_bg = ndimage.label(~m1)
    b2 = max(n_bg - 1, 0)
    b1 = b0 + b2 - chi
    n3 = n_y + n_t + n_other
    skel_len = float(skel.sum()) * dx
    return {
        "n_junction3": n3,
        "n_junction_higher": n_high,
        "frac_Y": n_y / n3 if n3 else float("nan"),
        "frac_T": n_t / n3 if n3 else float("nan"),
        "n_tips": int(ends.sum()),
        "loops_b1": int(b1),
        "loops_per_length": b1 / max(skel_len, 1e-9),
        "junction_angles": np.asarray(angles) if angles else np.zeros((0, 2)),
    }


def analyze(u: np.ndarray, p: m3.Params3D, lam_star: float) -> dict:
    probe = m3.InterfaceProbe(p)
    r_dirs = probe.radii(u)
    region, _, _ = m3.shell_region(p, r_dirs)
    center = es.grid_center(p)
    out = es.centerline_stats(u, region, p.dx, center)
    out.update(junction_stats(u, region, p.dx, center))
    m1 = ndimage.gaussian_filter(u, 0.8, mode="wrap") > 0.5
    med = m3.medial_shape_stats(m1, region, p.dx, radius=0.45 * lam_star)
    out["phi_shell"] = float(m1[region].mean())
    out["frac_tube"] = med.get("frac_tube", float("nan"))
    out["frac_sheet"] = med.get("frac_sheet", float("nan"))
    return out


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


def run_one(p: m3.Params3D, ell_w: float | None, th: dict, label: str, n_snap: int = 4) -> dict:
    s = GatedSolver(p, ell_w, w_c=th["w_c"], delta=0.1 * th["w_c"])
    probe = m3.InterfaceProbe(p)
    stop_r = 0.46 * p.L
    hist = {"t": [], "R_env": [], "R_max": [], "vol": [], "active_frac_iface": []}
    snaps = {}
    snap_every = max(1000, (p.n_steps // n_snap) // 500 * 500)
    t0 = time.perf_counter()
    step = 0
    stall = 0
    while True:
        u = s.u_host()
        r = probe.radii(u)
        vol = float((u > 0.5).sum()) * p.dx**3
        hist["t"].append(step * p.dt)
        hist["R_env"].append(float(np.percentile(r[r > 0], 90)) if np.any(r > 0) else 0.0)
        hist["R_max"].append(float(r.max()))
        hist["vol"].append(vol)
        mob = s.m_host()
        iface = np.abs(u - 0.5) < 0.25
        hist["active_frac_iface"].append(float(mob[iface].mean()) if mob is not None and iface.any() else 1.0)
        if step % snap_every == 0:
            snaps[step] = u[::2, ::2, ::2].astype(np.float16)
        if step % 2000 == 0:
            print(
                f"  [{label}] step {step:6d} R_env={hist['R_env'][-1]:6.2f} R_max={r.max():6.2f} vol={vol:8.1f} "
                f"active={hist['active_frac_iface'][-1]:.2f} ({time.perf_counter() - t0:6.1f}s)",
                flush=True,
            )
        if len(hist["vol"]) > 60:
            v_old = hist["vol"][-61]
            stall = stall + 1 if abs(vol - v_old) < 2e-3 * max(vol, 1.0) else 0
        if step >= p.n_steps or r.max() > stop_r or vol == 0.0 or stall > 20:
            break
        s.step(50)
        step += 50
    s.torch.cuda.synchronize()
    u = s.u_host()
    w = s.w_host()
    mob = s.m_host()
    snaps[step] = u[::2, ::2, ::2].astype(np.float16)
    wall = time.perf_counter() - t0
    reason = "boundary" if hist["R_max"][-1] > stop_r else ("vanished" if hist["vol"][-1] == 0 else ("stalled" if stall > 20 else "max_steps"))
    print(f"  [{label}] done: {step} steps ({reason}), wall {wall:.1f}s", flush=True)
    np.savez_compressed(
        RAW / f"mobility_{label}.npz",
        u=u.astype(np.float16),
        w=(w if w is not None else np.zeros(1)).astype(np.float16),
        snap_steps=np.array(sorted(snaps)),
        snaps=np.stack([snaps[k] for k in sorted(snaps)]),
        **{k: np.asarray(v) for k, v in hist.items()},
    )
    c = p.n // 2
    return {
        "p": p,
        "ell_w": ell_w,
        "label": label,
        "u": u,
        "w_mid": None if w is None else w[:, :, c].copy(),
        "m_mid": None if mob is None else mob[:, :, c].copy(),
        "snaps": snaps,
        "hist": {k: np.asarray(v) for k, v in hist.items()},
        "final_step": step,
        "reason": reason,
        "wall": wall,
        "th": th,
    }


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}


def score(st: dict) -> float:
    fy = st.get("frac_Y", float("nan"))
    ra = st.get("radial_alignment", float("nan"))
    if not (np.isfinite(fy) and np.isfinite(ra)) or st.get("n_junction3", 0) < 5:
        return -1.0
    return fy * (ra - 0.5)


def plot_grid(runs: list[dict], out: Path) -> None:
    fig = plt.figure(figsize=(5.0 * len(A_VALUES), 5.0 * len(ELL_W_VALUES)))
    for i, ell in enumerate(ELL_W_VALUES):
        for j, a in enumerate(A_VALUES):
            r = next(x for x in runs if x["ell_w"] == ell and x["p"].a == a)
            ax = fig.add_subplot(len(ELL_W_VALUES), len(A_VALUES), i * len(A_VALUES) + j + 1, projection="3d")
            if np.any(r["u"] > 0.5):
                v, f = m3.isosurface(r["u"], r["p"].L, stride=2, cut=False)
                base = (0.93, 0.72, 0.25) if ell is None else ((0.45, 0.70, 0.90) if ell == ELL_W_VALUES[1] else (0.55, 0.80, 0.55))
                m3.draw_surface(ax, v, f, extent=0.5 * r["p"].L * 0.95, base=base)
            st = r["stats"]
            gate = "no gate" if ell is None else f"ℓ_w={ell}"
            ax.set_title(
                f"a={a}, {gate}, t={r['final_step'] * r['p'].dt:.0f} ({r['reason']})\n"
                f"|t·r̂|={st.get('radial_alignment', np.nan):.2f}  ℓp/ρ={st.get('persistence_over_rho', np.nan):.1f}  "
                f"Y={st.get('frac_Y', np.nan):.2f} T={st.get('frac_T', np.nan):.2f}  loops={st.get('loops_b1', -1)}",
                fontsize=9,
            )
            ax.set_axis_off()
    fig.suptitle("3D branching + mobility gate M(w): rows = gate length ℓ_w, columns = symmetry-breaking a", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Saved {out}")


def plot_metrics(runs: list[dict], theory: list[dict], out: Path) -> None:
    keys = [
        ("radial_alignment", "radial alignment ⟨|t·r̂|⟩ (random 0.5)"),
        ("persistence_over_rho", "persistence ℓ_p / ρ"),
        ("frac_Y", "fraction of symmetric Y junctions"),
        ("frac_T", "fraction of lateral T junctions"),
        ("loops_per_length", "loops (b1) per skeleton length"),
        ("frac_tube", "tube fraction of medial axis"),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(20, 8.5))
    axs = axs.ravel()
    for ax, (k, title) in zip(axs, keys):
        for ell, mk in zip(ELL_W_VALUES, ["o-", "s-", "^-"]):
            ys = [next(x for x in runs if x["ell_w"] == ell and x["p"].a == a)["stats"].get(k, np.nan) for a in A_VALUES]
            ax.plot(A_VALUES, ys, mk, label="no gate" if ell is None else f"ℓ_w={ell}")
        ax.set_xlabel("a")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax = axs[6]
    ag = [t["a"] for t in theory]
    ax.plot(ag, [t["sigma_rel_l2"] for t in theory], "-", label="σ_2 (split in two)")
    ax.plot(ag, [t["sigma_rel_l3"] for t in theory], "-", label="σ_3")
    ax.plot(ag, [t["sigma_rel_l4"] for t in theory], "-", label="σ_4")
    ax.axhline(0, c="gray", lw=0.6)
    ax.set_ylim(-0.03, 0.01)
    ax.set_xlabel("a")
    ax.set_title("theory: tip-cap shape modes at R=ρ_cyl(a)", fontsize=10)
    ax.legend(fontsize=8)
    ax = axs[7]
    for r in runs:
        ang = r["stats"].get("junction_angles")
        if ang is not None and len(ang) and r["ell_w"] is not None:
            ax.scatter(ang[:, 0], ang[:, 1], s=8, alpha=0.6, label=f"a={r['p'].a}, ℓ_w={r['ell_w']}")
    ax.plot([0, 90], [0, 90], "k:", lw=0.8)
    ax.set_xlabel("smaller child angle θ1 [deg]")
    ax.set_ylabel("larger child angle θ2 [deg]")
    ax.set_title("junction geometry (Y: near diagonal, T: θ1≈0)", fontsize=10)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Saved {out}")


def plot_best(r: dict, out: Path) -> None:
    keys = sorted(r["snaps"])
    pick = [keys[int(round(i * (len(keys) - 1) / 3))] for i in range(4)]
    fig = plt.figure(figsize=(20, 9.5))
    for i, k in enumerate(pick):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        snap = r["snaps"][k].astype(np.float32)
        if np.any(snap > 0.5):
            v, f = m3.isosurface(snap, r["p"].L, stride=1, cut=True)
            m3.draw_surface(ax, v, f, extent=0.5 * r["p"].L * 0.95, base=(0.45, 0.70, 0.90))
        ax.set_title(f"t={k * r['p'].dt:.0f}", fontsize=10)
        ax.set_axis_off()
    c = r["p"].n // 2
    ext = [-r["p"].L / 2, r["p"].L / 2, -r["p"].L / 2, r["p"].L / 2]
    ax = fig.add_subplot(2, 4, 5)
    ax.imshow(r["u"][:, :, c].T, origin="lower", cmap="viridis", extent=ext)
    ax.set_title("u, z-midplane (final)")
    if r["w_mid"] is not None:
        ax = fig.add_subplot(2, 4, 6)
        im = ax.imshow(r["w_mid"].T, origin="lower", cmap="magma", extent=ext)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"coarse-grained w (ℓ_w={r['ell_w']}), w_c={r['th']['w_c']:.3f}")
        ax = fig.add_subplot(2, 4, 7)
        im = ax.imshow(r["m_mid"].T, origin="lower", cmap="gray", extent=ext, vmin=0, vmax=1)
        ax.contour(np.linspace(ext[0], ext[1], r["u"].shape[0]), np.linspace(ext[2], ext[3], r["u"].shape[1]), r["u"][:, :, c].T, [0.5], colors="C1", linewidths=0.7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("mobility M(w) (white = active) + u=1/2")
    ax = fig.add_subplot(2, 4, 8)
    h = r["hist"]
    ax.plot(h["t"], h["R_env"], label="R_env")
    ax.plot(h["t"], h["R_max"], label="R_max")
    ax2 = ax.twinx()
    ax2.plot(h["t"], h["active_frac_iface"], "k:", label="active interface fraction")
    ax2.set_ylim(0, 1.05)
    ax.set_xlabel("t")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    st = r["stats"]
    fig.suptitle(
        f"best gated run: a={r['p'].a}, ℓ_w={r['ell_w']}  —  |t·r̂|={st.get('radial_alignment', np.nan):.2f}, "
        f"ℓp/ρ={st.get('persistence_over_rho', np.nan):.1f}, Y={st.get('frac_Y', np.nan):.2f}, T={st.get('frac_T', np.nan):.2f}, loops={st.get('loops_b1', -1)}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--n", type=int, default=256)
    ap_.add_argument("--steps", type=int, default=16000)
    args = ap_.parse_args()
    RESULTS.mkdir(exist_ok=True)
    RAW.mkdir(exist_ok=True)
    base = m3.Params3D(n=args.n, n_steps=args.steps)
    lam_star = 2 * math.pi / 2.5

    theory = [tip_theory(a) for a in np.round(np.arange(0.02, 0.0801, 0.005), 3)]
    for t in theory:
        print("theory", {k: round(float(v), 4) for k, v in t.items()}, flush=True)
    th_by_a = {a: tip_theory(a) for a in A_VALUES}

    runs = []
    for a in A_VALUES:
        for ell in ELL_W_VALUES:
            p = replace(base, a=a)
            label = f"a{a:.2f}_" + ("nogate" if ell is None else f"lw{ell:.1f}")
            r = run_one(p, ell, th_by_a[a], label)
            r["stats"] = analyze(r["u"], p, lam_star) if np.any(r["u"] > 0.5) else {}
            print(f"  [{label}] stats:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in _clean(r["stats"]).items()}, flush=True)
            runs.append(r)
            partial = [{"a": x["p"].a, "ell_w": x["ell_w"], "final_step": x["final_step"], "reason": x["reason"], "wall_s": x["wall"], **_clean(x["stats"])} for x in runs]
            (RESULTS / f"{PREFIX}_scan.json").write_text(json.dumps({"theory": theory, "runs": partial}, indent=1, ensure_ascii=False, default=float))

    plot_grid(runs, RESULTS / f"{PREFIX}_scan.png")
    plot_metrics(runs, theory, RESULTS / f"{PREFIX}_metrics.png")
    gated = [r for r in runs if r["ell_w"] is not None]
    best = max(gated, key=lambda r: score(r["stats"]))
    plot_best(best, RESULTS / f"{PREFIX}_best.png")
    print(f"best: a={best['p'].a}, ell_w={best['ell_w']}, score={score(best['stats']):.3f}")
    (RAW / "mobility_scan.DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("done")


if __name__ == "__main__":
    main()
