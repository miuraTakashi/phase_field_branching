"""
3D 分枝形態形成フェーズフィールド（phase_field_branching.py の 3 次元版）。

    ∂_t u = d_u Δu + u(1-u)(u + v - 1 - a),   d_u = ε² d
    ∂_t v = d   Δv + (1 - u - v)

a = 0 が元のモデル（2D 版・Mathematica ノートと同一）。a は (u,v)→(1-u,1-v) 対称性を
破る対照パラメータで、形態予測（皺 vs 枝）の検証にだけ使う。

数値スキームは 2D 版と同じ半陰的スペクトル法（反応陽・拡散陰、7 点ラプラシアンの FFT 核）:
    f = u + Δt u(1-u)(u+v-1-a),   g = v + Δt (1-u-v)
    u ← IRFFTN( RFFTN(f) / (1 + Δt d_u λ_h(k)) ),   v ← 同様（d_v = d）
    λ_h(k) = Σ_i (2 - 2 cos k_i Δx)/Δx²   （= .nb の 1/Fourier[kernel] と同値）

高速化: u,v を 1 本のバッチ FFT にまとめ、float32 化し、
  numpy（2D 版と同じ核 FFT・float64）/ scipy.fft（workers 並列）/ torch CPU / torch CUDA
を切り替えて比較する（--bench）。

数理解析（シャープ界面・準定常 v、遮蔽長 ℓ = √d、c = √(2 d_u)）:
  界面法線速度  V_n = c (v_I - 1/2 - a) - d_u H      （H: 主曲率の和、凸で正）
  v_I - 1/2 ≈ (ℓ/4) H   ⇒ 有効表面張力 d_u - cℓ/4 が負（2√2 ε < 1）なら曲率不安定
  球 R の成長・球面調和モード Y_lm の成長率は変形球ベッセル関数 i_l, k_l で厳密に書ける。
  形態: a = 0 では界面の振幅方程式が h → -h 対称（2 次項なし）⇒ 表面は縞（皺）を選び、
        定常ラメラ（H = 0, v_I = 1/2）が φ = 1/2 で中立 ⇒ 半径方向に伸びるシート状の皺。
        a ≠ 0 では 2 次項が生じ斑点（先端）⇒ 管状の枝。

Usage:
    python3 phase_field_branching_3d.py             # 解析 + a=0 / a>0 の 3D 計算 + 図
    python3 phase_field_branching_3d.py --bench     # バックエンド速度比較も行う
    python3 phase_field_branching_3d.py --quick     # 128³ の短縮版
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import fft as sfft
from scipy import ndimage, optimize, special
from scipy.spatial import cKDTree

for _d in Path(__file__).resolve().parents:
    if (_d / "atlas_plotting.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
else:
    raise ImportError("atlas_plotting.py not found above " + str(__file__))
import atlas_plotting as ap

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import phase_field_branching as pb2d  # noqa: E402

RESULTS = HERE / "results"
RAW = HERE / "results_3d_raw"
PREFIX = "phase_field_branching_3d"


@dataclass(frozen=True)
class Params3D:
    n: int = 256
    dx: float = 0.1
    eps: float = 0.2
    d: float = 0.1
    a: float = 0.0
    dt: float = 0.5
    n_steps: int = 12000
    r0: float = 1.0
    noise: float = 1.0  # 2D 版と同じ: 格子点ごとの半径揺らぎ U(-noise, noise)（格子単位）
    seed: int = 42

    @property
    def L(self) -> float:
        return self.n * self.dx

    @property
    def du(self) -> float:
        return self.eps**2 * self.d


# ---------------------------------------------------------------------------
# 数理解析（シャープ界面極限）
# ---------------------------------------------------------------------------


def sharp_constants(eps: float, d: float) -> tuple[float, float, float]:
    du = eps**2 * d
    return du, math.sqrt(2.0 * du), math.sqrt(d)


def instability_number(eps: float) -> float:
    """Γ = 4 d_u / (c ℓ) = 2√2 ε。Γ < 1 で有効表面張力が負（曲率不安定）。"""
    return 2.0 * math.sqrt(2.0) * eps


def planar_sigma_sharp(k: np.ndarray, eps: float, d: float) -> np.ndarray:
    """平面界面 z = h e^{ikx} の成長率（a=0、準定常遮蔽拡散）。"""
    du, c, ell = sharp_constants(eps, d)
    k = np.asarray(k, dtype=float)
    return c / (2.0 * ell) * (1.0 - 1.0 / np.sqrt(1.0 + (k * ell) ** 2)) - du * k**2


def planar_kmax_sharp(eps: float) -> float:
    """dσ/dk = 0 ⇒ (1 + K²)^{-3/2} = Γ, K = kℓ。Γ ≥ 1 なら NaN。"""
    gam = instability_number(eps)
    if gam >= 1.0:
        return float("nan")
    return math.sqrt(gam ** (-2.0 / 3.0) - 1.0)


def _log_deriv_i(l: int | np.ndarray, x: float) -> np.ndarray:
    return special.spherical_in(l, x, derivative=True) / special.spherical_in(l, x)


def _log_deriv_k(l: int | np.ndarray, x: float) -> np.ndarray:
    return special.spherical_kn(l, x, derivative=True) / special.spherical_kn(l, x)


def sphere_interface_v(R: float, d: float) -> tuple[float, float]:
    """球 (u=1, 半径 R) の界面での準定常 v と内側の対数微分 p（x = R/ℓ 単位）。"""
    ell = math.sqrt(d)
    x = R / ell
    p = float(_log_deriv_i(0, x))
    s = float(-_log_deriv_k(0, x))
    return s / (p + s), p


def sphere_dRdt(R: float, eps: float, d: float, a: float = 0.0) -> float:
    du, c, _ell = sharp_constants(eps, d)
    v_i, _p = sphere_interface_v(R, d)
    return c * (v_i - 0.5 - a) - 2.0 * du / R


def sphere_sigma_l(l: np.ndarray, R: float, eps: float, d: float, a: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """球面調和摂動 R + δ Y_lm の成長率。戻り値: (絶対 σ_l, 形状相対 σ_l - Ṙ/R)。

    δ による u=1 領域の張り出しを v 方程式の面源 -δ Y_lm δ(r-R) とみなし、
    v_1 = C g_l(r) Y_lm（内 i_l, 外 k_l）で解く: C = -δ / (ℓ W_l)、
    W_l = i_l'/i_l - k_l'/k_l。l=1（平行移動）で σ_1 = 0 となる。
    """
    du, c, ell = sharp_constants(eps, d)
    l = np.asarray(l, dtype=int)
    x = R / ell
    v_i, p = sphere_interface_v(R, d)
    dv0 = p * v_i / ell
    w = _log_deriv_i(l, x) - _log_deriv_k(l, x)
    sig = c * (dv0 - 1.0 / (ell * w)) - du * (l - 1) * (l + 2) / R**2
    return sig, sig - sphere_dRdt(R, eps, d, a) / R


def integrate_sphere_radius(t: np.ndarray, r0: float, eps: float, d: float, a: float = 0.0) -> np.ndarray:
    out = np.empty_like(t, dtype=float)
    r = r0
    out[0] = r
    for i in range(1, t.size):
        h = t[i] - t[i - 1]
        k1 = sphere_dRdt(r, eps, d, a)
        k2 = sphere_dRdt(max(r + 0.5 * h * k1, 1e-3), eps, d, a)
        r = max(r + h * k2, 1e-3)
        out[i] = r
    return out


def diffuse_planar_dispersion(eps: float, d: float, k: np.ndarray) -> np.ndarray:
    """拡散界面（実モデル）の平面界面 max Re λ(k)（2D 版の Newton + 固有値を再利用、a=0）。"""
    x, u0, v0, ok = pb2d.steady_planar_front_1d(eps, d)
    if not ok:
        return np.full_like(k, np.nan, dtype=float)
    return pb2d.linearized_front_max_Re_lambda(x, u0, v0, eps, d, np.asarray(k, dtype=float))


def lamella_interface_v(phi: float, period: float, d: float) -> float:
    """ラメラ（u=1 幅 φP, u=0 幅 (1-φ)P）の界面での準定常 v = T0/(T0+T1)、T_i = tanh(半幅/ℓ)。"""
    ell = math.sqrt(d)
    t1 = math.tanh(0.5 * phi * period / ell)
    t0 = math.tanh(0.5 * (1.0 - phi) * period / ell)
    return t0 / (t0 + t1)


def lamella_fraction(a: float, period: float, d: float) -> float:
    """周期 period のラメラで界面 v = 1/2 + a となる u=1 体積分率 φ（平坦界面、H=0）。"""
    target = 0.5 + a

    def resid(phi: float) -> float:
        return lamella_interface_v(phi, period, d) - target

    try:
        return float(optimize.brentq(resid, 0.02, 0.98, xtol=1e-5))
    except ValueError:
        return float("nan")


def theory_summary(p: Params3D, k_diff: np.ndarray, lam_diff: np.ndarray) -> dict:
    du, c, ell = sharp_constants(p.eps, p.d)
    K = planar_kmax_sharp(p.eps)
    k_sharp = K / ell if np.isfinite(K) else float("nan")
    i = int(np.nanargmax(lam_diff))
    k_d = float(k_diff[i])
    return {
        "du": du,
        "c": c,
        "ell": ell,
        "Gamma_2sqrt2_eps": instability_number(p.eps),
        "sharp_unstable": instability_number(p.eps) < 1.0,
        "k_star_sharp": k_sharp,
        "lambda_star_sharp": 2 * math.pi / k_sharp if np.isfinite(k_sharp) else float("nan"),
        "k_star_diffuse": k_d,
        "lambda_star_diffuse": 2 * math.pi / k_d,
        "sigma_max_diffuse": float(lam_diff[i]),
        "sphere_dRdt_R0": sphere_dRdt(p.r0, p.eps, p.d, p.a),
        "phi_lamella": lamella_fraction(p.a, 2 * math.pi / k_d, p.d),
    }


# ---------------------------------------------------------------------------
# 半陰的スペクトル法ソルバ
# ---------------------------------------------------------------------------


def laplacian_symbol_rfft(n: int, dx: float) -> np.ndarray:
    """7 点ラプラシアンの正値シンボル λ_h(k)（rfftn レイアウト）。"""
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kr = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)
    l1 = (2.0 - 2.0 * np.cos(k * dx)) / dx**2
    lr = (2.0 - 2.0 * np.cos(kr * dx)) / dx**2
    return l1[:, None, None] + l1[None, :, None] + lr[None, None, :]


def kernel_multiplier_like_2d(n: int, dx: float, dt: float, D: float) -> np.ndarray:
    """2D 版 / .nb と同じ作り方: 1 / RFFTN(陰的 7 点核)。"""
    cu = dt * D / dx**2
    ker = np.zeros((n, n, n))
    ker[0, 0, 0] = 1.0 + 6.0 * cu
    for ax in range(3):
        for s in (1, -1):
            idx = [0, 0, 0]
            idx[ax] = s
            ker[tuple(idx)] = -cu
    return 1.0 / np.fft.rfftn(ker)


def initial_condition(p: Params3D) -> tuple[np.ndarray, np.ndarray]:
    """球（半径 r0）＋格子点ごとの半径揺らぎ（2D 版の円＋揺らぎを 3D 化）。"""
    rng = np.random.default_rng(p.seed)
    n = p.n
    idx = np.arange(1, n + 1, dtype=np.float32) - n / 2.0
    r2 = idx[:, None, None] ** 2 + idx[None, :, None] ** 2 + idx[None, None, :] ** 2
    noisy = p.r0 / p.dx + rng.uniform(-p.noise, p.noise, size=(n, n, n)).astype(np.float32)
    u = (r2 < noisy**2).astype(np.float32)
    v = np.zeros((n, n, n), dtype=np.float32)
    return u, v


class Solver:
    """backend: numpy64 | scipy32 | torch_cpu | torch_cuda | torch_cuda_compile"""

    def __init__(self, p: Params3D, backend: str = "torch_cuda", threads: int = -1):
        self.p = p
        self.backend = backend
        n = p.n
        lam = laplacian_symbol_rfft(n, p.dx)
        mult = np.stack(
            [1.0 / (1.0 + p.dt * p.du * lam), 1.0 / (1.0 + p.dt * p.d * lam)]
        )
        u0, v0 = initial_condition(p)
        self.threads = threads
        if backend == "numpy64":
            self.mu = kernel_multiplier_like_2d(n, p.dx, p.dt, p.du)
            self.mv = kernel_multiplier_like_2d(n, p.dx, p.dt, p.d)
            self.u = u0.astype(np.float64)
            self.v = v0.astype(np.float64)
        elif backend == "scipy32":
            self.mult = mult.astype(np.float32)
            self.uv = np.stack([u0, v0]).astype(np.float32)
        elif backend.startswith("torch"):
            import torch

            self.torch = torch
            dev = "cuda" if "cuda" in backend else "cpu"
            if dev == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            if dev == "cpu" and threads > 0:
                torch.set_num_threads(threads)
            self.dev = dev
            self.mult = torch.as_tensor(mult, dtype=torch.float32, device=dev)
            self.uv = torch.as_tensor(np.stack([u0, v0]), dtype=torch.float32, device=dev)
            react = self._torch_react
            if backend.endswith("compile"):
                react = torch.compile(self._torch_react, dynamic=False)
            self._react = react
        else:
            raise ValueError(backend)
        self.step_count = 0

    def _torch_react(self, uv):
        p = self.p
        u = uv[0]
        v = uv[1]
        f = u + p.dt * (u * (1.0 - u) * (u + v - 1.0 - p.a))
        g = v + p.dt * (1.0 - u - v)
        return self.torch.stack((f, g))

    def step(self, k: int = 1) -> None:
        p = self.p
        n = p.n
        b = self.backend
        for _ in range(k):
            if b == "numpy64":
                u, v = self.u, self.v
                f = u + p.dt * u * (1.0 - u) * (u + v - 1.0 - p.a)
                g = v + p.dt * (1.0 - u - v)
                ax3 = (0, 1, 2)
                self.u = np.fft.irfftn(np.fft.rfftn(f, axes=ax3) * self.mu, s=(n, n, n), axes=ax3)
                self.v = np.fft.irfftn(np.fft.rfftn(g, axes=ax3) * self.mv, s=(n, n, n), axes=ax3)
            elif b == "scipy32":
                uv = self.uv
                u, v = uv[0], uv[1]
                fg = np.empty_like(uv)
                fg[0] = u + np.float32(p.dt) * u * (1 - u) * (u + v - np.float32(1 + p.a))
                fg[1] = v + np.float32(p.dt) * (1 - u - v)
                spec = sfft.rfftn(fg, axes=(1, 2, 3), workers=self.threads)
                spec *= self.mult
                self.uv = sfft.irfftn(spec, s=(n, n, n), axes=(1, 2, 3), workers=self.threads).astype(np.float32, copy=False)
            else:
                torch = self.torch
                fg = self._react(self.uv)
                spec = torch.fft.rfftn(fg, dim=(1, 2, 3))
                spec *= self.mult
                self.uv = torch.fft.irfftn(spec, s=(n, n, n), dim=(1, 2, 3))
        self.step_count += k

    def sync(self) -> None:
        if self.backend.startswith("torch") and self.dev == "cuda":
            self.torch.cuda.synchronize()

    def u_host(self) -> np.ndarray:
        if self.backend == "numpy64":
            return self.u.astype(np.float32)
        if self.backend == "scipy32":
            return self.uv[0].copy()
        return self.uv[0].detach().float().cpu().numpy()

    def v_host(self) -> np.ndarray:
        if self.backend == "numpy64":
            return self.v.astype(np.float32)
        if self.backend == "scipy32":
            return self.uv[1].copy()
        return self.uv[1].detach().float().cpu().numpy()


def benchmark(ns: tuple[int, ...] = (128, 256), steps: int = 30) -> dict:
    """1 ステップあたりの壁時計時間（ms）。"""
    backends = ["numpy64", "scipy32", "torch_cpu", "torch_cuda", "torch_cuda_compile"]
    out: dict[str, dict[str, float]] = {}
    for n in ns:
        out[str(n)] = {}
        for b in backends:
            if b == "numpy64" and n > 192:
                st = max(4, steps // 6)
            else:
                st = steps
            try:
                s = Solver(Params3D(n=n), backend=b)
                s.step(3)
                s.sync()
                t0 = time.perf_counter()
                s.step(st)
                s.sync()
                ms = 1e3 * (time.perf_counter() - t0) / st
                out[str(n)][b] = ms
                print(f"  bench n={n:4d} {b:20s} {ms:9.2f} ms/step", flush=True)
                del s
            except Exception as exc:  # noqa: BLE001
                print(f"  bench n={n:4d} {b:20s} FAILED: {exc}", flush=True)
                out[str(n)][b] = float("nan")
    return out


def check_scheme_equivalence(n: int = 32) -> float:
    """numpy64（2D 版と同じ核 FFT）と torch/scipy（解析シンボル）の 1 ステップ差。"""
    p = Params3D(n=n, r0=0.8)
    a = Solver(p, "numpy64")
    b = Solver(p, "scipy32")
    a.step(20)
    b.step(20)
    return float(np.max(np.abs(a.u_host() - b.u_host())))


# ---------------------------------------------------------------------------
# 計測
# ---------------------------------------------------------------------------


def fibonacci_directions(m: int) -> np.ndarray:
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


class InterfaceProbe:
    """中心からのレイで最外の u=1/2 交点 r(Ω) を求め、実球面調和関数で展開する。"""

    def __init__(self, p: Params3D, n_dir: int = 4000, lmax: int = 40):
        self.p = p
        self.dirs = fibonacci_directions(n_dir)
        self.lmax = lmax
        self.rs = np.arange(0.0, 0.5 * p.L - p.dx, 0.5 * p.dx)
        c = p.n / 2.0 - 1.0
        pts = c + (self.rs[None, :, None] * self.dirs[:, None, :]) / p.dx
        self.coords = pts.reshape(-1, 3).T.copy()
        polar = np.arccos(np.clip(self.dirs[:, 2], -1, 1))
        azim = np.arctan2(self.dirs[:, 1], self.dirs[:, 0])
        cols = []
        self.l_of_col = []
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                y = special.sph_harm_y(l, abs(m), polar, azim)
                if m < 0:
                    col = math.sqrt(2) * y.imag
                elif m == 0:
                    col = y.real
                else:
                    col = math.sqrt(2) * y.real
                cols.append(col)
                self.l_of_col.append(l)
        self.basis = np.stack(cols, axis=1)
        self.pinv = np.linalg.pinv(self.basis)
        self.l_of_col = np.asarray(self.l_of_col)

    def radii(self, u: np.ndarray) -> np.ndarray:
        s = ndimage.map_coordinates(u, self.coords, order=1, mode="wrap").reshape(len(self.dirs), -1)
        inside = s > 0.5
        any_in = inside.any(axis=1)
        last = inside.shape[1] - 1 - np.argmax(inside[:, ::-1], axis=1)
        last = np.minimum(last, inside.shape[1] - 2)
        rows = np.arange(len(self.dirs))
        s0 = s[rows, last]
        s1 = s[rows, last + 1]
        frac = np.clip((s0 - 0.5) / np.maximum(s0 - s1, 1e-6), 0, 1)
        r = self.rs[last] + frac * (self.rs[1] - self.rs[0])
        return np.where(any_in, r, 0.0)

    def spectrum(self, r: np.ndarray) -> np.ndarray:
        coef = self.pinv @ r
        return np.bincount(self.l_of_col, weights=coef**2, minlength=self.lmax + 1)


def curvature_stats(u: np.ndarray, mask_region: np.ndarray, dx: float, sigma: float = 1.0) -> dict:
    """界面ボクセルで平均曲率の和 H（u=1 側が凸で正）とガウス曲率 K。"""
    us = ndimage.gaussian_filter(u.astype(np.float32), sigma, mode="wrap")
    g = np.gradient(us, dx)
    gn = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
    sel = (np.abs(us - 0.5) < 0.12) & mask_region & (gn > 0.5 * np.percentile(gn[mask_region], 90))
    idx = np.nonzero(sel)
    gx, gy, gz = (gi[idx] for gi in g)
    hxx = np.gradient(g[0], dx, axis=0)[idx]
    hyy = np.gradient(g[1], dx, axis=1)[idx]
    hzz = np.gradient(g[2], dx, axis=2)[idx]
    hxy = np.gradient(g[0], dx, axis=1)[idx]
    hxz = np.gradient(g[0], dx, axis=2)[idx]
    hyz = np.gradient(g[1], dx, axis=2)[idx]
    n2 = gx**2 + gy**2 + gz**2
    quad = gx * gx * hxx + gy * gy * hyy + gz * gz * hzz + 2 * (gx * gy * hxy + gx * gz * hxz + gy * gz * hyz)
    lap = hxx + hyy + hzz
    H = -(lap * n2 - quad) / n2**1.5
    adj = (
        gx * gx * (hyy * hzz - hyz**2)
        + gy * gy * (hxx * hzz - hxz**2)
        + gz * gz * (hxx * hyy - hxy**2)
        + 2 * gx * gy * (hxz * hyz - hxy * hzz)
        + 2 * gx * gz * (hxy * hyz - hxz * hyy)
        + 2 * gy * gz * (hxy * hxz - hxx * hyz)
    )
    K = adj / n2**2
    return {"H": H, "K": K}


def medial_shape_stats(mask: np.ndarray, region: np.ndarray, dx: float, radius: float, rng_seed: int = 0) -> dict:
    """距離変換の稜（中軸）を PCA し、シート状（面）か管状（線）かを判定。"""
    dist = ndimage.distance_transform_edt(mask)
    ridge = (dist == ndimage.maximum_filter(dist, size=3)) & mask & region & (dist >= 2.0)
    pts = np.argwhere(ridge).astype(np.float32) * dx
    if len(pts) < 50:
        return {"n_ridge": int(len(pts))}
    tree = cKDTree(pts)
    rng = np.random.default_rng(rng_seed)
    sample = pts[rng.choice(len(pts), size=min(3000, len(pts)), replace=False)]
    lin, pla = [], []
    for q in sample:
        nb = tree.query_ball_point(q, r=radius)
        if len(nb) < 12:
            continue
        x = pts[nb] - pts[nb].mean(axis=0)
        e = np.sort(np.linalg.eigvalsh(x.T @ x / len(nb)))[::-1]
        lin.append((e[0] - e[1]) / e[0])
        pla.append((e[1] - e[2]) / e[0])
    lin = np.asarray(lin)
    pla = np.asarray(pla)
    return {
        "n_ridge": int(len(pts)),
        "linearity": lin,
        "planarity": pla,
        "frac_sheet": float(np.mean(pla > lin)),
        "frac_tube": float(np.mean(lin > pla)),
        "thickness_mode": float(2 * dx * np.median(dist[ridge])),
    }


def shell_region(p: Params3D, r_dirs: np.ndarray, inner: float = 2.0, margin: float = 1.0) -> tuple[np.ndarray, float, float]:
    n = p.n
    idx = (np.arange(n) - (n / 2.0 - 1.0)) * p.dx
    r = np.sqrt(idx[:, None, None] ** 2 + idx[None, :, None] ** 2 + idx[None, None, :] ** 2)
    hit = r_dirs[r_dirs > 0]
    r_out = 0.85 * float(np.median(hit)) - margin if hit.size else 0.0
    if r_out < inner + 1.0:
        inner = 0.0
        r_out = max(float(np.median(hit)) if hit.size else 1.0, 4 * p.dx)
    return (r > inner) & (r < r_out), inner, r_out


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


def run_case(p: Params3D, label: str, backend: str, diag_every: int = 50, snap_steps: tuple[int, ...] = ()) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    solver = Solver(p, backend=backend)
    probe = InterfaceProbe(p)
    hist = {"t": [], "R_mean": [], "R_min": [], "R_max": [], "R_env": [], "R_vol": [], "phi_in": [], "spec": []}
    snaps: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    stop_r = 0.46 * p.L
    step = 0
    while True:
        u = solver.u_host()
        r = probe.radii(u)
        vol = float(np.count_nonzero(u > 0.5)) * p.dx**3
        hist["t"].append(step * p.dt)
        hist["R_mean"].append(float(r.mean()))
        hist["R_min"].append(float(r.min()))
        hist["R_max"].append(float(r.max()))
        r_env = float(np.percentile(r[r > 0], 90)) if np.any(r > 0) else 0.0
        hist["R_env"].append(r_env)
        hist["R_vol"].append((3 * vol / (4 * math.pi)) ** (1 / 3))
        hist["phi_in"].append(vol / (4 / 3 * math.pi * max(r_env, 1e-6) ** 3))
        hist["spec"].append(probe.spectrum(r))
        if step in snap_steps:
            snaps[step] = u[::2, ::2, ::2].astype(np.float16)
        if step % (diag_every * 20) == 0:
            el = time.perf_counter() - t0
            print(
                f"  [{label}] step {step:6d} t={step * p.dt:7.1f} R_mean={r.mean():6.2f} "
                f"R_max={r.max():6.2f} phi={hist['phi_in'][-1]:.3f}  ({el:6.1f}s)",
                flush=True,
            )
        if step >= p.n_steps or r.max() > stop_r:
            break
        solver.step(diag_every)
        step += diag_every
    solver.sync()
    wall = time.perf_counter() - t0
    u_final = solver.u_host()
    v_final = solver.v_host()
    snaps[step] = u_final[::2, ::2, ::2].astype(np.float16)
    np.savez_compressed(
        RAW / f"{label}.npz",
        u=u_final.astype(np.float16),
        v=v_final.astype(np.float16),
        snap_steps=np.array(sorted(snaps)),
        snaps=np.stack([snaps[k] for k in sorted(snaps)]),
        **{k: np.asarray(v) for k, v in hist.items()},
        params=json.dumps(asdict(p)),
    )
    print(f"  [{label}] done: {step} steps, wall {wall:.1f}s ({1e3 * wall / max(step, 1):.2f} ms/step incl. diagnostics)", flush=True)
    return {
        "p": p,
        "label": label,
        "hist": {k: np.asarray(v) for k, v in hist.items()},
        "u": u_final,
        "v": v_final,
        "snaps": snaps,
        "final_step": step,
        "wall": wall,
        "probe": probe,
        "r_final": probe.radii(u_final),
    }


def analyze_morphology(res: dict, lam_star: float) -> dict:
    p = res["p"]
    u = res["u"]
    region, r_in, r_out = shell_region(p, res["r_final"])
    us = ndimage.gaussian_filter(u, 0.8, mode="wrap")
    m1 = us > 0.5
    phi = float(m1[region].mean())
    curv = curvature_stats(u, region, p.dx)
    k_star = 2 * math.pi / lam_star
    H = curv["H"] / k_star
    K = curv["K"] / k_star**2
    med1 = medial_shape_stats(m1, region, p.dx, radius=0.45 * lam_star)
    med0 = medial_shape_stats(~m1, region, p.dx, radius=0.45 * lam_star)
    ell_frac = float(np.mean((K > 0.02) & (H > 0)))
    saddle_frac = float(np.mean(K < -0.02))
    summary = {
        "shell_r_in": r_in,
        "shell_r_out": r_out,
        "phi_u1_shell": phi,
        "H_over_k_median": float(np.median(H)),
        "frac_elliptic_convex_K>0": ell_frac,
        "frac_saddle_K<0": saddle_frac,
        "u1_frac_sheet": med1.get("frac_sheet", float("nan")),
        "u1_frac_tube": med1.get("frac_tube", float("nan")),
        "u0_frac_sheet": med0.get("frac_sheet", float("nan")),
        "u0_frac_tube": med0.get("frac_tube", float("nan")),
        "u1_thickness": med1.get("thickness_mode", float("nan")),
        "u0_thickness": med0.get("thickness_mode", float("nan")),
    }
    return {"summary": summary, "H": H, "K": K, "med1": med1, "med0": med0}


def measured_growth_rates(res: dict, t_window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """形状相対振幅 A_l = sqrt(P_l)/R_mean の対数勾配（線形段階の窓で最小二乗）。"""
    h = res["hist"]
    t = h["t"]
    sel = (t >= t_window[0]) & (t <= t_window[1])
    A = np.sqrt(np.maximum(h["spec"][sel], 1e-30)) / h["R_mean"][sel, None]
    ls = np.arange(A.shape[1])
    rates = np.array([np.polyfit(t[sel], np.log(A[:, l]), 1)[0] for l in ls])
    return ls, rates


# ---------------------------------------------------------------------------
# 描画
# ---------------------------------------------------------------------------


def isosurface(u: np.ndarray, L: float, *, stride: int = 2, sigma: float = 1.0, cut: bool = True, level: float = 0.5):
    from skimage.measure import marching_cubes

    us = ndimage.gaussian_filter(u.astype(np.float32), sigma, mode="wrap")[::stride, ::stride, ::stride]
    n = us.shape[0]
    if cut:
        c = n // 2
        us = us.copy()
        us[c:, :c, c:] = 0.0
    us[0, :, :] = us[-1, :, :] = 0.0
    us[:, 0, :] = us[:, -1, :] = 0.0
    us[:, :, 0] = us[:, :, -1] = 0.0
    h = L / n
    verts, faces, normals, _ = marching_cubes(us, level=level, spacing=(h, h, h))
    return verts - 0.5 * L, faces


def _shade(verts: np.ndarray, faces: np.ndarray, base: np.ndarray) -> np.ndarray:
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    light = np.array([0.45, -0.35, 0.82])
    light /= np.linalg.norm(light)
    lam = 0.28 + 0.72 * np.abs(nrm @ light)
    return np.clip(base[None, :] * lam[:, None], 0, 1)


def draw_surface(ax, verts, faces, extent: float, base=(0.93, 0.72, 0.25), title: str = "", elev=24, azim=-58):
    col = Poly3DCollection(verts[faces], facecolors=_shade(verts, faces, np.asarray(base)), edgecolor="none", linewidths=0)
    ax.add_collection3d(col)
    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(-extent, extent)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
    return col


def plot_theory(p: Params3D, k_diff: np.ndarray, lam_diff: np.ndarray, th: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    ax = axes[0]
    k = np.linspace(0.0, 4.5, 300)
    for eps in (0.2, 0.3):
        ax.plot(k, planar_sigma_sharp(k, eps, p.d), lw=1.4, label=rf"sharp, $\varepsilon$={eps}")
    ax.plot(k_diff, lam_diff, "k-", lw=1.6, label=r"diffuse (1D eigen), $\varepsilon$=0.2")
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\sigma(k)$")
    ax.set_title("planar interface: screened-diffusion instability")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    Rs = np.linspace(0.6, 12, 200)
    for a in (0.0, 0.02, 0.04, 0.06):
        ax.plot(Rs, [sphere_dRdt(R, p.eps, p.d, a) for R in Rs], lw=1.3, label=f"a={a}")
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("sphere radius R")
    ax.set_ylabel(r"$\dot R$ (sharp interface)")
    ax.set_title(r"sphere growth $\dot R=c(v_I-1/2-a)-2d_u/R$")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ls = np.arange(1, 41)
    Rg = np.linspace(0.8, 12, 120)
    S = np.array([sphere_sigma_l(ls, R, p.eps, p.d)[1] for R in Rg])
    vmax = np.nanmax(np.abs(S))
    im = ax.pcolormesh(Rg, ls, S.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax.contour(Rg, ls, S.T, levels=[0], colors="k", linewidths=0.8)
    ax.plot(Rg, Rg * th["k_star_sharp"], "w--", lw=1.0, label=r"$l=k^*_{sharp}R$")
    ax.plot(Rg, Rg * th["k_star_diffuse"], "y-", lw=1.2, label=r"$l=k^*_{diff}R$")
    ax.set_ylim(1, 40)
    ax.set_xlabel("R")
    ax.set_ylabel("spherical-harmonic degree l")
    ax.set_title(r"shape growth rate $\sigma_l-\dot R/R$ (sharp, exact $i_l,k_l$)")
    ax.legend(fontsize=7, loc="upper left")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[3]
    a_vals = np.linspace(-0.08, 0.08, 33)
    lam = th["lambda_star_diffuse"]
    phis = [lamella_fraction(a, lam, p.d) for a in a_vals]
    ax.plot(a_vals, phis, "C3-", lw=1.6)
    ax.axhline(0.5, color="0.5", lw=0.8, ls="--")
    ax.axvspan(-0.005, 0.005, color="C2", alpha=0.25)
    ax.text(0.0, 0.62, "a=0: symmetric\nno quadratic term\n→ stripes → WRINKLES\n(sheets, φ=1/2)", ha="center", fontsize=7.5)
    ax.text(0.05, 0.25, "a>0: φ<1/2\nquadratic term\n→ spots → BRANCHES\n(tubes)", ha="center", fontsize=7.5)
    ax.set_xlabel("symmetry-breaking a")
    ax.set_ylabel(r"lamella fraction $\varphi$ ($v_I=1/2+a$)")
    ax.set_title("morphology prediction")
    ax.grid(alpha=0.3)
    fig.suptitle(
        rf"3D branching phase field — analysis ($\varepsilon$={p.eps}, d={p.d}, $\Gamma=2\sqrt{{2}}\varepsilon$={th['Gamma_2sqrt2_eps']:.3f})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_evolution(results: list[dict], out: Path, n_cols: int = 4) -> None:
    fig = plt.figure(figsize=(4.2 * n_cols, 4.4 * len(results) * 2))
    for row, res in enumerate(results):
        p = res["p"]
        steps = sorted(res["snaps"])
        pick = [steps[int(round(i))] for i in np.linspace(0, len(steps) - 1, n_cols)]
        ext = 0.5 * p.L * 0.95
        for j, st in enumerate(pick):
            u = res["snaps"][st].astype(np.float32)
            ax = fig.add_subplot(2 * len(results), n_cols, 2 * row * n_cols + j + 1, projection="3d")
            try:
                v, f = isosurface(u, p.L, stride=1, sigma=0.7)
                r_max = float(np.max(np.linalg.norm(v, axis=1))) * 1.05
                draw_surface(ax, v, f, max(min(ext, r_max), 1.5), title=f"{res['label']}  t={st * p.dt:.0f}")
            except Exception:  # noqa: BLE001
                ax.set_axis_off()
            ax2 = fig.add_subplot(2 * len(results), n_cols, (2 * row + 1) * n_cols + j + 1)
            c = u.shape[0] // 2 - 1
            ap.atlas_imshow(ax2, u[:, :, c].T, heatmap="scalar", origin="lower", interpolation="nearest", vmin=0, vmax=1)
            ax2.set_title(f"z-midplane u, t={st * p.dt:.0f}", fontsize=8)
            ax2.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_validation(results: list[dict], th: dict, p: Params3D, out: Path, windows: dict) -> dict:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    rates_out = {}
    for ci, res in enumerate(results):
        h = res["hist"]
        pr = res["p"]
        t = h["t"]
        col = f"C{ci}"
        ax = axes[0, 0]
        ax.plot(t, h["R_mean"], color=col, lw=1.5, label=f"{res['label']}: mean ray radius")
        ax.plot(t, h["R_env"], color=col, lw=1.0, ls="-.", label=f"{res['label']}: envelope (90th pct)")
        ax.fill_between(t, h["R_min"], h["R_max"], color=col, alpha=0.15)
        ax.plot(t, integrate_sphere_radius(t, pr.r0, pr.eps, pr.d, pr.a), color=col, ls="--", lw=1.0, label=f"{res['label']}: smooth-sphere theory")
        ax = axes[0, 1]
        spec = h["spec"]
        ldom = 2 + np.argmax(spec[:, 2:], axis=1)
        ax.plot(h["R_mean"], ldom, "o", ms=2.5, color=col, label=f"{res['label']}: measured $l_{{dom}}$")
        ax = axes[0, 2]
        ax.plot(t, h["phi_in"], color=col, lw=1.4, label=res["label"])
        ls, rates = measured_growth_rates(res, windows[res["label"]])
        rates_out[res["label"]] = (ls, rates)
    ax = axes[0, 0]
    ax.set_xlabel("t")
    ax.set_ylabel("front radius")
    ax.set_title("front radius (band: min–max over directions)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[0, 1]
    Rg = np.linspace(0.8, max(r["hist"]["R_mean"].max() for r in results), 100)
    ax.plot(Rg, th["k_star_diffuse"] * Rg, "k-", lw=1.2, label=r"$l^*=k^*_{diff}R$")
    ax.plot(Rg, th["k_star_sharp"] * Rg, "k--", lw=1.0, label=r"$l^*=k^*_{sharp}R$")
    lstar_exact = [1 + int(np.argmax(sphere_sigma_l(np.arange(1, 81), R, p.eps, p.d)[1])) for R in Rg]
    ax.plot(Rg, lstar_exact, "k:", lw=1.2, label=r"$\arg\max_l$ sharp sphere")
    ax.set_xlabel("R_mean")
    ax.set_ylabel("dominant degree l")
    ax.set_title("dominant spherical-harmonic degree vs prediction")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[0, 2]
    ax.axhline(0.5, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$V(u>1/2)\,/\,(4\pi R_{env}^3/3)$")
    ax.set_title("filling fraction inside front")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    res0 = results[0]
    h = res0["hist"]
    ax = axes[1, 0]
    ts = h["t"]
    picks = np.linspace(0, len(ts) - 1, 7).astype(int)
    cmap = plt.get_cmap("viridis")
    for j, i in enumerate(picks):
        ax.semilogy(np.arange(h["spec"].shape[1]), np.maximum(h["spec"][i], 1e-12), color=cmap(j / 6), lw=1.1, label=f"t={ts[i]:.0f}, R={h['R_mean'][i]:.1f}")
    ax.set_xlabel("l")
    ax.set_ylabel(r"$P_l=\sum_m a_{lm}^2$")
    ax.set_title(f"{res0['label']}: interface spectrum")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    w = windows[res0["label"]]
    sel = (h["t"] >= w[0]) & (h["t"] <= w[1])
    Rw = float(np.mean(h["R_mean"][sel]))
    ls, rates = rates_out[res0["label"]]
    ax.plot(ls[2:], rates[2:], "o-", ms=3, color="C0", label=f"measured, t∈[{w[0]:.0f},{w[1]:.0f}]")
    lt = np.arange(2, ls[-1] + 1)
    ax.plot(lt, sphere_sigma_l(lt, Rw, p.eps, p.d)[1], "k--", lw=1.1, label=f"sharp sphere, R={Rw:.2f}")
    kk = np.sqrt((lt - 1) * (lt + 2)) / Rw
    lam_loc = np.interp(kk, windows["k_diff"], windows["lam_diff"], right=np.nan)
    ax.plot(lt, lam_loc - np.mean(np.gradient(h["R_mean"][sel], h["t"][sel])) / Rw, "k-", lw=1.1, label=r"diffuse planar at $k=\sqrt{(l-1)(l+2)}/R$")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("l")
    ax.set_ylabel(r"growth rate of $\sqrt{P_l}/R$")
    ax.set_title("linear stage: measured vs predicted")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    for ci, res in enumerate(results):
        hh = res["hist"]
        ax.plot(hh["t"], np.sqrt(hh["spec"][:, 2:].sum(axis=1)) / hh["R_mean"], color=f"C{ci}", label=res["label"])
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\sqrt{\sum_{l\geq2}P_l}/R$")
    ax.set_title("total shape-perturbation amplitude")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return rates_out


def plot_morphology(results: list[dict], morph: list[dict], out: Path) -> None:
    fig = plt.figure(figsize=(18, 4.6 * len(results)))
    for row, (res, mo) in enumerate(zip(results, morph)):
        p = res["p"]
        nc = 4
        ax = fig.add_subplot(len(results), nc, row * nc + 1, projection="3d")
        v, f = isosurface(res["u"], p.L, stride=2, sigma=1.0)
        r_max = float(np.max(np.linalg.norm(v, axis=1))) * 1.02
        draw_surface(ax, v, f, r_max, title=f"{res['label']}: cut-away, t={res['final_step'] * p.dt:.0f}")
        ax = fig.add_subplot(len(results), nc, row * nc + 2)
        H, K = mo["H"], mo["K"]
        ax.hist2d(np.clip(H, -1.5, 1.5), np.clip(K, -0.6, 0.6), bins=80, cmap="magma", density=True)
        ax.axhline(0, color="w", lw=0.5)
        ax.axvline(0, color="w", lw=0.5)
        ax.set_xlabel(r"$H/k^*$")
        ax.set_ylabel(r"$K/k^{*2}$")
        s = mo["summary"]
        ax.set_title(f"interface curvature: K>0 convex {s['frac_elliptic_convex_K>0']:.2f}, saddle {s['frac_saddle_K<0']:.2f}", fontsize=8)
        for k_, (med, name) in enumerate(((mo["med1"], "u=1 phase"), (mo["med0"], "u=0 phase"))):
            ax = fig.add_subplot(len(results), nc, row * nc + 3 + k_)
            if "linearity" in med:
                ax.scatter(med["linearity"], med["planarity"], s=3, alpha=0.3)
                ax.plot([0, 1], [0, 1], "k--", lw=0.8)
                ax.set_title(f"{name} medial axis: sheet {med['frac_sheet']:.2f} / tube {med['frac_tube']:.2f}", fontsize=8)
            ax.set_xlabel("linearity (tube-like)")
            ax.set_ylabel("planarity (sheet-like)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_benchmark(bench: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ns = list(bench)
    backends = list(next(iter(bench.values())))
    w = 0.8 / len(ns)
    for i, n in enumerate(ns):
        vals = [bench[n][b] for b in backends]
        x = np.arange(len(backends)) + i * w
        bars = ax.bar(x, vals, width=w, label=f"{n}³")
        base = bench[n]["numpy64"]
        for xb, vb in zip(x, vals):
            if np.isfinite(vb):
                ax.text(xb, vb, f"×{base / vb:.0f}" if np.isfinite(base) else "", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(backends)) + 0.4 - w / 2)
    ax.set_xticklabels(backends, rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel("ms / step (lower is better)")
    ax.set_title("semi-implicit 3D step: backend comparison (speed-up vs numpy64)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def save_rotating_gif(res: dict, out: Path, n_frames: int = 36) -> None:
    p = res["p"]
    v, f = isosurface(res["u"], p.L, stride=3, sigma=1.0)
    r_max = float(np.max(np.linalg.norm(v, axis=1))) * 1.02
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    draw_surface(ax, v, f, r_max, title=f"{res['label']} (cut-away), t={res['final_step'] * p.dt:.0f}")

    def update(i):
        ax.view_init(elev=24, azim=-58 + 360 * i / n_frames)
        return ()

    FuncAnimation(fig, update, frames=n_frames).save(str(out), writer=PillowWriter(fps=10), dpi=90)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--quick", action="store_true")
    ap_.add_argument("--bench", action="store_true")
    ap_.add_argument("--backend", default="torch_cuda")
    ap_.add_argument("--a-control", type=float, default=0.06)
    ap_.add_argument("--n", type=int, default=None)
    ap_.add_argument("--steps", type=int, default=None)
    args = ap_.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    base = Params3D()
    if args.quick:
        base = replace(base, n=128, n_steps=3000)
    if args.n:
        base = replace(base, n=args.n)
    if args.steps:
        base = replace(base, n_steps=args.steps)

    print(f"scheme equivalence (numpy kernel-FFT vs analytic symbol, 20 steps): max|Δu|={check_scheme_equivalence():.2e}")

    k_diff = np.linspace(0.05, 5.0, 100)
    lam_diff = diffuse_planar_dispersion(base.eps, base.d, k_diff)
    th = theory_summary(base, k_diff, lam_diff)
    th_ctrl = theory_summary(replace(base, a=args.a_control), k_diff, lam_diff)
    print("theory (a=0):", json.dumps(th, indent=1))
    print(f"theory (a={args.a_control}): phi_lamella={th_ctrl['phi_lamella']:.3f}, dR/dt(R0)={th_ctrl['sphere_dRdt_R0']:.4f}")
    plot_theory(base, k_diff, lam_diff, th, RESULTS / f"{PREFIX}_analysis.png")

    bench = None
    if args.bench:
        print("benchmark:")
        bench = benchmark()
        plot_benchmark(bench, RESULTS / f"{PREFIX}_benchmark.png")

    cases = [("a=0 (original)", base), (f"a={args.a_control} (control)", replace(base, a=args.a_control))]
    snap_steps = tuple(range(0, base.n_steps + 1, 500))
    results = []
    for label, p in cases:
        print(f"run {label}: n={p.n}³, L={p.L}, backend={args.backend}")
        results.append(run_case(p, label.split()[0], args.backend, snap_steps=snap_steps))

    windows = {"k_diff": k_diff, "lam_diff": lam_diff}
    for res in results:
        h = res["hist"]
        sel = np.nonzero((h["R_mean"] > 1.6) & (h["R_mean"] < 3.0))[0]
        t_lo = float(h["t"][sel[0]]) if len(sel) else float(h["t"][2])
        t_hi = float(h["t"][sel[-1]]) if len(sel) else float(h["t"][min(10, len(h["t"]) - 1)])
        windows[res["label"]] = (t_lo, t_hi)
    plot_validation(results, th, base, RESULTS / f"{PREFIX}_validation.png", windows)
    plot_evolution(results, RESULTS / f"{PREFIX}_evolution.png")

    lam_star = th["lambda_star_diffuse"]
    morph = [analyze_morphology(res, lam_star) for res in results]
    plot_morphology(results, morph, RESULTS / f"{PREFIX}_morphology.png")
    save_rotating_gif(results[0], RESULTS / f"{PREFIX}_a0_rotate.gif")
    save_rotating_gif(results[1], RESULTS / f"{PREFIX}_control_rotate.gif")

    summary = {
        "params": asdict(base),
        "theory_a0": th,
        "theory_control": th_ctrl,
        "benchmark_ms_per_step": bench,
        "runs": {
            res["label"]: {
                "final_step": res["final_step"],
                "wall_s": res["wall"],
                "R_mean_final": float(res["hist"]["R_mean"][-1]),
                **mo["summary"],
            }
            for res, mo in zip(results, morph)
        },
    }
    (RESULTS / f"{PREFIX}_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary["runs"], indent=1, default=float))


if __name__ == "__main__":
    main()
