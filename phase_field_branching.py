"""
分枝形態形成フェーズフィールドモデル
（Mathematica/1_continuous_scheme/1_1_diffusion_term/phase_field/phase_field_branching/1.1.8.BranchingPhaseField.nb から移植）。

notebook 最終セルと同じ半陰的スペクトル法:
  f(u,v) = u + dt*u*(1-u)*(u - 1/2 + (v - 0.5))
  g(u,v) = v + dt*(1 - u - v)
  u_{n+1} = IRFFT2( RFFT2(f) / RFFT2(kernelU) )
  v_{n+1} = IRFFT2( RFFT2(g) / RFFT2(kernelV) )
  （入力は実数のみのため RFFT2 / IRFFT2 に統一）

連続極限（README 記法）:
  ∂_t u = du ∂_xx u + u(1-u)(u+v-1),  du = ε² d
  ∂_t v = d ∂_xx v + (1-u-v)

シャープ界面極限（ε→0, du→0）の線形安定性:
  平面界面を y 方向に正弦摂動したときの固有値 λ(k)（成長率）は、上式を 1D 定常界面 (u_0(x),v_0(x))
  まわりで線形化し、δu,δv ∝ exp(λt+iky) とおくと
    λ δu = du(∂_xx−k²)δu + f_u δu + f_v δv,
    λ δv = d(∂_xx−k²)δv − δu − δv
  （f_u = ∂f/∂u, f_v = ∂f/∂v, f = u(1−u)(u+v−1)）。ε→0 で du→0 なら u の摂動方程式は
  界面外では主に代数的に拘束され、  界面層では曲率項と v 結合が競合し、有限 k で λ>0 となりうる。
  本コードは有限差分＋固有値で max Re λ(k) を数値的に求める。
"""
import numpy as np
import matplotlib.pyplot as plt
# --- atlas heatmap helpers ---
import sys
from pathlib import Path as _Path
for _d in _Path(__file__).resolve().parents:
    if (_d / "atlas_plotting.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
else:
    raise ImportError("atlas_plotting.py not found above " + str(__file__))
import atlas_plotting as ap

from pathlib import Path

try:
    from scipy import ndimage as _ndi  # type: ignore[import]
except Exception:  # noqa: BLE001
    _ndi = None
try:
    from scipy import optimize as _opt  # type: ignore[import]
except Exception:  # noqa: BLE001
    _opt = None


def interface_v_for_binary_stripe(
    width_u1: float,
    d: float,
    phi_u1: float,
    *,
    n_min: int = 400,
) -> float:
    """
    「u=1領域とu=0領域が交互に並ぶ」1D 周期ストライプモデルで、
    定常 v 方程式
        d v_xx + (1-u) - v = 0
    を解き、u=1→0 境界での v 値を返す。
    """
    phi = float(np.clip(phi_u1, 1e-3, 1.0 - 1e-3))
    period = width_u1 / phi
    if period <= 0.0:
        return np.nan
    n = max(n_min, int(np.ceil(period / 0.02)))
    dx = period / n
    x = np.arange(n, dtype=float) * dx
    u_bin = (x < width_u1).astype(float)
    source = 1.0 - u_bin

    # (d*L - I) v = -source, 周期境界
    main = (-2.0 * d / (dx * dx) - 1.0) * np.ones(n, dtype=float)
    off = (d / (dx * dx)) * np.ones(n - 1, dtype=float)
    a = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    a[0, -1] = d / (dx * dx)
    a[-1, 0] = d / (dx * dx)
    b = -source
    v = np.linalg.solve(a, b)

    # 境界 x=width_u1 での値（格子間線形補間）
    i = int(np.floor(width_u1 / dx)) % n
    i2 = (i + 1) % n
    x_i = i * dx
    alpha = (width_u1 - x_i) / dx
    v_if = (1.0 - alpha) * v[i] + alpha * v[i2]
    return float(v_if)


def interface_values_for_asymmetric_stripe(
    w1: float,
    w0: float,
    d: float,
    *,
    n_min: int = 600,
) -> tuple[float, float]:
    """
    交互ストライプ（u=1 幅 w1, u=0 幅 w0）の周期定常 v を解き、
    2つの界面（x=0 と x=w1）での v を返す。
    """
    if w1 <= 0.0 or w0 <= 0.0:
        return np.nan, np.nan
    period = w1 + w0
    n = max(n_min, int(np.ceil(period / 0.02)))
    dx = period / n
    x = np.arange(n, dtype=float) * dx
    u_bin = (x < w1).astype(float)
    source = 1.0 - u_bin

    main = (-2.0 * d / (dx * dx) - 1.0) * np.ones(n, dtype=float)
    off = (d / (dx * dx)) * np.ones(n - 1, dtype=float)
    a = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    a[0, -1] = d / (dx * dx)
    a[-1, 0] = d / (dx * dx)
    v = np.linalg.solve(a, -source)

    # interface at x=0 (period boundary)
    v_if0 = float(v[0])
    # interface at x=w1
    i = int(np.floor(w1 / dx)) % n
    i2 = (i + 1) % n
    x_i = i * dx
    alpha = (w1 - x_i) / dx
    v_if1 = float((1.0 - alpha) * v[i] + alpha * v[i2])
    return v_if0, v_if1


def predicted_width_from_stationary_condition_asymmetric(
    d: float,
    *,
    target_v_interface: float = 0.5,
) -> tuple[float, float, bool]:
    """
    非対称交互ストライプ (w1, w0) に対し、
      v_if(x=0)=v0, v_if(x=w1)=v0
    の連立条件を数値的に解く。
    戻り値: (w1_pred, w0_pred, success)
    """
    if _opt is None:
        return float("nan"), float("nan"), False

    def eq(z: np.ndarray) -> np.ndarray:
        w1, w0 = float(z[0]), float(z[1])
        if w1 <= 0.0 or w0 <= 0.0:
            return np.array([1e3, 1e3], dtype=float)
        v0, v1 = interface_values_for_asymmetric_stripe(w1, w0, d)
        return np.array([v0 - target_v_interface, v1 - target_v_interface], dtype=float)

    # 複数初期値で解探索（見つからない場合を減らす）
    guesses = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (3.0, 1.5), (1.5, 3.0)]
    best = None
    for g in guesses:
        sol = _opt.root(eq, np.array(g, dtype=float), method="hybr")
        if sol.success and sol.x[0] > 0 and sol.x[1] > 0:
            res = np.linalg.norm(eq(sol.x))
            if best is None or res < best[0]:
                best = (res, float(sol.x[0]), float(sol.x[1]))

    if best is None:
        return float("nan"), float("nan"), False
    _res, w1p, w0p = best
    return w1p, w0p, True


def interface_width_distribution(
    u: np.ndarray,
    dx: float,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    u を閾値で二値化し、Distance map + skeleton から構造太さの分布を推定。

    手順:
      1. u>threshold を 1, それ以外 0 とするマスクを作成。
      2. 距離変換 dist: 各 1 の点から最近の 0 までの距離（格子単位）。
      3. skeleton: dist の 3×3 近傍で局所最大となる点（かつ dist>0）。
      4. 各 skeleton 点の「太さ」 w ≈ 2 * dist * dx（長さ単位）を集計し、その分布を返す。
         （最頻値が代表界面幅の良い推定値になる。）
    """
    if _ndi is None:
        # scipy が無い場合は空配列と NaN を返して、呼び出し側で処理。
        empty = np.zeros_like(u, dtype=float)
        return np.array([], dtype=float), float("nan"), empty.astype(bool), empty, empty.astype(bool)

    mask = u > threshold
    if not np.any(mask):
        empty = np.zeros_like(u, dtype=float)
        return np.array([], dtype=float), float("nan"), mask, empty, empty.astype(bool)

    dist = _ndi.distance_transform_edt(mask.astype(float))
    # skeleton: 距離関数の局所最大
    local_max = (dist == _ndi.maximum_filter(dist, size=3)) & mask & (dist > 0.0)
    radii = dist[local_max]
    if radii.size == 0:
        return np.array([], dtype=float), float("nan"), mask, dist, local_max

    widths = 2.0 * radii * dx  # physical length units

    # 0 以外の最頻値を代表幅とする
    positive = widths[widths > 0]
    if positive.size == 0:
        return widths, float("nan"), mask, dist, local_max
    hist, bin_edges = np.histogram(positive, bins="auto")
    idx = int(np.argmax(hist))
    width_mode = float(0.5 * (bin_edges[idx] + bin_edges[idx + 1]))
    return widths, width_mode, mask, dist, local_max


def tanh_interface_width_from_profile(
    u: np.ndarray,
    dx: float,
    *,
    level: float = 0.5,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    最終場の中心断面 u(x) に tanh をフィットして界面幅を推定する。
      u(x) ≈ 0.5*(1 - tanh((x-x0)/ell))
    幅は 0.1〜0.9 幅として
      w = 2*atanh(0.8)*|ell|
    """
    if _opt is None:
        return float("nan"), np.array([]), np.array([]), np.array([])

    ny, nx = u.shape
    cy = ny // 2
    prof = np.asarray(u[cy, :], dtype=float)
    x = np.arange(nx, dtype=float) * dx

    s = prof - level
    cross = np.where(s[:-1] * s[1:] <= 0)[0]
    if cross.size == 0:
        return float("nan"), x, prof, np.array([])

    # 中央に最も近い界面を使う
    cx = nx // 2
    i0 = int(cross[np.argmin(np.abs(cross - cx))])
    x0_guess = x[i0]
    win = max(8, int(round(3.0 / dx)))
    lo = max(0, i0 - win)
    hi = min(nx, i0 + win + 1)
    xf = x[lo:hi]
    yf = prof[lo:hi]
    if xf.size < 8:
        return float("nan"), x, prof, np.array([])

    def model(xx: np.ndarray, x0: float, ell: float) -> np.ndarray:
        return 0.5 * (1.0 - np.tanh((xx - x0) / ell))

    try:
        popt, _pcov = _opt.curve_fit(
            model,
            xf,
            yf,
            p0=(x0_guess, 0.3),
            bounds=([x[lo], 1e-4], [x[hi - 1], 10.0]),
            maxfev=20000,
        )
        x0_fit, ell_fit = float(popt[0]), float(popt[1])
    except Exception:  # noqa: BLE001
        return float("nan"), x, prof, np.array([])

    width = float(2.0 * np.arctanh(0.8) * abs(ell_fit))
    y_fit = model(x, x0_fit, ell_fit)
    return width, x, prof, y_fit


def _standard_params_from_branching(eps: float, d: float) -> tuple[float, float, float]:
    """Map branching (eps, d) to standard-model symbols (c0, epsilon, tau)."""
    epsilon_std = float(eps)
    tau = 1.0 / (epsilon_std**2)
    c0 = float(np.sqrt(d))
    return c0, epsilon_std, tau


def _effective_force_from_v(v: np.ndarray, c0: float, epsilon_std: float) -> np.ndarray:
    """F(v) defined so that beta*F(v) = v - 1/2."""
    return (np.sqrt(2.0) * c0 / epsilon_std) * (v - 0.5)


def _reaction_f(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Original branching u-reaction: f(u,v)=u(1-u)(u+v-1)."""
    return u * (1.0 - u) * (u + v - 1.0)


def _reaction_derivatives(
    u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """∂f/∂u, ∂f/∂v, ∂g/∂u, ∂g/∂v。g = 1-u-v。"""
    fu = (1.0 - 2.0 * u) * (u + v - 1.0) + u * (1.0 - u)
    fv = u * (1.0 - u)
    gu = -np.ones_like(u)
    gv = -np.ones_like(u)
    return fu, fv, gu, gv


def _laplacian_dirichlet_interior(n: int, dx: float) -> np.ndarray:
    """端点を 0 としたディリクレ下での 1D ラプラシアン (n-2)×(n-2)。"""
    m = n - 2
    h2 = dx * dx
    main = (-2.0 / h2) * np.ones(m, dtype=float)
    off = (1.0 / h2) * np.ones(m - 1, dtype=float)
    lap = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    return lap


def _newton_jacobian_steady_1d(
    u_full: np.ndarray,
    v_full: np.ndarray,
    du: float,
    d: float,
    dx: float,
) -> np.ndarray:
    """内点残差 [r_u; r_v] について (u_int, v_int) のヤコビアン（解析式）。"""
    nx = u_full.size
    m = nx - 2
    lap = _laplacian_dirichlet_interior(nx, dx)
    fu, fv, gu, gv = _reaction_derivatives(u_full, v_full)
    j_uu = du * lap + np.diag(fu[1:-1])
    j_uv = np.diag(fv[1:-1])
    j_vu = np.diag(gu[1:-1])
    j_vv = d * lap + np.diag(gv[1:-1])
    return np.block([[j_uu, j_uv], [j_vu, j_vv]])


def steady_planar_front_1d(
    eps: float,
    d: float,
    *,
    x_max: float = 12.0,
    nx: int = 241,
    tol: float = 1e-10,
    max_newton: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    平面界面の 1D 定常解 (u(x), v(x)) を Newton 法で求める。

    境界: u(-L)=0, u(+L)=1, v(-L)=1, v(+L)=0（外側バルクに合わせたディリクレ）。
    戻り値: (x, u, v, converged)
    """
    x = np.linspace(-x_max, x_max, nx, dtype=float)
    dx = float(x[1] - x[0])
    du = (eps ** 2) * d

    u = 0.5 * (1.0 + np.tanh(x / max(0.15 * eps, 1e-6)))
    v = 0.5 * (1.0 - np.tanh(x / 2.0))

    def pack(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        return np.concatenate([uu[1:-1], vv[1:-1]])

    def unpack(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        uu = np.zeros(nx, dtype=float)
        vv = np.zeros(nx, dtype=float)
        uu[1:-1] = uv[: nx - 2]
        vv[1:-1] = uv[nx - 2 :]
        uu[0], uu[-1] = 0.0, 1.0
        vv[0], vv[-1] = 1.0, 0.0
        return uu, vv

    def residual(uv: np.ndarray) -> np.ndarray:
        uu, vv = unpack(uv)
        lap_u = (uu[2:] - 2.0 * uu[1:-1] + uu[:-2]) / (dx * dx)
        lap_v = (vv[2:] - 2.0 * vv[1:-1] + vv[:-2]) / (dx * dx)
        ru = du * lap_u + _reaction_f(uu, vv)[1:-1]
        rv = d * lap_v + (1.0 - uu - vv)[1:-1]
        return np.concatenate([ru, rv])

    uv0 = pack(u, v)
    r0 = residual(uv0)
    for _ in range(max_newton):
        if float(np.linalg.norm(r0, ord=np.inf)) < tol:
            break
        uu, vv = unpack(uv0)
        jac = _newton_jacobian_steady_1d(uu, vv, du, d, dx)
        try:
            delta = np.linalg.solve(jac, -r0)
        except np.linalg.LinAlgError:
            return x, u, v, False
        alpha = 1.0
        for _ls in range(14):
            uv_try = uv0 + alpha * delta
            r_try = residual(uv_try)
            if float(np.linalg.norm(r_try, ord=np.inf)) < float(
                np.linalg.norm(r0, ord=np.inf)
            ):
                uv0, r0 = uv_try, r_try
                break
            alpha *= 0.5
        else:
            return x, u, v, False

    u_out, v_out = unpack(uv0)
    ok = float(np.linalg.norm(r0, ord=np.inf)) < tol * 1e3
    return x, u_out, v_out, ok


def linearized_front_max_Re_lambda(
    x: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    eps: float,
    d: float,
    k_values: np.ndarray,
) -> np.ndarray:
    """
    定常 1D 界面 (u0,v0) に対し、横波数 k の摂動 exp(λt+iky) の固有値の最大実部 max Re λ。

    連続時間線形化: λ δu = du(L-k²I)δu + f_u δu + f_v δv 等（内点のみ、端で δ=0）。
    """
    nx = x.size
    dx = float(x[1] - x[0])
    du = (eps ** 2) * d
    m = nx - 2
    lap = _laplacian_dirichlet_interior(nx, dx)
    fu, fv, gu, gv = _reaction_derivatives(u0, v0)
    fu_i, fv_i = fu[1:-1], fv[1:-1]
    gu_i, gv_i = gu[1:-1], gv[1:-1]
    i_m = np.eye(m, dtype=float)
    out = np.empty_like(k_values, dtype=float)
    for ik, k in enumerate(k_values):
        k2 = float(k * k)
        a_uu = du * (lap - k2 * i_m) + np.diag(fu_i)
        a_uv = np.diag(fv_i)
        a_vu = np.diag(gu_i)
        a_vv = d * (lap - k2 * i_m) + np.diag(gv_i)
        big = np.block([[a_uu, a_uv], [a_vu, a_vv]])
        lam_vals = np.linalg.eigvals(big)
        out[ik] = float(np.max(lam_vals.real))
    return out


def plot_sharp_interface_linear_stability(
    eps_list: tuple[float, ...],
    d: float,
    out_path: Path,
    *,
    k_max: float = 4.5,
    nk: int = 51,
) -> bool:
    """
    各 ε について max Re λ(k) をプロット。いずれかの k で max Re λ>0 なら平面界面は線形不安定。

    シャープ界面極限: 本モデルでは du=ε²d を小さく（ε↓）しても v 結合による横モード不安定は
    弱まらず、数値上は max_k max Re λ が増大しやすい（界面曲率項の安定化が ε² で弱まるため）。
    """
    k_vals = np.linspace(0.0, k_max, nk, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    for eps in eps_list:
        x, u0, v0, ok = steady_planar_front_1d(eps, d)
        if not ok or x.size == 0:
            continue
        re_lam_max = linearized_front_max_Re_lambda(x, u0, v0, eps, d, k_vals)
        ax.plot(
            k_vals,
            re_lam_max,
            lw=1.4,
            label=rf"$\varepsilon$={eps:g}, max $\mathrm{{Re}}\,\lambda$={np.nanmax(re_lam_max):.4g}",
        )

    ax.axhline(0.0, color="0.35", ls="--", lw=0.9)
    ax.set_xlabel(r"transverse wavenumber $k$")
    ax.set_ylabel(r"max $\mathrm{Re}\,\lambda(k)$")
    ax.set_title(
        "Linear stability of planar interface\n"
        r"(1D steady front + $e^{\lambda t+iky}$ perturbation; $du=\varepsilon^2 d$, $d_v=d$)"
    )
    ax.grid(alpha=0.35)
    ax.legend(fontsize=8, loc="best")
    ax.text(
        0.02,
        0.02,
        r"$\mathrm{Re}\,\lambda>0$ $\Rightarrow$ planar interface linearly unstable"
        "\n(branching morphology is consistent with this spectrum)",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def simulate_phase_field_branching(
    domain_size: float = 20.0,
    dx: float = 0.1,
    eps: float = 0.2,
    d: float = 0.1,
    dt: float = 0.5,
    n_steps: int = 6000,
    seed: int = 42,
) -> tuple:
    """分枝形態形成フェーズフィールドのシミュレーション（.nb 準拠）。"""
    rng = np.random.default_rng(seed)
    grid_size = int(round(domain_size / dx))
    c0, epsilon_std, tau = _standard_params_from_branching(eps=eps, d=d)
    du = c0**2 / tau
    dv = d
    alpha = 1.0 / (epsilon_std**2 * tau)
    beta = epsilon_std / (np.sqrt(2.0) * c0)

    # 初期条件 (.nb): 中心円 (半径 1/dx) に点ごとの半径揺らぎ U(-1,1)
    cx, cy = grid_size // 2, grid_size // 2
    x_idx = np.arange(1, grid_size + 1)[:, None]
    y_idx = np.arange(1, grid_size + 1)[None, :]
    r2 = (x_idx - grid_size / 2.0) ** 2 + (y_idx - grid_size / 2.0) ** 2
    noisy_radius = (1.0 / dx) + rng.uniform(-1.0, 1.0, size=(grid_size, grid_size))
    u = (r2 < noisy_radius ** 2).astype(float)

    v = np.zeros((grid_size, grid_size))

    # 拡散の陰解法カーネル (.nb と同じ5点 stencil)
    cu = dt * du / (dx * dx)
    kernel_u = np.zeros((grid_size, grid_size))
    kernel_u[0, 0] = 1.0 + 4.0 * cu
    kernel_u[[1, -1, 0, 0], [0, 0, 1, -1]] = -cu
    # .nb: ku = 1 / Fourier[kernelU] / gridSize（実カーネル → rfft2）
    ku = 1.0 / np.fft.rfft2(kernel_u, norm="ortho") / grid_size

    cv = dt * dv / (dx * dx)
    kernel_v = np.zeros((grid_size, grid_size))
    kernel_v[0, 0] = 1.0 + 4.0 * cv
    kernel_v[[1, -1, 0, 0], [0, 0, 1, -1]] = -cv
    # .nb: kv = 1 / Fourier[kernelV] / gridSize
    kv = 1.0 / np.fft.rfft2(kernel_v, norm="ortho") / grid_size

    snapshots = [(u.copy(), v.copy())]
    save_every = max(1, n_steps // 4)

    for step in range(n_steps):
        # Standard-model form with F(v), exactly equivalent to original branching reaction.
        Fv = _effective_force_from_v(v, c0=c0, epsilon_std=epsilon_std)
        f_uv = u + dt * (alpha * u * (1.0 - u) * (u - 0.5 + beta * Fv))
        g_uv = v + dt * (1.0 - u - v)
        # .nb: InverseFourier[ku * Fourier[f[u,v]]]
        u = np.fft.irfft2(
            np.fft.rfft2(f_uv, norm="ortho") * ku,
            s=(grid_size, grid_size),
            norm="ortho",
        )
        v = np.fft.irfft2(
            np.fft.rfft2(g_uv, norm="ortho") * kv,
            s=(grid_size, grid_size),
            norm="ortho",
        )

        if (step + 1) % save_every == 0:
            snapshots.append((u.copy(), v.copy()))

    return snapshots, dx


def main():
    # 計算時間を従来比 3 倍にして準定常へ近づける
    n_steps = 6000
    snapshots, dx = simulate_phase_field_branching(n_steps=n_steps)

    fig, axes = plt.subplots(2, len(snapshots), figsize=(4 * len(snapshots), 8))
    save_every = max(1, n_steps // 4)
    for i, (u, v) in enumerate(snapshots):
        im1 = ap.atlas_imshow(
            axes[0, i],
            u.T,
            heatmap="scalar",
            origin="lower",
            interpolation="nearest",
        )
        axes[0, i].set_title(f"u (step={i * save_every})", fontsize=9)
        axes[0, i].axis("off")

        im2 = ap.atlas_imshow(
            axes[1, i],
            v.T,
            heatmap="scalar",
            origin="lower",
            interpolation="nearest",
        )
        axes[1, i].set_title(f"v (step={i * save_every})", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle("Phase-field branching morphogenesis (\u03b5=0.2, d=0.1)")
    plt.tight_layout()
    out_path = Path(__file__).with_name("results/phase_field_branching.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

    # --- 界面幅の解析的予測と数値測定（最終フレームのみ） ---
    d = 0.1
    u_last, _v_last = snapshots[-1]

    widths, width_mode, mask, dist, skeleton = interface_width_distribution(
        u_last, dx, threshold=0.5
    )
    width_tanh, _x_prof, _y_prof, _y_fit = tanh_interface_width_from_profile(u_last, dx)
    w1_pred, w0_pred, ok_pred = predicted_width_from_stationary_condition_asymmetric(d=d)
    width_th = w1_pred if ok_pred else float("nan")

    fig2, axes2 = plt.subplots(2, 2, figsize=(10.0, 7.6))
    ax_bin, ax_dist, ax_skel, ax = axes2.ravel()

    ax_bin.imshow(mask.T, origin="lower", cmap="gray", interpolation="nearest")
    ax_bin.set_title("Binary mask (u > 0.5)")
    ax_bin.axis("off")

    imd = ap.atlas_imshow(
        ax_dist, dist.T, heatmap="scalar", origin="lower", interpolation="nearest"
    )
    ax_dist.set_title("Distance map")
    ax_dist.axis("off")
    fig2.colorbar(imd, ax=ax_dist, fraction=0.046, pad=0.02)

    skel_img = np.zeros((*skeleton.shape, 3), dtype=float)
    skel_img[..., 0] = mask.astype(float)
    skel_img[..., 1] = mask.astype(float)
    skel_img[..., 2] = mask.astype(float)
    skel_img[skeleton, :] = np.array([1.0, 0.1, 0.1])
    ax_skel.imshow(skel_img.transpose(1, 0, 2), origin="lower", interpolation="nearest")
    ax_skel.set_title("Skeleton on mask")
    ax_skel.axis("off")

    if widths.size > 0:
        ax.hist(widths, bins="auto", color="C0", alpha=0.75, label="thickness samples (2*dist*dx)")
    if np.isfinite(width_th):
        ax.axvline(
            width_th,
            color="C2",
            ls="--",
            lw=1.1,
            label=f"derived w* ~ {width_th:.3f}",
        )
    if np.isfinite(width_mode):
        ax.axvline(width_mode, color="C3", ls="-.", lw=1.1, label=f"mode ~ {width_mode:.3f}")
    if np.isfinite(width_tanh):
        ax.axvline(width_tanh, color="C1", ls=":", lw=1.2, label=f"tanh fit ~ {width_tanh:.3f}")
    ax.set_xlabel("local interface thickness (length unit)")
    ax.set_ylabel("count")
    ax.set_title("Interface thickness distribution from distance map + skeleton")
    ax.grid(alpha=0.3)
    txt = (
        f"asym stripe solve: w1≈{w1_pred:.3f}, w0≈{w0_pred:.3f}\n"
        f"derived w1 ≈ {width_th:.3f}\n"
        f"mode (numeric) ≈ {width_mode:.3f}\n"
        f"tanh-fit width ≈ {width_tanh:.3f}"
        if np.isfinite(width_mode) and np.isfinite(width_th)
        else f"asym stripe solve: w1={w1_pred}, w0={w0_pred}\n"
        f"derived w1: {width_th}\n"
        "mode (numeric): N/A"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
    )
    ax.legend(fontsize=8, loc="best")
    fig2.tight_layout()
    out_w = Path(__file__).with_name("results/phase_field_branching_interface_width.png")
    fig2.savefig(out_w, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out_w}")

    out_st = Path(__file__).with_name("results/phase_field_branching_linear_stability.png")
    # Use the fixed model epsilon only (no epsilon sweep).
    plot_sharp_interface_linear_stability((0.2,), d, out_st)
    print(f"Saved {out_st}")


if __name__ == "__main__":
    main()
