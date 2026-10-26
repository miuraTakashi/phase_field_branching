# phase_field_branching — 分枝形態形成フェーズフィールド（2 場、半陰的拡散）

対応スクリプト: [`phase_field_branching.py`](phase_field_branching.py)。3D 版（球＋摂動、皺／枝の予測と検証、GPU 高速化）は [`phase_field_branching_3d.md`](phase_field_branching_3d.md)。

## モデルの導出（現象論）

枝分かれ成長は **界面の曲率** と **物質供給** の競合で説明される。

- **仮説**: 秩序変数 $\phi$（フェーズフィールド）が界面エネルギー $F[\phi]$ を最小化しつつ拡散する Allen–Cahn / Cahn–Hilliard 型。
- **構成**: 化学ポテンシャル $\mu = \delta F/\delta\phi$、フラックス $J=-M
abla\mu$（物質保存）。
- 曲率依存成長則をフェーズフィールド勾配エネルギーに織り込み、次節の PDE へ。

## 支配方程式（半離散化の 1 ステップ）

連続的には 2 場 $u,v$ の反応–拡散型。コードは各ステップで

$$
f = u + \Delta t\, u(1-u)\bigl(u-\tfrac12 + (v-\tfrac12)\bigr),
$$

$$
g = v + \Delta t\,(1-u-v),
$$

$$
u^{n+1} = \mathcal{F}^{-1}\bigl[k_u \,\mathcal{F}(f)\bigr],\quad
v^{n+1} = \mathcal{F}^{-1}\bigl[k_v \,\mathcal{F}(g)\bigr].
$$

ここで $\mathcal{F}$ は 2D 実 FFT（`rfft2`）、$k_u,k_v$ は拡散を陰的に処理する周波数空間の乗数。拡散係数は $d_u=(\varepsilon^2)d$, $d_v=d$。

## 出典（原著・標準文献）

分枝・デンドライト成長をフェーズフィールド＋拡散場（秩序変数と補助場）で扱う系として、**Kobayashi の Physica D 論文**が最も広く参照される古典である。

- **Kobayashi, R.** “Modeling and numerical simulations of dendritic crystal growth.” *Physica D* **63**, 410–423 (1993). <https://doi.org/10.1016/0167-2789(93)90120-P>  
  キネティック項・異方性を含むフェーズフィールド方程式と熱（拡散）場の結合によるデンドライト分枝の数値デモの出発点。
- **Karma, A. & Rappel, W.-J.** “Quantitative phase-field modeling of dendritic growth in two and three dimensions.” *Physical Review E* **57**, 4323–4349 (1998). <https://doi.org/10.1103/PhysRevE.57.4323>  
  Kobayashi 系を踏まえた**薄界面極限との整合**を重視した定量フェーズフィールドモデリング（分枝・先端速度などの精密化）。
- **Murray, J. D.** *Mathematical Biology II: Spatial Models and Biomedical Applications*（3rd ed., Springer, 2003）. <https://doi.org/10.1007/b98869> — 反応拡散・パターン形成の応用数学の文脈。

本コードの具体的な $f,g$（および Mathematica ノート由来の離散式）は **アトラス用デモ**であり、上記の凝固フェーズフィールドと式を 1 対 1 で同一視するものではないが、**秩序場 $u$ と遅い拡散場 $v$ の 2 場構造＋界面近傍の不安定化**という読み方では Kobayashi 系の系譜に位置づけられる。

## 数値計算スキーム

- **空間**: 周期格子、5 点ラプラシアンの FFT 陰式核。
- **時間**: 反応陽式＋拡散陰式（スペクトル）。

## パラメータ一覧

| 識別子 | 既定値 | 意味 |
|--------|--------|------|
| `domain_size` | 20.0 | 領域一辺の長さ |
| `dx` | 0.1 | 空間刻み |
| `eps` | 0.2 | $\varepsilon$ |
| `d` | 0.1 | 拡散スケール $d$ |
| `dt` | 0.5 | 時間刻み |
| `n_steps` | 2000 | ステップ数 |
| `seed` | 42 | 初期ノイズ用 RNG |

## 数理解析

- 2D パターン自体の厳密解は扱わないが、**平面界面のシャープ界面極限まわりの線形安定性**は
  [`phase_field_branching.py`](phase_field_branching.py) 内で数値的に実施している。
  1D 定常界面を Newton 法で求め、摂動 $\propto e^{\lambda t + i k y}$ の固有値 $\lambda$ から
  $\max_k \mathrm{Re}\,\lambda(k)$ を評価。既定パラメータでは **有限の横波数 $k$ で
  $\mathrm{Re}\,\lambda \gt 0$** となり、平面界面は線形不安定（分枝が起こりうることと整合）。
  `main()` 実行時に `results/phase_field_branching_linear_stability.png` を出力。

## PhaseFieldStandardModel 形式との対応（u 方程式）

`u` の更新は、以下の対応で `PhaseFieldStandardModel` 形式
$$
\frac{\partial u}{\partial t}
= \frac{c_0^2}{\tau}\Delta u
+ \frac{1}{\epsilon^2\tau}u(1-u)\left(u-\frac12+\frac{\epsilon}{\sqrt2\,c_0}F\right)
$$
に読み替え可能（数値的に同値）。

### 対応式

$$
\epsilon=\mathrm{eps},\quad
\tau=\frac{1}{\mathrm{eps}^2},\quad
c_0=\sqrt{d},\quad
F(v)=\frac{\sqrt{2}\,c_0}{\epsilon}\left(v-\frac12\right)
$$

このとき
$$
\frac{c_0^2}{\tau}=\mathrm{eps}^2 d,\quad
\frac{1}{\epsilon^2\tau}=1,\quad
\frac{\epsilon}{\sqrt2 c_0}F(v)=v-\frac12
$$
となるため、元の反応項
$$
u(1-u)\left(u-\frac12+(v-\frac12)\right)
$$
と厳密に一致する。

注意: これは **u 方程式の同値変形**であり、モデル全体は `v` の進化式
$$
\frac{\partial v}{\partial t}=d\,\Delta v+(1-u-v)
$$
を含む 2 場系である。


## 2D pattern stock
- 異なるシードでstep 3000の画像を100%