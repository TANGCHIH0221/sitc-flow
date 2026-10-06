"""第一關 README 用圖。數字由 impact_panel.parquet 重算(與 03 / 04 相同定義)。
配色:台股慣例 紅 = 買、藍 = 賣(dataviz 參考調色盤的 diverging 兩端)。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUY, SELL, NEUTRAL = "#e34948", "#2a78d6", "#b9b8b3"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.family": ["Heiti TC", "Arial Unicode MS"], "font.size": 10,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": .8,
    "axes.titleweight": "bold", "axes.axisbelow": True, "axes.ymargin": .12, "axes.titlesize": 12, "axes.titlelocation": "left",
})
OUT = Path("figures"); OUT.mkdir(exist_ok=True)

u = pd.read_parquet("impact_panel.parquet")


def label(ax, bars, vals, fmt="{:+.0f}"):
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK2)


# 圖 1:每年 大買 / 大賣 日的盤中超額
yr = u[u.big_buy | u.big_sell].groupby(["year", "big_buy"]).oc_x.mean().unstack() * 1e4
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(yr)); w = .38
b1 = ax.bar(x - w / 2 - .01, yr[True], w, color=BUY, label="投信大買日")
b2 = ax.bar(x + w / 2 + .01, yr[False], w, color=SELL, label="投信大賣日")
label(ax, b1, yr[True]); label(ax, b2, yr[False])
ax.axhline(0, color=INK2, lw=.8)
ax.set_xticks(x, yr.index); ax.set_ylabel("開盤→收盤 超額報酬(bp)")
ax.set_title("投信大買 / 大賣日的盤中超額報酬,2019–2025 每年")
ax.legend(frameon=False, loc="upper right", ncol=2)
ax.text(0, -.16, "超額 = 個股報酬 − 同產業其他股票平均。大買 = 投信淨買 ≥ 當日成交量 10% 且 ≥ 20 日均量 5%;範圍 20 日均額 ≥ 1 億。",
        transform=ax.transAxes, fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(OUT / "fig1_by_year.png", dpi=160); plt.close(fig)

# 圖 2:依 投信淨買 / 當日成交量 分組(單調性)
edges = [-1, -.3, -.2, -.1, -.05, -1e-9, 1e-9, .05, .1, .2, .3, 1]
labs = ["≤−30%", "−30~−20", "−20~−10", "−10~−5", "−5~0", "0", "0~5", "5~10", "10~20", "20~30", "≥30%"]
u["nv_bin"] = pd.cut(u.net_vol, edges, labels=labs)
m = u.groupby("nv_bin", observed=True).oc_x.mean() * 1e4
fig, ax = plt.subplots(figsize=(8, 4))
cols = [SELL if i < 5 else NEUTRAL if i == 5 else BUY for i in range(len(m))]
bars = ax.bar(range(len(m)), m.values, .7, color=cols)
label(ax, bars, m.values)
ax.axhline(0, color=INK2, lw=.8)
ax.set_xticks(range(len(m)), m.index, fontsize=8.5)
ax.set_xlabel("投信淨買 / 當日成交量(%)"); ax.set_ylabel("開盤→收盤 超額報酬(bp)")
ax.set_title("投信淨買佔成交量越高,當天盤中越強(2019–2025)")
fig.tight_layout(); fig.savefig(OUT / "fig2_monotonic.png", dpi=160); plt.close(fig)

# 圖 3:依市值(與 04_impact_by_size.py 同定義,2020–2025)
so = pd.read_parquet("/Users/chih/quant/data/raw/finmind_daily/shares_outstanding_quarter.parquet")
so["snap_date"] = pd.to_datetime(so.snap_date.astype(str))
v = u[u.year >= "2020"].copy(); v["dt"] = pd.to_datetime(v.date)
v = pd.merge_asof(v.sort_values("dt"), so.sort_values("snap_date")[["snap_date", "stock_id", "shares_outstanding"]],
                  left_on="dt", right_on="snap_date", by="stock_id", direction="backward")
v["mcap"] = v.shares_outstanding * v.prev_close / 1e8
v = v.dropna(subset=["mcap", "oc_x"])
v["size"] = pd.cut(v.mcap, [0, 100, 500, 2000, 1e9], labels=["<100 億", "100–500 億", "500–2000 億", "≥2000 億"], right=False)
s = v[v.big_buy | v.big_sell].groupby(["size", "big_buy"], observed=True).oc_x.mean().unstack() * 1e4
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(s))
b1 = ax.bar(x - w / 2 - .01, s[True], w, color=BUY, label="投信大買日")
b2 = ax.bar(x + w / 2 + .01, s[False], w, color=SELL, label="投信大賣日")
label(ax, b1, s[True]); label(ax, b2, s[False])
ax.axhline(0, color=INK2, lw=.8)
ax.set_xticks(x, s.index); ax.set_xlabel("市值(前一日收盤 × 最近一季股數)")
ax.set_ylabel("開盤→收盤 超額報酬(bp)")
ax.set_title("市值越小,投信的影響越大(2020–2025)")
ax.legend(frameon=False, loc="upper right", ncol=2)
fig.tight_layout(); fig.savefig(OUT / "fig3_by_size.png", dpi=160); plt.close(fig)
print("figures done")

# 圖 4:排除金融後,市值細分(與 07_size_bins.py 同定義)
import runpy, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("07_size_bins.py")
w4 = ns["u"]
m4b = w4[w4.big_buy].groupby("mb", observed=True).oc_x.mean() * 1e4
m4s = w4[w4.big_sell].groupby("mb", observed=True).oc_x.mean() * 1e4
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(m4b))
b1 = ax.bar(x - w / 2 - .01, m4b.values, w, color=BUY, label="投信大買日")
b2 = ax.bar(x + w / 2 + .01, m4s.values, w, color=SELL, label="投信大賣日")
label(ax, b1, m4b.values); label(ax, b2, m4s.values)
ax.axhline(0, color=INK2, lw=.8)
ax.set_xticks(x, [l.replace("≥", "≥") + " 億" for l in m4b.index], fontsize=8.5)
ax.set_xlabel("市值"); ax.set_ylabel("開盤→收盤 超額報酬(bp)")
ax.set_title("排除金融股:大買 500 億以上持平,大賣隨市值單調變弱(2020–2025)")
ax.legend(frameon=False, loc="upper right", ncol=2)
fig.tight_layout(); fig.savefig(OUT / "fig4_size_bins.png", dpi=160); plt.close(fig)
print("fig4 done")
