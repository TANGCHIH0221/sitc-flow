"""範圍定案:排除金融、不設市值上限。市值切細 bin 看大買/大賣日影響是否隨市值單調。
bin 用整數(億):50/100/200/500/1000/2000/5000。2020–2025。"""
import runpy, io, contextlib
import numpy as np
import pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("06_vol_normalized.py")
u = ns["u"]
u = u[u.grp != "金融"].copy()
EDGES = [0, 50, 100, 200, 500, 1000, 2000, 5000, 1e9]
LABS = ["<50", "50-100", "100-200", "200-500", "500-1000", "1000-2000", "2000-5000", "≥5000"]
u["mb"] = pd.cut(u.mcap, EDGES, labels=LABS, right=False)

def ev(x):
    return pd.Series({"n": len(x), "股票數": x.stock_id.nunique(), "oc_bp": 1e4 * x.oc_x.mean(),
                      "oc_med": 1e4 * x.oc_x.median(), "win%": 100 * (x.oc_x > 0).mean(),
                      "σ_bp": 1e4 * x.sig20.median(), "z": x.z.mean(),
                      "正的年數": (x.groupby("year").oc_x.mean() > 0).sum() if x.oc_x.mean() > 0 else (x.groupby("year").oc_x.mean() < 0).sum()})
for lab, m in [("大買", u.big_buy), ("大賣", u.big_sell)]:
    print(f"\n== {lab}(排除金融,2020–2025) ==")
    print(u[m].groupby("mb", observed=True).apply(ev, include_groups=False).round(2).to_string())
print("\n== 大買 依產業分開:oc_bp / z ==")
t = u[u.big_buy].groupby(["mb", "grp"], observed=True).agg(n=("z", "size"), oc_bp=("oc_x", lambda x: 1e4 * x.mean()), z=("z", "mean")).round(2)
print(t.unstack("grp").to_string())
# 單調性:Spearman(bin 序, 平均)
from scipy.stats import spearmanr
for lab, m in [("大買", u.big_buy), ("大賣", u.big_sell)]:
    g = u[m].groupby("mb", observed=True)
    print("%s Spearman(市值序, bp) = %.2f, (市值序, z) = %.2f" % (
        lab, spearmanr(range(g.ngroups), g.oc_x.mean())[0], spearmanr(range(g.ngroups), g.z.mean())[0]))
