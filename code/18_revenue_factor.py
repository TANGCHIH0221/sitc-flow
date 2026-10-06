"""月營收因子(MOPS 彙總 CSV)→ T+1 投信大買機率,以及放進 16 的核心組合後的 T+1 報酬。

營收可用日:資料月份的下個月 11 日(法定期限 10 日,保守,不偷看)。
特徵:yoy = 去年同月增減;rec12 = 當月營收為近 12 個月最高;fresh = 可用後 5 個日曆天內。
"""
import glob
import re
import numpy as np
import pandas as pd

rows = []
for f in sorted(glob.glob("/Users/chih/quant/data/raw/mops_revenue/*.csv")):
    d = pd.read_csv(f, encoding="utf-8-sig", dtype={"公司代號": str})
    rows.append(d[["資料年月", "公司代號", "營業收入-當月營收", "營業收入-去年同月增減(%)"]])
rv = pd.concat(rows)
rv.columns = ["ym", "stock_id", "rev", "yoy_pct"]
ym = rv.ym.astype(str).str.extract(r"(\d+)/(\d+)").astype(int)
rv["y"], rv["m"] = ym[0] + 1911, ym[1]
rv = rv.drop_duplicates(["stock_id", "y", "m"]).sort_values(["stock_id", "y", "m"])
rv["yoy"] = pd.to_numeric(rv.yoy_pct, errors="coerce") / 100
rv["rec12"] = rv.rev >= rv.groupby("stock_id").rev.transform(lambda s: s.rolling(12, min_periods=12).max())
nm = (rv.m % 12) + 1
ny = rv.y + (rv.m == 12)
rv["avail"] = pd.to_datetime(dict(year=ny, month=nm, day=11))
print("營收:%d 筆、%d 檔、%d-%02d ~ %d-%02d;rec12 比例 %.1f%%" % (
    len(rv), rv.stock_id.nunique(), rv.y.min(), rv[rv.y == rv.y.min()].m.min(), rv.y.max(), rv[rv.y == rv.y.max()].m.max(),
    100 * rv.rec12.mean()))

with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
    ns = __import__("runpy").run_path("16_combo.py")
u, ev = ns["u"].drop(columns=["yoy", "rec12", "avail", "rev_fresh"]), ns["ev"]
u["dt"] = pd.to_datetime(u.date)
u = pd.merge_asof(u.sort_values("dt"), rv[["avail", "stock_id", "yoy", "rec12"]].dropna(subset=["yoy"]).sort_values("avail"),
                  left_on="dt", right_on="avail", by="stock_id", direction="backward")
u["fresh"] = (u.dt - u.avail).dt.days <= 5
u["yoy_b"] = pd.cut(u.yoy, [-100, -.2, 0, .2, .5, 1, 1e9], labels=["≤-20%", "-20~0", "0~20", "20~50", "50~100", ">100%"])

for name, pop in [("全部", u), ("沉寂", u[u.quiet])]:
    base = pop.groupby("per").y.mean()
    print(f"\n#### {name}:基準 dev {100*base['dev']:.2f}% / val {100*base['val']:.2f}%")
    for lab, col in [("月營收年增", "yoy_b"), ("營收 12 個月新高", "rec12"), ("營收剛可用 5 天內", "fresh")]:
        t = pop.groupby([col, "per"], observed=True).y.agg(["size", "mean"]).unstack("per")
        out = pd.DataFrame({"n_dev": t[("size", "dev")], "dev%": 100 * t[("mean", "dev")], "倍dev": t[("mean", "dev")] / base["dev"],
                            "n_val": t[("size", "val")], "val%": 100 * t[("mean", "val")], "倍val": t[("mean", "val")] / base["val"]})
        print(f"\n-- {lab} --"); print(out.round(2).to_string())

core = (u.it_days20 > 5) & (u.ret20.between(0, .20)) & (u.ret1 < .095) & (u.dist_hi60 >= -.10)
print("\n== 核心組合(投信近20日買>5天、20日漲0~20%、沒漲停、距60日高≤10%)× 營收 ==")
res = {}
for lab, m in [("全部", core), ("+ 年增 ≥ 20%", core & (u.yoy >= .2)), ("+ 12 個月新高", core & u.rec12.fillna(False).astype(bool)),
               ("+ 剛公布且年增 ≥ 20%", core & u.fresh & (u.yoy >= .2))]:
    for p, x in u[m.fillna(False)].groupby("per"):
        res[(lab, p)] = ev(x)
print(pd.DataFrame(res).T.round(1).to_string())
