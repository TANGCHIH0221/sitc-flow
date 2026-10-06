"""第三關-2:單因子逐步疊加(整數門檻、邏輯先行)→ T+1 投信大買命中率 + T+1 報酬。

注意:條件方向是看過 dev 與 val 的單因子結果後才挑的 → val 不算乾淨;2026 保留當最後樣本外。
T+1 報酬:跳空 gap_x、盤中 oc_x(同產業超額,impact_panel 定義);另列原始 oc。
"""
import numpy as np
import pandas as pd

u = pd.read_parquet("t1_features.parquet")
px = pd.read_parquet("panel.parquet", columns=["date", "stock_id"]).sort_values(["stock_id", "date"])
px["date_next"] = px.groupby("stock_id").date.shift(-1)
u = u.merge(px, on=["date", "stock_id"], how="left")
imp = pd.read_parquet("impact_panel.parquet", columns=["date", "stock_id", "gap_x", "oc_x", "oc", "big_buy", "net_fi", "Trading_Volume"])
imp = imp.rename(columns={"date": "date_next", "big_buy": "bb_next", "net_fi": "net_fi_next", "Trading_Volume": "vol_next"})
u = u.merge(imp, on=["date_next", "stock_id"], how="left")
u["fi_next"] = u.net_fi_next / u.vol_next

STEPS = [
    ("近20日投信買進 >5 天", lambda x: x.it_days20 > 5),
    ("+ 外資5日累計 ≤ −0.5 均量", lambda x: x.fi_cum5 <= -0.5),
    ("+ 20日漲 0~20% 且今天沒漲停", lambda x: x.ret20.between(0, .20) & (x.ret1 < .095)),
    ("+ 距60日高 10% 以內", lambda x: x.dist_hi60 >= -.10),
    ("+ 同產業今天 ≥5% 被投信大買", lambda x: x.peer_bb >= .05),
]


def ev(x):
    return pd.Series({
        "n": len(x), "每年": len(x) / x.date.str[:4].nunique(),
        "命中%": 100 * x.y.mean(),
        "T+1跳空bp": 1e4 * x.gap_x.mean(), "T+1盤中超額bp": 1e4 * x.oc_x.mean(),
        "盤中t": x.oc_x.mean() / x.oc_x.std() * np.sqrt(x.oc_x.count()),
        "盤中勝率%": 100 * (x.oc_x > 0).mean(), "T+1盤中原始bp": 1e4 * x.oc.mean()})


print("基準(全部):"); print(u.groupby("per").apply(ev, include_groups=False).round(1).to_string())
m = pd.Series(True, index=u.index)
rows = []
for lab, f in STEPS:
    m &= f(u).fillna(False)
    s = u[m].groupby("per").apply(ev, include_groups=False)
    s.index = [f"{lab} [{p}]" for p in s.index]
    rows.append(s)
print("\n== 逐步疊加 ==")
print(pd.concat(rows).round(1).to_string())

# 拆解:最後一層裡,隔天真的大買 vs 沒大買;以及隔天外資反向與否(事後,用來理解)
last = u[m]
print("\n== 最後一層拆解(事後) ==")
print(last.groupby(last.y.rename("隔天真的大買")).apply(ev, include_groups=False).round(1).to_string())
last = last.assign(外資隔天=np.select([last.fi_next <= -.05, last.fi_next >= .05], ["反向", "同向"], "±5%"))
print(last.groupby("外資隔天").apply(ev, include_groups=False).round(1).to_string())
print("\n依年:"); print(last.groupby(last.date.str[:4]).apply(ev, include_groups=False).round(1).to_string())
