"""停損訊號在價格上有沒有東西:T-1 收盤後已知「部位第一次跌破成本 −10%」→ T 日跳空 / 盤中超額。
對照:同部位中 獲利中、虧損但未創新低。前一天投信沒賣的子樣本。"""
import numpy as np
import pandas as pd

h = pd.read_parquet("episode_days.parquet")
imp = pd.read_parquet("impact_panel.parquet", columns=["date", "stock_id", "gap_x", "oc_x"])
h = h.merge(imp, on=["date", "stock_id"], how="inner")
c = h[h.net_prev >= 0].copy()
c["grp"] = np.select(
    [(c.ret_cost < -0.10) & c.new_low, (c.ret_cost < -0.10) & ~c.new_low, (c.ret_cost < 0), c.ret_cost >= 0],
    ["跌破-10%(第一次)", "<-10%(之前跌過)", "虧損0~10%", "獲利中"], "?")
def ev(x):
    cc = x.gap_x + x.oc_x
    return pd.Series({"n": len(x), "隔天賣10%部位%": 100 * x.sell10.mean(),
                      "跳空bp": 1e4 * x.gap_x.mean(), "盤中bp": 1e4 * x.oc_x.mean(),
                      "盤中t": x.oc_x.mean() / x.oc_x.std() * np.sqrt(x.oc_x.count()),
                      "盤中勝率%(跌)": 100 * (x.oc_x < 0).mean()})
print(c.groupby("grp").apply(ev, include_groups=False).round(2).to_string())
s = c[c.grp == "跌破-10%(第一次)"]
print("\n跌破 -10% 第一次:依年")
print(s.groupby(s.date.str[:4]).apply(ev, include_groups=False).round(1).to_string())
print("\n跌破 -10% 第一次:當天投信真的賣 vs 沒賣(事後)")
print(s.groupby(s.sell10.astype(bool)).apply(ev, include_groups=False).round(1).to_string())
