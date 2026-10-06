"""控制「前一天大跌 = 壞消息」:同樣的 T-1 單日跌幅下,第一次跌破成本深度是否仍多賣。
前一天投信沒賣(net_prev >= 0)的子樣本。"""
import numpy as np
import pandas as pd

h = pd.read_parquet("episode_days.parquet")
px = pd.read_parquet("panel.parquet", columns=["date", "stock_id", "close"]).sort_values(["stock_id", "date"])
px["k"] = px.groupby("stock_id").cumcount()
px["r1"] = px.groupby("stock_id").close.pct_change()
h = h.merge(px[["stock_id", "k", "r1"]].rename(columns={"r1": "r_prev"}), on=["stock_id", "k"], how="left")
c = h[(h.net_prev >= 0)].copy()
c["deep"] = np.select([c.ret_cost < -0.10, c.ret_cost < -0.05, c.ret_cost < 0], ["<-10%", "-10~-5%", "-5~0%"], "獲利中")
c["rb"] = pd.cut(c.r_prev, [-1, -.05, -.02, 0, 1], labels=["T-1 跌≥5%", "跌2~5%", "跌0~2%", "T-1 漲"])
t = c.groupby(["rb", "deep", "new_low"], observed=True).agg(n=("sell10", "size"), sell10=("sell10", lambda s: 100 * s.mean()),
                                                         big_sell=("big_sell", lambda s: 100 * s.mean())).round(1)
print(t.to_string())
# 摘要:同 T-1 跌幅下,<-10% 新低 vs 獲利中
