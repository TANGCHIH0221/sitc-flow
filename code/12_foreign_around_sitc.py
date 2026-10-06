"""外資與投信:(1) 投信大買/大賣日前後 ±5 日外資淨額/量(卡位?接手?)
(2) 第 0 天外資反向時,之後外資是否續反向 (3) 外資自身延續性:投信事件股 vs 其他。
非金融、20 日均額 >= 1 億、2019–2025。外資大買定義同投信:淨買 >= 當日量 10% 且 >= ADV20 5%。"""
import numpy as np
import pandas as pd

FIN = {"金融保險業", "金融業", "金融保險"}
df = pd.read_parquet("panel.parquet", columns=["date", "stock_id", "net_it", "net_fi", "Trading_Volume", "adv20",
                                               "liquid", "industry_category", "big_buy", "big_sell"])
df = df[(df.date <= "2025-12-31") & ~df.industry_category.isin(FIN)].sort_values(["stock_id", "date"]).reset_index(drop=True)
df["fi_vol"] = df.net_fi / df.Trading_Volume
df["it_vol"] = df.net_it / df.Trading_Volume
df["fi_big_buy"] = (df.fi_vol >= .10) & (df.net_fi / df.adv20 >= .05)
df["fi_big_sell"] = (df.fi_vol <= -.10) & (df.net_fi / df.adv20 <= -.05)
g = df.groupby("stock_id")
for k in range(-5, 6):
    df[f"fi{k}"] = g.fi_vol.shift(-k)
    df[f"it{k}"] = g.it_vol.shift(-k)
u = df[df.liquid]
base_fi, base_it = u.fi_vol.mean(), u.it_vol.mean()
print("基準:流動股票-日 外資淨額/量 平均 %.2f%%、投信 %.2f%%" % (100 * base_fi, 100 * base_it))

for lab, m in [("投信大買", u.big_buy), ("投信大賣", u.big_sell)]:
    e = u[m]
    t = pd.DataFrame({"外資%": [100 * e[f"fi{k}"].mean() for k in range(-5, 6)],
                      "投信%": [100 * e[f"it{k}"].mean() for k in range(-5, 6)]}, index=range(-5, 6)).T.round(2)
    print(f"\n== {lab}日(第 0 天)前後,淨額/當日量 平均,n={len(e)} ==")
    print(t.to_string())

print("\n== 投信大買日外資反向(≤ −5% 量) vs 沒反向:之後外資 ==")
e = u[u.big_buy].copy()
e["opp"] = e.fi0 <= -.05
t = e.groupby("opp")[[f"fi{k}" for k in range(-3, 6)]].mean().mul(100).round(2)
t.index = ["外資沒反向", "外資反向"]; print(t.to_string())
print("外資反向那組,第 1 天外資仍反向(≤ −5%%)比例 %.0f%%;全部股票-日外資 ≤ −5%% 的比例 %.0f%%" % (
    100 * (e[e.opp].fi1 <= -.05).mean(), 100 * (u.fi_vol <= -.05).mean()))

print("\n== 外資自身延續性:今天外資大買/大賣 → 明天同向 ==")
d = df.copy()
d["n_fb"] = g.fi_big_buy.shift(-1); d["n_fs"] = g.fi_big_sell.shift(-1)
d["sitc_name"] = g.big_buy.transform(lambda s: s.rolling(60, min_periods=1).max().shift(1)).fillna(0) > 0  # 過去 60 日有投信大買
d = d[d.liquid & d.n_fb.notna()]
for lab, col, nc in [("外資大買", "fi_big_buy", "n_fb"), ("外資大賣", "fi_big_sell", "n_fs")]:
    for nm, sub in [("過去 60 日有投信大買的股票", d[d.sitc_name]), ("其他股票", d[~d.sitc_name])]:
        x = sub[sub[col]]
        print("%s|%s:n=%d 明天同向 %.1f%%(基準 %.1f%%)" % (lab, nm, len(x), 100 * x[nc].astype(bool).mean(), 100 * sub[nc].astype(bool).mean()))
