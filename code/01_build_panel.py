"""建投信流量 panel + 第 0 步描述。

範圍：4 碼普通股(不以 0 開頭)，上市+上櫃。
大買：投信淨買 >= 當日成交量 10% 且 >= 20 日均量 5%；大賣反向。
20 日均量 / 均額只用 T-1 以前(shift 1)，避免把當天量放進門檻。
最後一天(2026-09-16)是初版數字，排除。
"""
import glob
import numpy as np
import pandas as pd

R = "/Users/chih/quant/data/raw/finmind_daily"
inst = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{R}/institutional/year=*/*.parquet"))])
inst = inst[inst.name.isin(["Investment_Trust", "Foreign_Investor"])]
inst["date"] = inst.date.astype(str)
w = inst.pivot_table(index=["date", "stock_id"], columns="name", values=["buy", "sell"], aggfunc="sum")
w.columns = [f"{a}_{'it' if b == 'Investment_Trust' else 'fi'}" for a, b in w.columns]
w = w.reset_index()

px = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{R}/price/year=*/*.parquet"))])
px["date"] = px.date.astype(str)
px = px[px.stock_id.str.fullmatch(r"[1-9]\d{3}")]
info = pd.read_parquet(f"{R}/info.parquet").drop_duplicates("stock_id", keep="last")[["stock_id", "industry_category", "type"]]

df = px.merge(w, on=["date", "stock_id"], how="left").merge(info, on="stock_id", how="left")
for c in ["buy_it", "sell_it", "buy_fi", "sell_fi"]:
    df[c] = df[c].fillna(0)
df = df[(df.Trading_Volume > 0) & (df.date < "2026-09-16")].sort_values(["stock_id", "date"])
g = df.groupby("stock_id")
df["adv20"] = g.Trading_Volume.transform(lambda s: s.rolling(20, min_periods=15).mean().shift(1))
df["adm20"] = g.Trading_money.transform(lambda s: s.rolling(20, min_periods=15).mean().shift(1))
df["net_it"] = df.buy_it - df.sell_it
df["net_fi"] = df.buy_fi - df.sell_fi
df["f_it"] = df.net_it / df.adv20                      # 投信流量 / 20 日均量
df["f_fi"] = df.net_fi / df.adv20
df["part_it"] = (df.buy_it + df.sell_it) / (2 * df.Trading_Volume)  # 投信參與率(單邊)
df["net_vol"] = df.net_it / df.Trading_Volume
df["big_buy"] = (df.net_vol >= 0.10) & (df.f_it >= 0.05)
df["big_sell"] = (df.net_vol <= -0.10) & (df.f_it <= -0.05)
df["liquid"] = df.adm20 >= 1e8
df["year"] = df.date.str[:4]
df.to_parquet("panel.parquet")

u = df[df.liquid]
print("panel", df.shape, df.date.min(), "~", df.date.max())
print("\n== 流動性範圍(20 日均額 >= 1 億)內，每年 ==")
t = u.groupby("year").agg(股票日=("stock_id", "size"), 平均檔數=("stock_id", lambda s: s.size / u.loc[s.index, "date"].nunique()),
                          大買=("big_buy", "sum"), 大賣=("big_sell", "sum"),
                          投信有交易比例=("part_it", lambda p: (p > 0).mean()),
                          投信參與率中位_有交易=("part_it", lambda p: p[p > 0].median()))
t["大買率%"] = 100 * t.大買 / t.股票日; t["大賣率%"] = 100 * t.大賣 / t.股票日
print(t.round(3).to_string())

print("\n== 同一天大買檔數分布(集中 → 可能是 ETF 換股 / 申購) ==")
d = u.groupby("date").agg(nb=("big_buy", "sum"), ns=("big_sell", "sum"))
print("每天大買檔數分位:", d.nb.quantile([.5, .9, .99, 1]).to_dict(), " 大賣:", d.ns.quantile([.5, .9, .99, 1]).to_dict())
print("大買最集中的 15 天:\n", d.sort_values("nb", ascending=False).head(15).to_string())
print("\n事件佔比：前 5%% 的日子佔大買事件 %.0f%%" % (100 * d.nb.sort_values(ascending=False).head(int(len(d) * .05)).sum() / d.nb.sum()))
