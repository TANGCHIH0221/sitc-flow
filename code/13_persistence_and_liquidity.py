"""(1) 延續性:外資 vs 投信,同定義(淨買 >= 當日量 10% 且 >= ADV20 5%)。
(2) 使用者假設:T-1 外資一直賣 = 提供流動性 → T 投信大買?固定 T-1 投信狀態後看外資 T-1 賣壓的額外資訊。
非金融、20 日均額 >= 1 億、2019–2025。"""
import numpy as np
import pandas as pd

FIN = {"金融保險業", "金融業", "金融保險"}
df = pd.read_parquet("panel.parquet", columns=["date", "stock_id", "net_it", "net_fi", "Trading_Volume", "adv20",
                                               "liquid", "industry_category", "big_buy", "big_sell"])
df = df[(df.date <= "2025-12-31") & ~df.industry_category.isin(FIN)].sort_values(["stock_id", "date"]).reset_index(drop=True)
df["fi_vol"] = df.net_fi / df.Trading_Volume
df["it_vol"] = df.net_it / df.Trading_Volume
df["fb"] = (df.fi_vol >= .10) & (df.net_fi / df.adv20 >= .05)
df["fs"] = (df.fi_vol <= -.10) & (df.net_fi / df.adv20 <= -.05)
g = df.groupby("stock_id")
for c in ["big_buy", "big_sell", "fb", "fs"]:
    df["n_" + c] = g[c].shift(-1)
for c in ["fi_vol", "it_vol"]:
    df["p_" + c] = g[c].shift(1)
df["fi_5d"] = g.fi_vol.transform(lambda s: s.rolling(5).mean())     # 含今天的 5 日平均(當作 T-1 特徵時已是過去)
u = df[df.liquid & df.n_big_buy.notna()].copy()
for c in ["n_big_buy", "n_big_sell", "n_fb", "n_fs"]:
    u[c] = u[c].astype(bool)

print("== (1) 延續性:今天大買/大賣 → 明天同向 ==")
for who, b, s, nb, ns in [("投信", "big_buy", "big_sell", "n_big_buy", "n_big_sell"), ("外資", "fb", "fs", "n_fb", "n_fs")]:
    for lab, cur, nxt in [("大買", b, nb), ("大賣", s, ns)]:
        p, base = u[u[cur]][nxt].mean(), u[nxt].mean()
        print("%s%s:n=%7d 明天同向 %.1f%%  基準 %.1f%%  倍數 %.1f×" % (who, lab, u[cur].sum(), 100 * p, 100 * base, p / base))
print("\n淨額/量 一階自相關(個股內,平均):投信 %.2f 外資 %.2f" % (
    u.groupby("stock_id").apply(lambda x: x.it_vol.corr(x.p_it_vol), include_groups=False).mean(),
    u.groupby("stock_id").apply(lambda x: x.fi_vol.corr(x.p_fi_vol), include_groups=False).mean()))

print("\n== (2) T(今天)外資淨額/量 → T+1 投信大買機率,固定今天投信狀態 ==")
u["fi_bin"] = pd.cut(u.fi_vol, [-1, -.20, -.10, -.05, 0, .05, .10, 1],
                     labels=["≤-20%", "-20~-10", "-10~-5", "-5~0", "0~5", "5~10", "≥10%"])
u["it_state"] = pd.cut(u.it_vol, [-1, -.02, .02, .10, 1], labels=["投信在賣", "投信沒動(±2%)", "投信買2~10%", "投信買≥10%"])
t = u.groupby(["it_state", "fi_bin"], observed=True).n_big_buy.agg(["size", "mean"])
t["mean"] = (100 * t["mean"]).round(1)
print(t["mean"].unstack("fi_bin").to_string())
print("\n樣本數:")
print(t["size"].unstack("fi_bin").to_string())
print("\n同一張表,看明天投信大賣:")
t2 = u.groupby(["it_state", "fi_bin"], observed=True).n_big_sell.mean().mul(100).round(1)
print(t2.unstack("fi_bin").to_string())
