"""使用者觀察：國巨(2327)市值 > 2000 億但似乎很受投信影響。
檢查 ≥2000 億的股票裡，投信影響力的個股差異，以及什麼特徵能分辨。"""
import numpy as np
import pandas as pd

u = pd.read_parquet("impact_panel.parquet")
u = u[u.year >= "2020"].copy()
so = pd.read_parquet("/Users/chih/quant/data/raw/finmind_daily/shares_outstanding_quarter.parquet")
so["snap_date"] = pd.to_datetime(so.snap_date.astype(str))
u["dt"] = pd.to_datetime(u.date)
u = pd.merge_asof(u.sort_values("dt"), so.sort_values("snap_date")[["snap_date", "stock_id", "shares_outstanding"]],
                  left_on="dt", right_on="snap_date", by="stock_id", direction="backward")
u["mcap"] = u.shares_outstanding * u.prev_close / 1e8
u = u.dropna(subset=["mcap", "oc_x"])
u["part_fi"] = (u.buy_fi + u.sell_fi) / (2 * u.Trading_Volume)
big = u[u.mcap >= 2000]

print("== 國巨 2327 ==")
k = u[u.stock_id == "2327"]
print(k.groupby("year").agg(市值億=("mcap", "median"), 大買=("big_buy", "sum"), 大賣=("big_sell", "sum"),
                            投信參與率=("part_it", "mean"), 外資參與率=("part_fi", "median")).round(3).to_string())
for lab, m in [("大買", k.big_buy), ("大賣", k.big_sell)]:
    x = k[m]
    print("%s n=%d 盤中超額 %+.0f bp 中位 %+.0f 勝率 %.0f%%" % (lab, len(x), 1e4 * x.oc_x.mean(), 1e4 * x.oc_x.median(), 100 * (x.oc_x > 0).mean()))
kk = k[k.mcap >= 2000]
print("只看國巨市值 ≥2000 億的日子:大買 n=%d %+.0f bp / 大賣 n=%d %+.0f bp" % (
    kk.big_buy.sum(), 1e4 * kk[kk.big_buy].oc_x.mean(), kk.big_sell.sum(), 1e4 * kk[kk.big_sell].oc_x.mean()))

print("\n== ≥2000 億:各股大買日盤中超額(大買 ≥ 10 次) ==")
s = big.groupby("stock_id").agg(n_buy=("big_buy", "sum"), n_sell=("big_sell", "sum"),
                                投信參與率=("part_it", "mean"), 外資參與率=("part_fi", "median"),
                                市值億=("mcap", "median"))
s["buy_oc"] = big[big.big_buy].groupby("stock_id").oc_x.mean() * 1e4
s["sell_oc"] = big[big.big_sell].groupby("stock_id").oc_x.mean() * 1e4
s = s[s.n_buy >= 10].sort_values("buy_oc", ascending=False)
print(s.round(3).head(15).to_string()); print("…"); print(s.round(3).tail(8).to_string())

print("\n== 換個切法:依『投信參與率』(過去 60 日平均,T-1 以前)分組,不管市值 ==")
u = u.sort_values(["stock_id", "date"])
u["it_share60"] = u.groupby("stock_id").part_it.transform(lambda p: p.rolling(60, min_periods=40).mean().shift(1))
u["fi_share60"] = u.groupby("stock_id").part_fi.transform(lambda p: p.rolling(60, min_periods=40).mean().shift(1))
u["itb"] = pd.cut(u.it_share60, [0, .01, .02, .05, .1, 1], labels=["<1%", "1-2%", "2-5%", "5-10%", "≥10%"], right=False)
u["sz"] = pd.cut(u.mcap, [0, 500, 2000, 1e9], labels=["<500億", "500-2000億", "≥2000億"], right=False)
def ev(x):
    return pd.Series({"n": len(x), "oc_bp": 1e4 * x.oc_x.mean(), "win%": 100 * (x.oc_x > 0).mean()})
t = u[u.big_buy].groupby(["sz", "itb"], observed=True).apply(ev, include_groups=False).round(1)
print(t.unstack("sz").to_string())

print("\n== 依產業 × 市值:大買日盤中超額 ==")
fin = {"金融保險業", "金融業", "金融保險"}
trad_div = {"水泥工業", "食品工業", "塑膠工業", "紡織纖維", "鋼鐵工業", "油電燃氣業", "航運業", "貿易百貨", "通信網路業"}
u["grp"] = np.where(u.industry_category.isin(fin), "金融",
           np.where(u.industry_category.isin(trad_div), "傳產/電信", "電子/其他"))
t = u[u.big_buy].groupby(["sz", "grp"], observed=True).apply(ev, include_groups=False).round(1)
print(t.to_string())
print("\n每檔每年平均大買次數(股票-年):")
c = u.groupby(["sz", "grp", "stock_id", "year"], observed=True).big_buy.sum().groupby(["sz", "grp"], observed=True).mean().round(1)
print(c.unstack("grp").to_string())
