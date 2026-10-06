"""第三關-1:T 收盤已知的單因子 → T+1 投信大買機率。規則式、整數門檻、單因子先測。

母體:非金融、20 日均額 >= 1 億、2019–2025;另看「沉寂」子母體(T 日投信 |淨額| < 當日量 2%)。
開發 2019–2023 / 驗證 2024–2025,兩段都要成立。
月營收沒有公布時間欄位 → 保守假設每月 11 日起可用(法定期限 10 日)。
"""
import glob
import numpy as np
import pandas as pd

R = "/Users/chih/quant/data/raw/finmind_daily"
FIN = {"金融保險業", "金融業", "金融保險"}

df = pd.read_parquet("panel.parquet", columns=[
    "date", "stock_id", "open", "close", "max", "Trading_Volume", "Trading_money", "net_it", "net_fi",
    "adv20", "adm20", "liquid", "industry_category", "big_buy", "big_sell"])
df = df[(df.date <= "2025-12-31") & ~df.industry_category.isin(FIN)].sort_values(["stock_id", "date"]).reset_index(drop=True)
g = df.groupby("stock_id")
df["it_vol"] = df.net_it / df.Trading_Volume
df["fi_vol"] = df.net_fi / df.Trading_Volume
df["it_act"] = (df.net_it.abs() >= .02 * df.adv20)
df["it_buyday"] = (df.net_it >= .02 * df.adv20)

# 投信軌跡
df["it_days20"] = g.it_buyday.transform(lambda s: s.rolling(20, min_periods=1).sum())
df["it_cum5"] = g.net_it.transform(lambda s: s.rolling(5, min_periods=1).sum()) / df.adv20
k = df.groupby("stock_id").cumcount()
last_act = k.where(df.it_act).groupby(df.stock_id).ffill()
df["days_since_it"] = (k - last_act).fillna(999)
# 外資
df["fi_cum5"] = g.net_fi.transform(lambda s: s.rolling(5, min_periods=1).sum()) / df.adv20
# 價量
df["ret1"] = g.close.pct_change()
df["ret20"] = g.close.pct_change(20)
df["hi60_prev"] = g.close.transform(lambda s: s.rolling(60, min_periods=40).max().shift(1))
df["dist_hi60"] = df.close / df.hi60_prev - 1
df["vol_ratio"] = df.Trading_Volume / df.adv20
# 族群:同產業(不含自己)今天投信大買的股票比例
ind_n = df.groupby(["date", "industry_category"]).stock_id.transform("size")
ind_bb = df.groupby(["date", "industry_category"]).big_buy.transform("sum")
df["peer_bb"] = (ind_bb - df.big_buy) / (ind_n - 1).replace(0, np.nan)
# 日曆
d = pd.to_datetime(df.date)
df["dom"] = d.dt.day
df["q_end"] = d.dt.month.isin([3, 6, 9, 12]) & (d.dt.day >= 20)
# 大盤
tx = pd.read_parquet(f"{R}/taiex.parquet")[["date", "close"]].rename(columns={"close": "tx"})
tx["date"] = tx.date.astype(str); tx = tx.sort_values("date")
tx["tx_ret1"] = tx.tx.pct_change(); tx["tx_ret5"] = tx.tx.pct_change(5)
df = df.merge(tx[["date", "tx_ret1", "tx_ret5"]], on="date", how="left")
# 月營收:yoy、12 個月新高;可用日 = 公布月 11 日
rv = pd.read_parquet(f"{R}/monthly_revenue.parquet")[["date", "stock_id", "revenue", "revenue_year", "revenue_month"]]
rv = rv.sort_values(["stock_id", "revenue_year", "revenue_month"])
rv["yoy"] = rv.groupby(["stock_id", "revenue_month"]).revenue.pct_change()
rv["rec12"] = rv.revenue >= rv.groupby("stock_id").revenue.transform(lambda s: s.rolling(12, min_periods=12).max())
rv["avail"] = pd.to_datetime(rv.date.astype(str)) + pd.Timedelta(days=10)
df["dt"] = d
df = pd.merge_asof(df.sort_values("dt"), rv[["avail", "stock_id", "yoy", "rec12"]].dropna(subset=["yoy"]).sort_values("avail"),
                   left_on="dt", right_on="avail", by="stock_id", direction="backward")
df["rev_fresh"] = (df.dt - df.avail).dt.days <= 5         # 營收剛可用 5 天內
# 融資 5 日變化(2020 起)
mg = pd.concat(pd.read_parquet(f) for f in glob.glob(f"{R}/margin/**/*.parquet", recursive=True))
mg = mg[["date", "stock_id", "MarginPurchaseTodayBalance"]].rename(columns={"MarginPurchaseTodayBalance": "mbal"})
mg["date"] = mg.date.astype(str)
df = df.merge(mg, on=["date", "stock_id"], how="left").sort_values(["stock_id", "date"]).reset_index(drop=True)
df["margin5"] = df.groupby("stock_id").mbal.pct_change(5)

# 目標:T+1 投信大買
df["y"] = df.groupby("stock_id").big_buy.shift(-1)
u = df[df.liquid & df.y.notna()].copy()
u["y"] = u.y.astype(bool)
u["per"] = np.where(u.date <= "2023-12-31", "dev", "val")
u["quiet"] = u.it_vol.abs() < .02
u.to_parquet("t1_features.parquet")

FACTORS = {
    "投信近20日買進天數": ("it_days20", [-1, 0, 2, 5, 10, 21], ["0", "1-2", "3-5", "6-10", ">10"]),
    "距上次投信動作(日)": ("days_since_it", [-1, 0, 5, 20, 60, 1e9], ["今天", "1-5", "6-20", "21-60", ">60"]),
    "外資今日淨額/量": ("fi_vol", [-1, -.2, -.1, -.05, .05, .1, .2, 1], ["≤-20%", "-20~-10", "-10~-5", "±5", "5~10", "10~20", "≥20%"]),
    "外資5日累計/均量": ("fi_cum5", [-100, -1, -.5, -.2, .2, .5, 1, 100], ["≤-1", "-1~-.5", "-.5~-.2", "±.2", ".2~.5", ".5~1", "≥1"]),
    "今日漲跌": ("ret1", [-1, -.05, -.02, 0, .02, .05, .095, 1], ["≤-5%", "-5~-2", "-2~0", "0~2", "2~5", "5~9.5", "≥9.5(漲停)"]),
    "20日報酬": ("ret20", [-1, -.2, -.1, 0, .1, .2, .4, 100], ["≤-20%", "-20~-10", "-10~0", "0~10", "10~20", "20~40", ">40%"]),
    "距60日高": ("dist_hi60", [-1, -.3, -.2, -.1, -.05, 0, 100], ["≤-30%", "-30~-20", "-20~-10", "-10~-5", "-5~0", "創60日新高"]),
    "量比(今日/20日均量)": ("vol_ratio", [0, .5, 1, 2, 3, 5, 1e9], ["<0.5", "0.5-1", "1-2", "2-3", "3-5", "≥5"]),
    "同產業今天投信大買比例": ("peer_bb", [-1, 0, .05, .1, .2, 1.01], ["0", "0-5%", "5-10%", "10-20%", ">20%"]),
    "月營收年增": ("yoy", [-100, -.2, 0, .2, .5, 1, 1e9], ["≤-20%", "-20~0", "0~20", "20~50", "50~100", ">100%"]),
    "融資5日變化": ("margin5", [-10, -.1, -.03, .03, .1, .2, 1e9], ["≤-10%", "-10~-3", "±3", "3~10", "10~20", ">20%"]),
    "大盤5日報酬": ("tx_ret5", [-1, -.05, -.02, 0, .02, .05, 1], ["≤-5%", "-5~-2", "-2~0", "0~2", "2~5", ">5%"]),
    "日期(幾號)": ("dom", [0, 5, 10, 20, 31], ["1-5", "6-10", "11-20", "21-31"]),
}
BOOL = {"季底(3/6/9/12 月 20 日後)": "q_end", "營收剛公布 5 天內": "rev_fresh", "營收 12 個月新高": "rec12"}


def table(pop, name):
    base = {p: pop[pop.per == p].y.mean() for p in ["dev", "val"]}
    print(f"\n######## 母體:{name} 基準 T+1 大買率 dev {100*base['dev']:.2f}% / val {100*base['val']:.2f}% ########")
    for lab, (col, edges, labs) in FACTORS.items():
        x = pop.dropna(subset=[col]).copy()
        x["b"] = pd.cut(x[col], edges, labels=labs)
        t = x.groupby(["b", "per"], observed=True).y.agg(["size", "mean"]).unstack("per")
        out = pd.DataFrame({"n_dev": t[("size", "dev")], "dev%": 100 * t[("mean", "dev")],
                            "倍dev": t[("mean", "dev")] / base["dev"], "n_val": t[("size", "val")],
                            "val%": 100 * t[("mean", "val")], "倍val": t[("mean", "val")] / base["val"]})
        print(f"\n-- {lab} --"); print(out.round(2).to_string())
    for lab, col in BOOL.items():
        x = pop.dropna(subset=[col])
        t = x.groupby([x[col].astype(bool), "per"]).y.mean().unstack("per")
        print(f"\n-- {lab} --  " + "  ".join(f"{k}: dev {100*t.loc[k,'dev']:.2f}% ({t.loc[k,'dev']/base['dev']:.2f}×) val {100*t.loc[k,'val']:.2f}% ({t.loc[k,'val']/base['val']:.2f}×)"
                                        for k in t.index))


table(u, "全部")
table(u[u.quiet], "沉寂(今天投信 |淨額| < 量 2%)")
