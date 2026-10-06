"""使用者假設：權值股上投信影響力較小(外資主導)。
市值 = 最近一季(as-of)流通股數 × T-1 收盤；整數門檻分組。2020–2025。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

u = pd.read_parquet("impact_panel.parquet")
u = u[u.year >= "2020"].copy()
so = pd.read_parquet("/Users/chih/quant/data/raw/finmind_daily/shares_outstanding_quarter.parquet")
so["snap_date"] = pd.to_datetime(so.snap_date.astype(str))
u["dt"] = pd.to_datetime(u.date)
u = pd.merge_asof(u.sort_values("dt"), so.sort_values("snap_date")[["snap_date", "stock_id", "shares_outstanding"]],
                  left_on="dt", right_on="snap_date", by="stock_id", direction="backward")
u["mcap"] = u.shares_outstanding * u.prev_close / 1e8   # 億
u = u.dropna(subset=["mcap", "oc_x"])
u["size"] = pd.cut(u.mcap, [0, 100, 500, 2000, 1e9], labels=["<100億", "100-500億", "500-2000億", "≥2000億"], right=False)
u["part_fi"] = (u.buy_fi + u.sell_fi) / (2 * u.Trading_Volume)

print("== 各市值組：基本面貌(中位數) ==")
print(u.groupby("size", observed=True).agg(
    股票日=("oc_x", "size"), 檔數=("stock_id", "nunique"),
    投信參與率=("part_it", "median"), 外資參與率=("part_fi", "median"),
    投信有交易=("part_it", lambda p: (p > 0).mean()),
    大買率pct=("big_buy", lambda b: 100 * b.mean()), 大賣率pct=("big_sell", lambda b: 100 * b.mean())).round(3).to_string())

def ev(x):
    return pd.Series({"n": len(x), "oc_bp": 1e4 * x.oc_x.mean(), "oc_med_bp": 1e4 * x.oc_x.median(),
                      "t": x.oc_x.mean() / x.oc_x.std() * np.sqrt(len(x)), "win%": 100 * (x.oc_x > 0).mean()})
print("\n== 大買日 盤中超額，依市值 ==")
print(u[u.big_buy].groupby("size", observed=True).apply(ev, include_groups=False).round(1).to_string())
print("\n== 大賣日 ==")
print(u[u.big_sell].groupby("size", observed=True).apply(ev, include_groups=False).round(1).to_string())

print("\n== 迴歸 oc_x(bp) ~ f_it + f_fi，各市值組分開，雙向群聚 ==")
lo_hi = {c: u[c].quantile([.01, .99]).values for c in ["f_it", "f_fi"]}
rows = []
for s, d in u.groupby("size", observed=True):
    d = d.dropna(subset=["f_it", "f_fi"]).copy()
    for c in ["f_it", "f_fi"]: d[c] = d[c].clip(*lo_hi[c])
    gg = np.column_stack([pd.factorize(d.date)[0], pd.factorize(d.stock_id)[0]])
    m = sm.OLS(1e4 * d.oc_x, sm.add_constant(d[["f_it", "f_fi"]])).fit(cov_type="cluster", cov_kwds={"groups": gg})
    # 同樣「金額」的衝擊：每 1 億淨買的 bp（用該組中位 均額 換算）
    rows.append({"size": s, "n": int(m.nobs), "β投信": m.params.f_it, "t投信": m.tvalues.f_it,
                 "β外資": m.params.f_fi, "t外資": m.tvalues.f_fi, "投信/外資": m.params.f_it / m.params.f_fi,
                 "中位均額億": d.adm20.median() / 1e8, "R2": m.rsquared})
r = pd.DataFrame(rows).set_index("size")
r["投信每1億淨買_bp"] = r["β投信"] / r["中位均額億"]
print(r.round(2).to_string())
