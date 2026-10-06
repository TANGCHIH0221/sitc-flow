"""第一關：大買有沒有影響力（事後已知 = 天花板）。

報酬三段：gap = open/prev_close-1、oc = close/open-1、post = close_{T+5}/close_T-1
超額 = 減去 當日流動性範圍等權平均 與 同產業(排除自己)平均。
gap / post 未做除權息調整(初步)，|單日漲跌| > 10.5% 視為無效(減資等)。oc 不受影響。
樣本 2019–2025；2026 保留到最後。
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_parquet("panel.parquet")
df = df.sort_values(["stock_id", "date"])
g = df.groupby("stock_id")
df["prev_close"] = g.close.shift(1)
df["gap"] = df.open / df.prev_close - 1
df["oc"] = df.close / df.open - 1
df["cc"] = df.close / df.prev_close - 1
df["post5"] = g.close.shift(-5) / df.close - 1
bad = df.cc.abs() > 0.105
df.loc[bad, ["gap", "cc"]] = np.nan
# post5 期間內有任何無效日 → 無效
df["bad_fwd"] = g["cc"].transform(lambda s: s.isna().astype(int)[::-1].rolling(5, min_periods=1).max()[::-1].shift(-1))
df.loc[df.bad_fwd == 1, "post5"] = np.nan

u = df[df.liquid & (df.year <= "2025") & (df.open > 0)].copy()
for r in ["gap", "oc", "post5"]:
    mkt = u.groupby("date")[r].transform("mean")
    s = u.groupby(["date", "industry_category"])[r].transform("sum")
    n = u.groupby(["date", "industry_category"])[r].transform("count")
    ind = (s - u[r]) / (n - 1)                         # 同產業排除自己
    u[r + "_x"] = u[r] - mkt - (ind - mkt).fillna(0)   # 等價於 r - ind（有同業時），否則 r - mkt
u.to_parquet("impact_panel.parquet")

def ev(x):
    return pd.Series({"n": len(x),
                      "gap_bp": 1e4 * x.gap_x.mean(), "oc_bp": 1e4 * x.oc_x.mean(), "oc_med_bp": 1e4 * x.oc_x.median(),
                      "oc_t": x.oc_x.mean() / x.oc_x.std() * np.sqrt(x.oc_x.count()),
                      "oc_win%": 100 * (x.oc_x > 0).mean(), "post5_bp": 1e4 * x.post5_x.mean()})

u["grp"] = np.select([u.big_buy, u.big_sell], ["大買", "大賣"], "其他")
print("== 超額報酬(bp)：大買 / 大賣 / 其他，2019–2025 ==")
print(u.groupby("grp").apply(ev, include_groups=False).round(1).to_string())
print("\n== 每年：大買 ==")
print(u[u.big_buy].groupby("year").apply(ev, include_groups=False).round(1).to_string())
print("\n== 每年：大賣 ==")
print(u[u.big_sell].groupby("year").apply(ev, include_groups=False).round(1).to_string())

# 依流量大小分組(淨買 / 當日量)
u["nv_bin"] = pd.cut(u.net_vol, [-1, -0.3, -0.2, -0.1, -0.05, -1e-9, 1e-9, 0.05, 0.1, 0.2, 0.3, 1])
print("\n== 依 投信淨買 / 當日成交量 分組 ==")
print(u.groupby("nv_bin", observed=True).apply(ev, include_groups=False).round(1).to_string())

# 迴歸：oc_x ~ f_it + f_fi（winsor 1/99），雙向群聚
reg = u.dropna(subset=["oc_x", "f_it", "f_fi"]).copy()
for c in ["f_it", "f_fi"]:
    lo, hi = reg[c].quantile([.01, .99]); reg[c] = reg[c].clip(lo, hi)
reg["sq_it"] = np.sign(reg.f_it) * np.sqrt(reg.f_it.abs())
reg["sq_fi"] = np.sign(reg.f_fi) * np.sqrt(reg.f_fi.abs())
grp = np.column_stack([pd.factorize(reg.date)[0], pd.factorize(reg.stock_id)[0]])
for y, xs in [("oc_x", ["f_it", "f_fi"]), ("oc_x", ["sq_it", "sq_fi"]), ("gap_x", ["f_it", "f_fi"]), ("post5_x", ["f_it", "f_fi"])]:
    d = reg.dropna(subset=[y])
    gg = np.column_stack([pd.factorize(d.date)[0], pd.factorize(d.stock_id)[0]])
    m = sm.OLS(1e4 * d[y], sm.add_constant(d[xs])).fit(cov_type="cluster", cov_kwds={"groups": gg})
    print(f"\n{y} ~ {xs}: n={int(m.nobs)} R2={m.rsquared:.4f}")
    print(pd.DataFrame({"coef_bp": m.params, "t": m.tvalues}).round(2).to_string())
