"""使用者假設：金融股影響小是因為波動小、掛單厚。
衝擊改用個股自己的盤中波動當單位:z = oc_x / σ20,σ20 = 前 20 日 oc_x 標準差(T-1 以前)。
若金融 z ≈ 電子 → 只是波動小;若金融 z 仍明顯小 → 流量性質不同(如被動 ETF)。"""
import runpy, io, contextlib
import numpy as np
import pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("05_large_cap_check.py")
u = ns["u"].sort_values(["stock_id", "date"]).copy()
u["sig20"] = u.groupby("stock_id").oc_x.transform(lambda s: s.rolling(20, min_periods=15).std().shift(1))
u = u.dropna(subset=["sig20"])
u = u[u.sig20 > 0]
u["z"] = u.oc_x / u.sig20
b = u[u.big_buy]
t = b.groupby(["sz", "grp"], observed=True).agg(
    n=("z", "size"), oc_bp=("oc_x", lambda x: 1e4 * x.mean()), sigma_bp=("sig20", lambda x: 1e4 * x.median()),
    z_mean=("z", "mean"), z_med=("z", "median"), 淨買佔量中位=("net_vol", "median"), 淨買佔均量中位=("f_it", "median"))
print(t.round(2).to_string())
s = u[u.big_sell]
print("\n大賣(z):")
print(s.groupby(["sz", "grp"], observed=True).agg(n=("z", "size"), oc_bp=("oc_x", lambda x: 1e4 * x.mean()),
      z_mean=("z", "mean")).round(2).to_string())
