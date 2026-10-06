"""使用者假設：大型股投信大買時外資常對做,衝擊被吸收。
大買/大賣日依 外資當日淨額 / 當日成交量 分組(同向 / 小 / 反向),各市值組比較盤中超額。排除金融,2020–2025。"""
import runpy, io, contextlib
import numpy as np
import pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("07_size_bins.py")
u = ns["u"]
u["fi_vol"] = (u.buy_fi - u.sell_fi) / u.Trading_Volume
u["sz3"] = pd.cut(u.mcap, [0, 200, 500, 2000, 1e9], labels=["<200億", "200-500億", "500-2000億", "≥2000億"], right=False)
for lab, m, sgn in [("大買", u.big_buy, 1), ("大賣", u.big_sell, -1)]:
    x = u[m].copy()
    f = sgn * x.fi_vol   # >0 = 外資同向
    x["外資"] = np.select([f <= -0.05, f >= 0.05], ["反向≥5%量", "同向≥5%量"], "±5%內")
    t = x.groupby(["sz3", "外資"], observed=True).agg(n=("oc_x", "size"), oc_bp=("oc_x", lambda v: 1e4 * v.mean()),
                                                     z=("z", "mean")).round(2).unstack("外資")
    print(f"\n== {lab}日:依外資當日方向 ==")
    print(t.to_string())
    print("外資反向比例:", (x.assign(r=(sgn * x.fi_vol <= -0.05)).groupby("sz3", observed=True).r.mean() * 100).round(0).to_dict())
