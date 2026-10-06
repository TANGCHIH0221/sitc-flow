"""盤前觀察名單:每天幾檔、命中率、涵蓋率(隔天全部投信大買有幾成在名單裡)、名單內大買的 T+1 盤中超額。"""
import runpy, io, contextlib
import numpy as np
import pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("16_combo.py")
u = ns["u"]
v = u[u.per == "val"]          # 2024–2025
days = v.date.nunique()
tot = v.y.sum()
R = {
    "A 投信近20日買 ≥1 天": v.it_days20 >= 1,
    "B 投信近20日買 >5 天": v.it_days20 > 5,
    "C B + 20日漲0~20% 沒漲停 距高≤10%": (v.it_days20 > 5) & v.ret20.between(0, .2) & (v.ret1 < .095) & (v.dist_hi60 >= -.1),
    "D C + 外資5日 ≤ −0.5": (v.it_days20 > 5) & v.ret20.between(0, .2) & (v.ret1 < .095) & (v.dist_hi60 >= -.1) & (v.fi_cum5 <= -.5),
    "E 投信近20日買 >10 天": v.it_days20 > 10,
}
rows = {}
for lab, m in R.items():
    x = v[m.fillna(False)]
    hit = x[x.y]
    rows[lab] = {"每天幾檔": len(x) / days, "命中%": 100 * x.y.mean(), "涵蓋率%": 100 * x.y.sum() / tot,
                 "名單內大買 T+1盤中bp": 1e4 * hit.oc_x.mean()}
out = v[~(v.it_days20 >= 1) & v.y]
rows["(名單外:近20日完全沒買)"] = {"每天幾檔": np.nan, "命中%": np.nan, "涵蓋率%": 100 * len(out) / tot,
                              "名單內大買 T+1盤中bp": 1e4 * out.oc_x.mean()}
print("2024–2025,%d 天,隔天投信大買共 %d 次(每天 %.0f 次)" % (days, tot, tot / days))
print(pd.DataFrame(rows).T.round(1).to_string())
