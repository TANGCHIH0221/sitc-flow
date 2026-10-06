"""16 的發現:能預測到的大買多是外資在賣的那種 → 衝擊被吸收。
改成『外資沒在賣』的版本,看命中率與 T+1 報酬的取捨。"""
import runpy, io, contextlib
import numpy as np
import pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("16_combo.py")
u, ev = ns["u"], ns["ev"]
core = (u.ret20.between(0, .20) & (u.ret1 < .095) & (u.dist_hi60 >= -.10)).fillna(False)
rows = {}
for it_lab, itm in [("投信近20日買 >5 天", u.it_days20 > 5), ("投信近20日買 1~5 天", u.it_days20.between(1, 5))]:
    for fi_lab, fim in [("外資5日 ≤ −0.5(在賣)", u.fi_cum5 <= -.5), ("外資5日 −0.5~0", u.fi_cum5.between(-.5, 0)),
                        ("外資5日 0~0.5", u.fi_cum5.between(0, .5)), ("外資5日 ≥ 0.5(在買)", u.fi_cum5 >= .5)]:
        x = u[core & itm & fim.fillna(False)]
        for p, xx in x.groupby("per"):
            rows[(it_lab, fi_lab, p)] = ev(xx)
print(pd.DataFrame(rows).T.round(1).to_string())
