"""第四關-3:名單內,10:00 / 11:00 的盤中特徵 → 今天投信大買機率 & 檢查點→收盤的剩餘報酬。
剩餘報酬超額 = 個股 (收盤/檢查點價 − 1) − 同日名單內所有股票平均(粗略大盤調整)。
期間切兩段:A 2025-03~08、B 2025-09~2026-02。成本參考:當沖 25–35 bp。
"""
import numpy as np
import pandas as pd

f = pd.read_parquet("tick_features.parquet")
f["per"] = np.where(f.date < "2025-09-01", "A", "B")
f["adv_lot"] = f.adv20 / 1000
for tag in ["0920", "0930", "1000"]:
    frac = (f[f"v{tag}"] / f.vol_day).median()
    f[f"vr{tag}"] = f[f"v{tag}"] / (f.adv_lot * frac)                      # 相對平常同時段的量
    f[f"ret_o{tag}"] = f[f"p{tag}"] / f.open - 1
    rem = f.close / f[f"p{tag}"] - 1
    f[f"rem{tag}"] = rem - rem.groupby(f.date).transform("mean")
print("樣本 %d stock-days,今天投信大買率 %.1f%%(A %.1f%% / B %.1f%%)" % (
    len(f), 100 * f.big_buy.mean(), 100 * f[f.per == "A"].big_buy.mean(), 100 * f[f.per == "B"].big_buy.mean()))
print("大買日 vs 其他:收盤相對開盤 %+.0f vs %+.0f bp(粗略超額)" % (
    1e4 * (f.close / f.open - 1 - (f.close / f.open - 1).groupby(f.date).transform("mean"))[f.big_buy].mean(),
    1e4 * (f.close / f.open - 1 - (f.close / f.open - 1).groupby(f.date).transform("mean"))[~f.big_buy].mean()))

BINS = {
    "imb": ([-1, -.2, -.1, 0, .1, .2, 1.01], ["≤-20%", "-20~-10", "-10~0", "0~10", "10~20", ">20%"]),
    "simb": ([-1, -.2, -.1, 0, .1, .2, 1.01], ["≤-20%", "-20~-10", "-10~0", "0~10", "10~20", ">20%"]),
    "spers": ([-.01, .3, .4, .5, .6, .7, 1.01], ["<30%", "30-40", "40-50", "50-60", "60-70", "≥70%"]),
    "vr": ([0, .5, 1, 1.5, 2, 3, 1e9], ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "≥3"]),
    "ret_o": ([-1, -.02, -.01, 0, .01, .02, .04, 1], ["≤-2%", "-2~-1", "-1~0", "0~1", "1~2", "2~4", ">4%"]),
}
NAMES = {"imb": "外盤−內盤 / 量(全部成交)", "simb": "外盤−內盤 / 量(1–5 張小單)", "spers": "小單偏外盤的分鐘比例",
         "vr": "量 / 平常同時段", "ret_o": "開盤到檢查點報酬"}


def ev(x, tag):
    return pd.Series({"n": len(x), "大買%": 100 * x.big_buy.mean(), f"剩餘超額bp": 1e4 * x[f"rem{tag}"].mean(),
                      "t": x[f"rem{tag}"].mean() / x[f"rem{tag}"].std() * np.sqrt(len(x))})


for tag in ["0920", "0930", "1000"]:
    print(f"\n################ 檢查點 {tag} ################")
    for k, (edges, labs) in BINS.items():
        col = f"{k}{tag}"
        x = f.dropna(subset=[col]).copy()
        x["b"] = pd.cut(x[col], edges, labels=labs)
        t = x.groupby(["b", "per"], observed=True).apply(lambda g: ev(g, tag), include_groups=False).unstack("per")
        print(f"\n-- {NAMES[k]} --"); print(t.round(1).to_string())


# 組合(整數門檻):小單持續偏外盤 + 小單淨外盤 + 量能
print("\n################ 組合 ################")
for tag in ["0920", "0930", "1000"]:
    rows = {}
    for lab, m in [("小單偏外盤分鐘 ≥60%", f[f"spers{tag}"] >= .6),
                   ("+ 小單淨外盤 ≥10%", (f[f"spers{tag}"] >= .6) & (f[f"simb{tag}"] >= .1)),
                   ("+ 量 ≥ 平常 1 倍", (f[f"spers{tag}"] >= .6) & (f[f"simb{tag}"] >= .1) & (f[f"vr{tag}"] >= 1)),
                   ("+ 全部成交淨外盤 ≥10%", (f[f"spers{tag}"] >= .6) & (f[f"simb{tag}"] >= .1) & (f[f"imb{tag}"] >= .1)),
                   ("小單偏外盤分鐘 ≥70%", f[f"spers{tag}"] >= .7)]:
        for pp, x in f[m.fillna(False)].groupby("per"):
            rows[(tag, lab, pp)] = ev(x, tag)
    print(pd.DataFrame(rows).T.round(1).to_string())
