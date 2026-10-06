"""第四關-2:觀察名單(T-1 投信近 20 日買 > 5 天)的盤中特徵,10:00 / 11:00 兩個檢查點。
逐筆:data/raw/{tse,otc}/tick/{stock}/{stock}_{date}[_IS|_OS].parquet;量單位 = 張;內外盤 > 0 = 外盤(買方主動)、< 0 = 內盤
(上市存 ±1,上櫃存 方向 × 張數 × 100,一律取 sign)。
標記:當天投信大買(panel,收盤後公布,只當答案用)。
"""
import glob
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd

R = "/Users/chih/quant/data/raw"
FIN = {"金融保險業", "金融業", "金融保險"}
CKPTS = ["09:10", "09:20", "09:30", "10:00", "11:00"]


def tick_path():
    m = {}
    for mkt in ["tse", "otc"]:
        for f in glob.glob(f"{R}/{mkt}/tick/*/*.parquet"):
            b = os.path.basename(f).replace(".parquet", "").split("_")
            m[(b[0], b[1])] = f
    return m


def feats(args):
    sid, date, path = args
    try:
        d = pd.read_parquet(path, columns=["撮合時間", "成交價", "累計成交數量", "成交序號", "內外盤", "開盤註記", "收盤註記"])
    except Exception:
        return None
    d = d[(d.成交序號 > 0) & (d.成交價 > 0)].copy()
    if len(d) < 50:
        return None
    d["t"] = d.撮合時間.astype(str).str[:8]
    d["內外盤"] = np.sign(d.內外盤)            # 上櫃檔存「方向 × 張數 × 100」,上市存 ±1 → 只取方向
    d["q"] = d.累計成交數量.diff().fillna(d.累計成交數量)
    op = d.iloc[0]
    cont = d[(d.t >= "09:00:00") & (d.t < "13:25:00") & (d.開盤註記 == 0)].iloc[1:]   # 連續交易,去掉開盤競價那筆
    close = d[d.收盤註記 != 0]
    out = {"stock_id": sid, "date": date, "open": op.成交價,
           "close": (close.成交價.iloc[-1] if len(close) else d.成交價.iloc[-1]), "vol_day": d.累計成交數量.iloc[-1]}
    for c in CKPTS:
        x = cont[cont.t < c + ":00"]
        tag = c.replace(":", "")
        if len(x) == 0:
            return None
        sgn = x.內外盤 * x.q
        small = x.q <= 5
        out[f"v{tag}"] = d[d.t < c + ":00"].累計成交數量.iloc[-1]
        out[f"p{tag}"] = x.成交價.iloc[-1]
        out[f"imb{tag}"] = sgn.sum() / x.q.sum()
        out[f"simb{tag}"] = sgn[small].sum() / max(x.q[small].sum(), 1)
        out[f"limb{tag}"] = sgn[x.q >= 20].sum() / max(x.q[x.q >= 20].sum(), 1)
        mi = x[small].assign(m=x.t.str[:5]).groupby("m").apply(lambda g: (g.內外盤 * g.q).sum(), include_groups=False)
        mi = mi[mi.index >= "09:05"]
        out[f"spers{tag}"] = (mi > 0).mean() if len(mi) else np.nan
        out[f"n_small{tag}"] = int(small.sum())
    return out


if __name__ == "__main__":
    pan = pd.read_parquet("panel.parquet", columns=["date", "stock_id", "net_it", "adv20", "big_buy", "liquid",
                                                    "industry_category", "Trading_Volume"])
    pan = pan[~pan.industry_category.isin(FIN)].sort_values(["stock_id", "date"])
    g = pan.groupby("stock_id")
    pan["buyday"] = pan.net_it >= .02 * pan.adv20
    pan["it_days20_prev"] = g.buyday.transform(lambda s: s.rolling(20, min_periods=1).sum().shift(1))
    pan["liquid_prev"] = g.liquid.shift(1)
    w = pan[(pan.date >= "2025-03-01") & (pan.date <= "2026-02-28") & (pan.it_days20_prev > 5) & (pan.liquid_prev == True)]
    tp = tick_path()
    jobs = [(s, d, tp[(s, d)]) for s, d in zip(w.stock_id, w.date) if (s, d) in tp]
    print("名單 stock-days %d,有逐筆 %d" % (len(w), len(jobs)), flush=True)
    with Pool(4) as p:
        res = [r for r in p.imap_unordered(feats, jobs, chunksize=50) if r]
    f = pd.DataFrame(res).merge(w[["date", "stock_id", "big_buy", "net_it", "adv20", "Trading_Volume", "industry_category"]],
                                on=["date", "stock_id"])
    f.to_parquet("tick_features.parquet")
    print("完成 %d 筆" % len(f))
