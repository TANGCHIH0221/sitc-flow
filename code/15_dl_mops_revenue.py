"""公開資訊觀測站 每月營收彙總 CSV(上市 sii / 上櫃 otc),2018-12 ~ 2025-12。
本機 finmind monthly_revenue.parquet 每年只有 12 月一筆,不能用。"""
import time, io, pathlib
import requests, pandas as pd

OUT = pathlib.Path("/Users/chih/quant/data/raw/mops_revenue")
URL = "https://mopsov.twse.com.tw/server-java/FileDownLoad?step=9&functionName=show_file2&filePath=/t21/{m}/&fileName=t21sc03_{y}_{mo}.csv"
months = [(y, mo) for y in range(2018, 2026) for mo in range(1, 13) if (y, mo) >= (2018, 12)]
for mkt in ["sii", "otc"]:
    for y, mo in months:
        f = OUT / f"{mkt}_{y}_{mo:02d}.csv"
        if f.exists() and f.stat().st_size > 1000:
            continue
        for attempt in range(3):
            try:
                r = requests.get(URL.format(m=mkt, y=y - 1911, mo=mo), timeout=60, headers={"User-Agent": "Mozilla/5.0"})
                if r.ok and len(r.content) > 1000:
                    f.write_bytes(r.content); break
            except requests.RequestException:
                pass
            time.sleep(5)
        else:
            print("FAIL", mkt, y, mo, flush=True)
        time.sleep(1.5)
print("done", len(list(OUT.glob("*.csv"))))
