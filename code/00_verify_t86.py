"""第 0 步：FinMind Investment_Trust vs 證交所 T86 原始報表抽查(上市)。"""
import glob, time, requests
import pandas as pd

def finmind(date):
    y = date[:4]
    x = pd.read_parquet(f"/Users/chih/quant/data/raw/finmind_daily/institutional/year={y}/data.parquet")
    x = x[(x.date.astype(str) == date) & (x.name == "Investment_Trust")]
    return x.set_index("stock_id")[["buy", "sell"]]

def t86(date):
    r = requests.get("https://www.twse.com.tw/rwd/zh/fund/T86",
                     params={"date": date.replace("-", ""), "selectType": "ALLBUT0999", "response": "json"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30).json()
    f = r["fields"]
    df = pd.DataFrame(r["data"], columns=f)
    num = lambda s: pd.to_numeric(s.str.replace(",", ""), errors="coerce")
    bcol = [c for c in f if c.startswith("投信買進")][0]
    scol = [c for c in f if c.startswith("投信賣出")][0]
    return pd.DataFrame({"stock_id": df["證券代號"].str.strip(), "buy": num(df[bcol]), "sell": num(df[scol])}).set_index("stock_id"), (bcol, scol)

for d in ["2019-06-18", "2023-03-15", "2026-09-16"]:
    fm = finmind(d)
    tw, cols = t86(d)
    j = tw.join(fm, rsuffix="_fm", how="inner")
    act = j[(j.buy + j.sell) > 0]
    match = ((act.buy == act.buy_fm) & (act.sell == act.sell_fm)).mean()
    only_tw = len(tw[(tw.buy + tw.sell) > 0].index.difference(fm.index))
    print(f"{d} 欄位{cols}: 證交所有投信交易 {len(tw[(tw.buy+tw.sell)>0])} 檔, 對上 FinMind {len(act)} 檔, 完全一致 {match:.1%}, 證交所有但 FinMind 沒有 {only_tw}")
    bad = act[(act.buy != act.buy_fm) | (act.sell != act.sell_fm)]
    if len(bad): print(bad.head(5).to_string())
    time.sleep(4)
