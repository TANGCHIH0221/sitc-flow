"""第 0 步：FinMind Investment_Trust vs 櫃買中心三大法人(上櫃)抽查。
欄位名重複，依位置：0 代號 1 名稱 | 2-4 外資(不含自營) | 5-7 外資自營 | 8-10 外資合計 | 11-13 投信 | 14-16 自營自行 | 17-19 自營避險 | 20-22 自營合計 | 23 合計
"""
import time, requests
import pandas as pd
exec(open("00_verify_t86.py").read().split("def t86")[0])

def tpex(date):
    y, m, d = date.split("-")
    r = requests.get("https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade",
                     params={"type": "Daily", "sect": "EW", "date": f"{y}/{m}/{d}", "id": "", "response": "json"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30).json()
    df = pd.DataFrame(r["tables"][0]["data"])
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
    return pd.DataFrame({"stock_id": df[0].str.strip(), "buy": num(df[11]), "sell": num(df[12]), "net": num(df[13])}).set_index("stock_id")

for d in ["2019-06-18", "2023-03-15", "2024-09-16", "2026-09-15"]:
    fm = finmind(d); tw = tpex(d)
    assert ((tw.buy - tw.sell) == tw.net).all(), "欄位位置錯"
    j = tw.join(fm, rsuffix="_fm", how="inner"); act = j[(j.buy + j.sell) > 0]
    miss = len(tw[(tw.buy + tw.sell) > 0].index.difference(fm.index))
    print(f"{d} 上櫃: 投信有交易 {len(tw[(tw.buy+tw.sell)>0])} 檔, 對上 {len(act)}, 完全一致 {((act.buy==act.buy_fm)&(act.sell==act.sell_fm)).mean():.1%}, FinMind 缺 {miss}")
    time.sleep(4)
