"""第二關-1:把投信每日買賣串成「部位」,檢驗出場是否有停損 / 停利痕跡。

部位(投信全體、單一股票):
  開始:淨買 >= ADV20 5%,且前 20 個交易日沒有任何一天 |淨額| >= ADV20 2%(沉寂後突然買);開始日 20 日均額 >= 1 億
  部位:開始日起累計淨買(股);成本:淨買日以當日均價(成交金額/成交量)加權平均,賣出不改成本
  結束:部位 <= 高峰 20%,或連續 60 個交易日沒有 |淨額| >= ADV20 2% 的動作,或部位 <= 0
範圍:4 碼普通股、排除金融、2019–2025(2026 保留)。

停損檢驗:部位存續中的每一天 T,用 T-1 收盤已知的狀態
  ret_cost = close_{T-1} / cost_{T-1} - 1,dd_peak = close_{T-1} / 開始後最高收盤 - 1,hold = 持有天數
→ T 日結果:big_sell(第一關定義)、sell10(T 日淨賣 >= T-1 部位 10%)。
連續性陷阱:T-1 投信在賣 → T-1 價跌、T 續賣,會假裝成「越跌越賣」→ 另看 T-1 淨額 >= 0 的子樣本。
"""
import numpy as np
import pandas as pd

FIN = {"金融保險業", "金融業", "金融保險"}
START_F, QUIET_F, QUIET_N, IDLE_N, END_FRAC = 0.05, 0.02, 20, 60, 0.20

df = pd.read_parquet("panel.parquet", columns=[
    "date", "stock_id", "close", "Trading_Volume", "Trading_money", "net_it", "adv20", "adm20",
    "industry_category", "big_sell", "big_buy", "year", "net_vol"])
df = df[(df.date <= "2025-12-31") & ~df.industry_category.isin(FIN)].sort_values(["stock_id", "date"])
df["avgpx"] = df.Trading_money / df.Trading_Volume

ep_rows, day_rows = [], []
eid = 0
for sid, g in df.groupby("stock_id", sort=False):
    g = g.reset_index(drop=True)
    net, adv, adm = g.net_it.to_numpy(float), g.adv20.to_numpy(float), g.adm20.to_numpy(float)
    px, close = g.avgpx.to_numpy(float), g.close.to_numpy(float)
    act = np.abs(net) >= QUIET_F * adv                     # 有明顯動作的日子
    n = len(g)
    i = QUIET_N
    while i < n:
        if not (adv[i] > 0 and net[i] >= START_F * adv[i] and adm[i] >= 1e8 and not act[i - QUIET_N:i].any()):
            i += 1
            continue
        eid += 1
        pos, cost, peak_pos, peak_px, last_act = net[i], px[i], net[i], close[i], i
        start, end_reason = i, "data_end"
        j = i
        while True:
            # 記下 j 日收盤後的狀態(給 j+1 日當特徵)
            day_rows.append((eid, sid, j, pos, cost, peak_pos, peak_px))
            j += 1
            if j >= n:
                break
            pos_prev = pos
            pos += net[j]
            if net[j] > 0:
                cost = (cost * pos_prev + px[j] * net[j]) / pos if pos_prev > 0 else px[j]
            peak_pos = max(peak_pos, pos)
            peak_px = max(peak_px, close[j])
            if act[j]:
                last_act = j
            if pos <= 0:
                end_reason = "sold_out"
            elif pos <= END_FRAC * peak_pos:
                end_reason = "sold_80pct"
            elif j - last_act >= IDLE_N:
                end_reason = "idle"
            else:
                continue
            break                                           # 結束日的結果已由前一天的狀態列涵蓋
        ep_rows.append((eid, sid, g.date[start], g.date[min(j, n - 1)], j - start, end_reason,
                        peak_pos / adv[start], close[min(j, n - 1)] / cost - 1, g.industry_category[start]))
        i = j + 1

ep = pd.DataFrame(ep_rows, columns=["eid", "stock_id", "start", "end", "days", "end_reason",
                                    "peak_pos_adv", "ret_end_vs_cost", "industry"])
ep.to_parquet("episodes.parquet")
st = pd.DataFrame(day_rows, columns=["eid", "stock_id", "k", "pos", "cost", "peak_pos", "peak_px"])

# 狀態 (j 日收盤) → 結果 (j+1 日)
key = df.reset_index(drop=True)
key["k"] = key.groupby("stock_id").cumcount()
nxt = key[["stock_id", "k", "date", "net_it", "big_sell", "big_buy", "net_vol", "close", "adv20"]].copy()
nxt["k"] -= 1                                               # 對到前一天的狀態
cur = key[["stock_id", "k", "close", "net_it"]].rename(columns={"close": "close_prev", "net_it": "net_prev"})
h = st.merge(cur, on=["stock_id", "k"]).merge(nxt, on=["stock_id", "k"])
h = h.sort_values(["eid", "k"])
h["ret_cost"] = h.close_prev / h.cost - 1
h["dd_peak"] = h.close_prev / h.peak_px - 1
h["hold"] = h.groupby("eid").cumcount() + 1
h["sell10"] = (-h.net_it) >= 0.10 * h.pos
h["prev_min_ret"] = h.groupby("eid").ret_cost.transform(lambda s: s.cummin().shift(1))
h["new_low"] = h.ret_cost < h.prev_min_ret.fillna(np.inf)  # 距成本創這段部位的新低
h.to_parquet("episode_days.parquet")

print("部位數 %d,股票 %d;部位-日 %d" % (len(ep), ep.stock_id.nunique(), len(h)))
print("持有天數中位 %d(25/75%%: %d / %d);結束原因:%s" % (
    ep.days.median(), ep.days.quantile(.25), ep.days.quantile(.75), ep.end_reason.value_counts().to_dict()))
print("高峰部位 / 開始日均量 中位 %.2f(75%% %.2f)" % (ep.peak_pos_adv.median(), ep.peak_pos_adv.quantile(.75)))

BINS = [-1, -.30, -.25, -.20, -.15, -.10, -.05, 0, .05, .10, .20, .30, .50, 10]
LABS = ["<-30", "-30~-25", "-25~-20", "-20~-15", "-15~-10", "-10~-5", "-5~0", "0~5", "5~10", "10~20", "20~30", "30~50", "≥50"]
h["rc_bin"] = pd.cut(h.ret_cost, BINS, labels=LABS)
h["dd_bin"] = pd.cut(h.dd_peak, [-1, -.30, -.25, -.20, -.15, -.10, -.05, -1e-9, 1e-9],
                     labels=["<-30", "-30~-25", "-25~-20", "-20~-15", "-15~-10", "-10~-5", "-5~0", "在高點"])

def haz(x):
    return pd.Series({"n": len(x), "大賣%": 100 * x.big_sell.mean(), "賣10%部位%": 100 * x.sell10.mean()})

clean = h[h.net_prev >= 0]
for lab, col in [("距成本(%)", "rc_bin"), ("距高點回落(%)", "dd_bin")]:
    a = h.groupby(col, observed=True).apply(haz, include_groups=False)
    b = clean.groupby(col, observed=True).apply(haz, include_groups=False)
    t = pd.concat({"全部": a, "前一天投信沒賣": b}, axis=1).round(2)
    print(f"\n== 隔天賣出機率 vs {lab} ==")
    print(t.to_string())

print("\n== 第一次跌到這個深度(距成本創新低) vs 之前就跌過,前一天投信沒賣 ==")
c = clean[clean.ret_cost < 0]
t = c.groupby(["rc_bin", "new_low"], observed=True).apply(haz, include_groups=False).round(2)
print(t.unstack("new_low").to_string())
