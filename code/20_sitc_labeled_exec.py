"""第四關-1:有身分標記的資料(7 檔 × 2024-01~02),投信(M)重押日怎麼下單。
重押日:M 淨額 >= 當日量 5%(買 13 天)或 <= −5%(賣 37 天);只看 M 在「當天方向」的成交。
(1) 時段分布 vs 全市場 (2) 主動吃單 vs 被動掛單(成交對手兩張委託誰後進場,後進場者 = 主動方;集合競價另計)
(3) 委託大小與間隔。只描述、找候選特徵,不驗證。不使用 IOC 相關特徵。
"""
import numpy as np
import pandas as pd

P = "/Users/chih/quant/data/raw/otc"
t = pd.read_parquet(f"{P}/trade_detail.parquet", columns=[
    "trade_date", "stock_id", "side", "trade_time", "trade_seq", "trade_shares", "trade_price", "investor_type",
    "link_code1", "link_code2"])
ob = pd.read_parquet(f"{P}/order_book.parquet", columns=[
    "order_date", "stock_id", "side", "order_time", "link_code1", "link_code2", "modified_trade_type",
    "modified_shares", "order_price", "investor_type"])

# 重押日
buy = t[t.side == "B"]
vol = buy.groupby(["stock_id", "trade_date"]).trade_shares.sum().rename("vol")
mm = t[t.investor_type == "M"].assign(s=lambda x: x.trade_shares * x.side.map({"B": 1, "S": -1}))
mnet = mm.groupby(["stock_id", "trade_date"]).s.sum().rename("mnet")
days = pd.concat([vol, mnet], axis=1).dropna()
days["r"] = days.mnet / days.vol
days["dir"] = np.select([days.r >= .05, days.r <= -.05], ["B", "S"], "")
days = days[days.dir != ""].reset_index()
print("重押日:買 %d、賣 %d" % ((days.dir == "B").sum(), (days.dir == "S").sum()))

# (1) 時段
def bucket(tt):
    h = tt.str[:5]
    return np.select([h < "09:01", h < "09:30", h < "10:00", h < "11:00", h < "12:00", h < "13:25", h >= "13:30"],
                     ["開盤競價", "09:00-09:30", "09:30-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:25", "收盤競價"], "13:25-13:30")
t["bk"] = bucket(t.trade_time)
x = t.merge(days[["stock_id", "trade_date", "dir"]], on=["stock_id", "trade_date"])
m_dir = x[(x.investor_type == "M") & (x.side == x.dir)]
allv = x[x.side == "B"]
order = ["開盤競價", "09:00-09:30", "09:30-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:25", "13:25-13:30", "收盤競價"]
prof = pd.DataFrame({"投信(當天方向)%": 100 * m_dir.groupby("bk").trade_shares.sum() / m_dir.trade_shares.sum(),
                     "全市場%": 100 * allv.groupby("bk").trade_shares.sum() / allv.trade_shares.sum()}).reindex(order)
prof["投信累計%"] = prof["投信(當天方向)%"].cumsum(); prof["全市場累計%"] = prof["全市場%"].cumsum()
print("\n== (1) 重押日成交量時段分布 ==")
print(prof.round(1).to_string())
# 每一天各自到 10:00 / 11:00 完成幾成
dd = m_dir.assign(by10=m_dir.trade_time < "10:00", by11=m_dir.trade_time < "11:00")
per = dd.groupby(["stock_id", "trade_date"]).apply(lambda g: pd.Series({
    "到10點%": 100 * g.trade_shares[g.by10].sum() / g.trade_shares.sum(),
    "到11點%": 100 * g.trade_shares[g.by11].sum() / g.trade_shares.sum()}), include_groups=False)
print("每天各自:到 10 點完成 中位 %.0f%%(25/75: %.0f/%.0f)、到 11 點 中位 %.0f%%" % (
    per["到10點%"].median(), per["到10點%"].quantile(.25), per["到10點%"].quantile(.75), per["到11點%"].median()))

# (2) 主動 vs 被動:委託進場時間 = 該委託第一筆事件時間
key = ["stock_id", "link_code1", "link_code2"]
ob["d"] = ob.order_date
first = ob.groupby(key + ["d"]).order_time.min().rename("entry").reset_index().rename(columns={"d": "trade_date"})
tc = x[x.bk.isin(order[1:7])].merge(first, on=key + ["trade_date"], how="left")
pair = tc.pivot_table(index=["stock_id", "trade_date", "trade_seq"], columns="side", values="entry", aggfunc="first")
tc = tc.merge(pair.rename(columns={"B": "eB", "S": "eS"}).reset_index(), on=["stock_id", "trade_date", "trade_seq"])
tc["taker"] = np.where(tc.eB > tc.eS, "B", np.where(tc.eS > tc.eB, "S", "?"))
tc["aggr"] = tc.taker == tc.side
print("\n== (2) 連續交易時段,成交量中『主動吃單』的比例(依身分,重押日) ==")
for iv in ["M", "F", "I", "J"]:
    z = tc[(tc.investor_type == iv) & tc.eB.notna() & tc.eS.notna()]
    zz = z[z.side == z.dir] if iv == "M" else z
    print("  %s%s:主動 %.0f%%(成交量加權),n 成交 %d" % (
        iv, "(當天方向)" if iv == "M" else "", 100 * (zz.trade_shares * zz.aggr).sum() / zz.trade_shares.sum(), len(zz)))

# (3) 委託:新單大小、每天筆數、間隔
mo = ob[(ob.investor_type == "M") & ob.modified_trade_type.isin(["1", "4"])].copy()
mo["side_dir"] = np.where(mo.modified_trade_type == "1", "B", "S")
mo = mo.merge(days[["stock_id", "trade_date", "dir"]].rename(columns={"trade_date": "order_date"}), on=["stock_id", "order_date"])
mo = mo[mo.side_dir == mo.dir].sort_values(["stock_id", "order_date", "order_time"])
print("\n== (3) 投信重押日,當天方向的新委託 ==")
print("每天委託筆數 中位 %d(25/75: %d/%d)" % tuple(mo.groupby(["stock_id", "order_date"]).size().quantile([.5, .25, .75])))
sz = mo.modified_shares / 1000
print("每筆張數分位:", sz.quantile([.1, .25, .5, .75, .9]).round(0).to_dict())
print("最常見張數:", sz.round(0).value_counts().head(8).to_dict())
tm = pd.to_timedelta(mo.order_time)
gap = tm.groupby([mo.stock_id, mo.order_date]).diff().dt.total_seconds().dropna()
print("相鄰兩筆委託間隔(秒)分位:", gap.quantile([.1, .25, .5, .75, .9]).round(0).to_dict())
