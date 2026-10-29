"""이탈 위험 군집(R 평균이 가장 큰 군집)의 주문 금액 분포 분석.

실행: python analysis/03_churnrisk_order_amount.py  (프로젝트 루트 기준, 02 먼저 실행)
주문 금액 = 주문(invoice) 단위 line_amount 합계, 기간·구매 기준은 02와 동일.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _plot_style  # 한글 글꼴·마이너스 기호 설정 (analysis/_plot_style.py)

REF = pd.Timestamp("2011-09-01"); START = REF - pd.Timedelta(days=180)
df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
rfm = pd.read_csv("outputs/tables/rfm_customers.csv")
risk = rfm.groupby("cluster")["R"].mean().idxmax()
names = {c: ("이탈 위험" if c == risk else f"군집 {c}") for c in rfm.cluster.unique()}

w = df[(df.invoice_date >= START) & (df.invoice_date < REF) & (~df.is_cancel)]
orders = (w.groupby(["customer_id", "invoice"]).agg(amount=("line_amount", "sum"), items=("quantity", "sum"))
          .reset_index().merge(rfm[["customer_id", "cluster"]], on="customer_id"))
assert abs(orders.amount.sum() - w.line_amount.sum()) < 1 and orders.invoice.nunique() == w.invoice.nunique()
orders["group"] = orders.cluster.map(names)
r = orders[orders.cluster == risk]

bins = [0, 100, 250, 500, 1000, np.inf]; labels = ["<100", "100~250", "250~500", "500~1,000", "1,000+"]
r = r.assign(band=pd.cut(r.amount, bins, labels=labels, right=False))
band = r.groupby("band", observed=True).agg(orders=("amount", "size"), amount=("amount", "sum")).reindex(labels)
band["order_share_%"] = band.orders / band.orders.sum() * 100; band["amount_share_%"] = band.amount / band.amount.sum() * 100
stat = r.amount.describe(percentiles=[.1, .25, .5, .75, .9, .99]).round(2)
cmp_ = orders.groupby("group").amount.agg(orders="size", mean="mean", median="median", p90=lambda s: s.quantile(.9), max="max").round(2)
cust = r.groupby("customer_id").agg(n=("invoice", "size"), mean=("amount", "mean"))

fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
a = ax[0]; lo = np.logspace(np.log10(max(r.amount.min(), 1)), np.log10(r.amount.max()), 40)
a.hist(r.amount, bins=lo, color="#c0504d", alpha=.85, edgecolor="white"); a.set_xscale("log")
a.axvline(r.amount.median(), color="k", ls="--", label=f"중앙값 {r.amount.median():,.0f}")
a.axvline(r.amount.mean(), color="navy", ls=":", label=f"평균 {r.amount.mean():,.0f}")
a.set(title=f"이탈 위험 고객 주문 금액 분포 (주문 {len(r):,}건, 로그 축)", xlabel="주문 금액 (로그)", ylabel="주문 수"); a.legend()
b = ax[1]; order = [names[c] for c in orders.groupby("cluster").amount.median().sort_values().index]
b.boxplot([orders[orders.group == g].amount for g in order], tick_labels=order, showfliers=False, patch_artist=True,
          boxprops=dict(facecolor="#e6b8b7"), medianprops=dict(color="k"))
b.set_yscale("log"); b.set(title="군집별 주문 금액 (이상치 제외, 로그 축)", ylabel="주문 금액 (로그)")
c = ax[2]; x = np.arange(len(labels)); wd = .38
c.bar(x - wd/2, band["order_share_%"], wd, label="주문 비중", color="#c0504d")
c.bar(x + wd/2, band["amount_share_%"], wd, label="금액 비중", color="#4f81bd")
for i, (o, m) in enumerate(zip(band["order_share_%"], band["amount_share_%"])):
    c.text(i - wd/2, o + .5, f"{o:.0f}%", ha="center", fontsize=9); c.text(i + wd/2, m + .5, f"{m:.0f}%", ha="center", fontsize=9)
c.set_xticks(x, labels); c.set(title="주문 금액 구간별 비중 (이탈 위험)", xlabel="주문 금액 구간", ylabel="%"); c.legend()
plt.tight_layout(); fig.savefig("outputs/figures/churnrisk_order_amount.png", dpi=130)

band.round(2).to_csv("outputs/tables/churnrisk_order_amount_bands.csv", encoding="utf-8-sig")
cmp_.to_csv("outputs/tables/order_amount_by_cluster.csv", encoding="utf-8-sig")
pd.set_option("display.width", 200)
print("이탈 위험 군집 번호:", risk, "| 고객", r.customer_id.nunique(), "| 주문", len(r), "| 금액", round(r.amount.sum(), 2))
print(stat.to_string()); print(band.round(2).to_string()); print(cmp_.to_string())
print("음수/0 주문:", (r.amount <= 0).sum(), "| 평균 주문 수/고객", round(cust.n.mean(), 2),
      "| 1회 구매 고객 비중 %", round((cust.n == 1).mean() * 100, 1))
print("왜도:", round(r.amount.skew(), 2), "| 상위10% 주문의 금액 비중 %", round(r.amount[r.amount >= r.amount.quantile(.9)].sum() / r.amount.sum() * 100, 1))
