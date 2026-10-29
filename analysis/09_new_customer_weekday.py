"""신규 고객(첫 구매 기준) 요일별 지표: 신규 고객 수, 첫 주문 금액 합계, 첫 주문당 평균 금액.

실행: python analysis/09_new_customer_weekday.py  (프로젝트 루트 기준, 01 먼저 실행)
신규 고객 정의: 데이터 시작(2009-12-01) 후 12개월(~2010-11-30) 동안 구매 이력이 없다가
2010-12-01 이후 처음 구매한 고객. (데이터 시작 시점의 기존 고객을 신규로 오인하지 않도록 12개월 워밍업 적용)
대상 주문: 해당 고객의 첫 구매 주문(취소 제외), 요일은 첫 주문 시각 기준.
"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _plot_style  # 한글 글꼴·마이너스 기호 설정 (analysis/_plot_style.py)

NEW_FROM = pd.Timestamp("2010-12-01")
df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
p = df[~df.is_cancel]
inv = p.groupby(["customer_id", "invoice"]).agg(date=("invoice_date", "min"), amount=("line_amount", "sum")).reset_index()
first = inv.sort_values(["customer_id", "date", "invoice"]).groupby("customer_id").head(1)
new = first[first.date >= NEW_FROM].copy()
n_all, n_old = first.customer_id.nunique(), (first.date < NEW_FROM).sum()
assert len(new) == new.customer_id.nunique() and n_all == len(new) + n_old
new["wd"] = new.date.dt.weekday
names = ["월", "화", "수", "목", "금", "토", "일"]

# 영업일 수: 같은 기간(2010-12-01~)에 전체 판매가 있었던 날짜 수 (요일별 영업일 차이 보정)
bd = p[p.invoice_date >= NEW_FROM].assign(d=lambda x: x.invoice_date.dt.normalize(), wd=lambda x: x.invoice_date.dt.weekday)
bdays = bd.groupby("wd").d.nunique().reindex(range(7)).fillna(0)
allo = bd.groupby("wd").invoice.nunique().reindex(range(7)).fillna(0)

g = new.groupby("wd").agg(new_customers=("customer_id", "size"), first_sales=("amount", "sum")).reindex(range(7)).fillna(0)
g["first_aov"] = g.first_sales / g.new_customers.replace(0, float("nan"))
g["new_cust_share_%"] = g.new_customers / g.new_customers.sum() * 100
g["first_sales_share_%"] = g.first_sales / g.first_sales.sum() * 100
g["business_days"] = bdays; g["new_cust_per_day"] = g.new_customers / bdays.replace(0, float("nan"))
g["all_orders_share_%"] = allo / allo.sum() * 100                      # 비교용: 같은 기간 전체 주문의 요일 비중
g["new_share_of_orders_%"] = g.new_customers / allo.replace(0, float("nan")) * 100  # 요일별 전체 주문 중 신규 첫 주문 비율
g.index = pd.Index(names, name="weekday")
assert g.new_customers.sum() == len(new) and abs(g.first_sales.sum() - new.amount.sum()) < 0.01
g.round(2).to_csv("outputs/tables/new_customer_weekday.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
spec = [(ax[0, 0], "new_customers", "요일별 신규 고객 수", "new_cust_share_%", "#4f81bd"),
        (ax[0, 1], "first_sales", "요일별 신규 고객 첫 주문 금액 합계", "first_sales_share_%", "#9bbb59"),
        (ax[1, 0], "first_aov", "요일별 첫 주문당 평균 금액", None, "#c0504d"),
        (ax[1, 1], "new_cust_per_day", "요일별 영업일당 신규 고객 수", None, "#8064a2")]
for a, c, t, sh, cl in spec:
    v = g[c].fillna(0); bars = a.bar(names, v, color=cl)
    for i, (b, val) in enumerate(zip(bars, v)):
        lab = "-" if val == 0 else f"{val:,.1f}" if c == "new_cust_per_day" else f"{val:,.0f}"
        if sh and val: lab += f"\n({g[sh].iloc[i]:.1f}%)"
        a.text(b.get_x() + b.get_width() / 2, val, lab, ha="center", va="bottom", fontsize=9)
    a.set_title(t); a.set_ylim(0, max(v.max(), 1) * 1.2); a.yaxis.set_major_formatter(lambda x, _: f"{x:,.0f}"); a.grid(axis="y", alpha=.3)
fig.suptitle(f"신규 고객 {len(new):,}명 (2010-12-01 이후 첫 구매) 요일별 지표", fontsize=13)
plt.tight_layout(); fig.savefig("outputs/figures/new_customer_weekday.png", dpi=130)

pd.set_option("display.width", 230); pd.set_option("display.max_columns", 20)
print(f"구매 고객 {n_all:,}명 | 기존(2010-12-01 이전 첫 구매) {n_old:,}명 | 신규 {len(new):,}명 | 신규 첫 주문 금액 {new.amount.sum():,.2f} | 평균 {new.amount.mean():.2f} 중앙값 {new.amount.median():.2f}")
print(g.round(2).to_string())
