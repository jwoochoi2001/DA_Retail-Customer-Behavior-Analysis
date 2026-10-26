"""요일별 판매 금액·주문 수·주문당 평균 금액 (전체 기간 합산) 집계와 시각화.

실행: python analysis/08_weekday_sales.py  (프로젝트 루트 기준, 01 먼저 실행)
대상: retail_clean.csv 중 구매 행(is_cancel=False). 요일은 InvoiceDate 기준(월~일).
영업일 수 = 해당 요일에 판매가 1건 이상 있었던 날짜 수 -> 영업일당 평균으로 요일별 영업일 수 차이를 보정.
"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
p = df[~df.is_cancel].copy()
p["date"] = p.invoice_date.dt.normalize(); p["wd"] = p.invoice_date.dt.weekday
names = ["월", "화", "수", "목", "금", "토", "일"]

g = p.groupby("wd").agg(sales=("line_amount", "sum"), orders=("invoice", "nunique"), business_days=("date", "nunique"),
                        customers=("customer_id", "nunique")).reindex(range(7)).fillna(0)
g["aov"] = g.sales / g.orders.replace(0, float("nan"))
g["sales_share_%"] = g.sales / g.sales.sum() * 100; g["orders_share_%"] = g.orders / g.orders.sum() * 100
g["sales_per_day"] = g.sales / g.business_days.replace(0, float("nan"))
g["orders_per_day"] = g.orders / g.business_days.replace(0, float("nan"))
g.index = pd.Index(names, name="weekday")
assert abs(g.sales.sum() - p.line_amount.sum()) < 0.01 and g.orders.sum() == p.invoice.nunique()  # 주문은 하루에 걸치지 않음
tot = g.loc[["월","화","수","목","금","토","일"]]
g.round(2).to_csv("outputs/tables/weekday_sales.csv", encoding="utf-8-sig")

col = ["#4f81bd"] * 7
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
spec = [(ax[0, 0], "sales", "요일별 판매 금액 (전체 기간 합계)", "sales_share_%", "{:,.0f}"),
        (ax[0, 1], "orders", "요일별 주문 수 (전체 기간 합계)", "orders_share_%", "{:,.0f}"),
        (ax[1, 0], "aov", "요일별 주문당 평균 금액", None, "{:,.0f}"),
        (ax[1, 1], "sales_per_day", "요일별 영업일당 평균 판매 금액", None, "{:,.0f}")]
colors = ["#4f81bd", "#9bbb59", "#c0504d", "#8064a2"]
for (a, c, t, sh, f), cl in zip(spec, colors):
    v = g[c].fillna(0); bars = a.bar(names, v, color=cl)
    for b, val, i in zip(bars, v, range(7)):
        lab = "판매 없음" if val == 0 else f.format(val) + (f"\n({g[sh].iloc[i]:.1f}%)" if sh else "")
        a.text(b.get_x() + b.get_width() / 2, val, lab, ha="center", va="bottom", fontsize=9)
    a.set_title(t); a.set_ylim(0, v.max() * 1.18); a.yaxis.set_major_formatter(lambda x, _: f"{x:,.0f}"); a.grid(axis="y", alpha=.3)
plt.tight_layout(); fig.savefig("outputs/figures/weekday_sales.png", dpi=130)

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print(g.round(2).to_string())
print("기간:", p.date.min().date(), "~", p.date.max().date(), "| 전체 영업일:", p.date.nunique())
