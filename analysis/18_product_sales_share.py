"""상품별 판매 금액 비중, 상위 10개 상품, 파레토(누적 비중).

실행: python analysis/18_product_sales_share.py  (프로젝트 루트 기준, 01 먼저 실행)
- 대상: 전처리 데이터(retail_clean.csv) 중 구매 행(is_cancel=False). 판매 금액 = 수량 x 단가 합계.
- 상품 = stock_code. 상품명은 코드별 가장 많이 쓰인 Description.
- 참고: 취소·반품을 차감한 순판매 기준 상위 10개도 함께 계산해 순위가 달라지는지 확인.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
buy = df[~df.is_cancel]
name = df.groupby("stock_code").description.agg(lambda s: s.mode().iat[0])

g = buy.groupby("stock_code").agg(sales=("line_amount", "sum"), quantity=("quantity", "sum"), orders=("invoice", "nunique"),
                                  customers=("customer_id", "nunique")).sort_values("sales", ascending=False)
total = g.sales.sum()
assert abs(total - buy.line_amount.sum()) < 0.01 and g.index.is_unique
g["description"] = name.reindex(g.index)
g["share_%"] = g.sales / total * 100
g["cum_share_%"] = g["share_%"].cumsum()
g["rank"] = np.arange(1, len(g) + 1)
g["avg_price"] = g.sales / g.quantity
g = g[["rank", "description", "sales", "share_%", "cum_share_%", "quantity", "avg_price", "orders", "customers"]]
g.round(3).to_csv("outputs/tables/product_sales_share.csv", encoding="utf-8-sig")
top = g.head(10)

# 순판매(취소 반영) 기준 비교
net = df.groupby("stock_code").line_amount.sum().sort_values(ascending=False)
net_top = net.head(10).index.tolist(); same = len(set(net_top) & set(top.index))

n = len(g)
conc = {f"상위 {p}%": g["share_%"].head(max(1, int(np.ceil(n * p / 100)))).sum() for p in (1, 5, 10, 20)}
n80 = int((g["cum_share_%"] < 80).sum() + 1)

fig, ax = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})
a = ax[0]; y = np.arange(10)[::-1]
b = a.barh(y, top.sales, color="#4f81bd")
for yy, s, sh, c in zip(y, top.sales, top["share_%"], top["cum_share_%"]):
    a.text(s, yy, f"  {s:,.0f}  ({sh:.2f}%)", va="center", fontsize=9)
a.set_yticks(y, [f"{i}. {d[:30]}" for i, d in zip(top["rank"], top.description)], fontsize=9)
a.set_xlim(0, top.sales.max() * 1.28); a.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
a.set_xlabel("판매 금액"); a.grid(axis="x", alpha=.3)
a.set_title(f"판매 금액 상위 10개 상품 (합계 {top.sales.sum():,.0f} = 전체의 {top['share_%'].sum():.2f}%)")
a = ax[1]; x = np.arange(1, n + 1) / n * 100
a.plot(x, g["cum_share_%"], color="#c0504d", lw=2)
for p, mk in ((1, "o"), (5, "s"), (10, "^"), (20, "D")):
    k = max(1, int(np.ceil(n * p / 100))); v = g["cum_share_%"].iloc[k - 1]
    a.plot(p, v, mk, color="#222"); a.annotate(f"상위 {p}% = {v:.1f}%", (p, v), textcoords="offset points", xytext=(8, -12), fontsize=9)
a.axhline(80, color="gray", ls=":", lw=1); a.axvline(n80 / n * 100, color="gray", ls=":", lw=1)
a.text(n80 / n * 100 + 1, 5, f"80% 도달: 상위 {n80:,}개\n({n80 / n * 100:.1f}%)", fontsize=9)
a.set_xlabel(f"상품 순위 (상위 %, 전체 {n:,}개 상품)"); a.set_ylabel("누적 판매 금액 비중 (%)"); a.set_title("파레토: 판매 금액 누적 비중"); a.grid(alpha=.3); a.set_ylim(0, 102)
plt.tight_layout(); fig.savefig("outputs/figures/product_top10_sales_share.png", dpi=130)

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print("상품 수", n, "| 전체 판매 금액", round(total, 2), "| 상위 10개 합", round(top.sales.sum(), 2), "비중 %", round(top["share_%"].sum(), 3))
print(top.round(2).to_string())
print({k: round(v, 1) for k, v in conc.items()}, "| 80% 도달 상품 수", n80)
print("순판매 기준 상위10과 겹치는 상품 수:", same, "| 순판매 상위10 비중 %:", round(net.head(10).sum() / net.sum() * 100, 2))
print("판매 금액 1위 상품 평균 단가", round(top.avg_price.iloc[0], 2), "| 상위10 평균 단가", round(top.sales.sum() / top.quantity.sum(), 2), "전체", round(total / g.quantity.sum(), 2))
print("상위10 주문 포함 비율 % (전체 주문 대비):", [round(o / buy.invoice.nunique() * 100, 1) for o in top.orders.head(10)])
