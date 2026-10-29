"""2011-09-19 주(기준 주) vs 직전 주(09-12 주) 판매 금액 증가 원인 - KPI 트리와 기여도(LMDI) 분해.

실행: python analysis/12_kpi_tree_0919.py  (프로젝트 루트 기준, 01 먼저 실행)
판매 금액 = 구매 고객 수 x 고객당 주문 수 x 주문당 수량 x 평균 단가(금액/수량). 취소 주문 제외.
기여도: LMDI(로그평균 분해) - 항목별 기여액의 합이 판매 금액 증가액과 정확히 일치하고 순서에 무관.
드릴다운: 신규/기존 고객, 주문 금액 구간, 상품별 증감.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import _plot_style  # 한글 글꼴·마이너스 기호 설정 (analysis/_plot_style.py)

PREV, BASE = pd.Timestamp("2011-09-12"), pd.Timestamp("2011-09-19")
df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
p = df[~df.is_cancel].copy()
wk = lambda s: p[(p.invoice_date >= s) & (p.invoice_date < s + pd.Timedelta(days=7))]
w0, w1 = wk(PREV), wk(BASE)


def kpis(w):
    S, Q, O, C = w.line_amount.sum(), w.quantity.sum(), w.invoice.nunique(), w.customer_id.nunique()
    return dict(sales=S, customers=C, orders=O, qty=Q, opc=O / C, aov=S / O, qpo=Q / O, price=S / Q)


k0, k1 = kpis(w0), kpis(w1)
dS = k1["sales"] - k0["sales"]
L = dS / np.log(k1["sales"] / k0["sales"])
c = {key: L * np.log(k1[key] / k0[key]) for key in ["customers", "opc", "qpo", "price"]}
c["orders"] = c["customers"] + c["opc"]
c["aov"] = c["qpo"] + c["price"]
assert abs(c["orders"] + c["aov"] - dS) < 1e-6 and abs(c["customers"] + c["opc"] + c["qpo"] + c["price"] - dS) < 1e-6

names = {"sales": "판매 금액", "orders": "주문 수", "aov": "주문당 평균 금액", "customers": "구매 고객 수",
         "opc": "고객당 주문 수", "qpo": "주문당 수량", "price": "평균 단가"}
tree = pd.DataFrame([{"kpi": names[k], "prev": k0[k], "base": k1[k], "diff": k1[k] - k0[k],
                      "change_%": (k1[k] / k0[k] - 1) * 100,
                      "contribution": dS if k == "sales" else c[k],
                      "contribution_%": 100 if k == "sales" else c[k] / dS * 100} for k in names]).round(3)
tree.to_csv("outputs/tables/kpi_tree_0919.csv", index=False, encoding="utf-8-sig")

# ---- 드릴다운 ----
first = p.groupby("customer_id").invoice_date.min()


def split(w, s):
    new = w.customer_id.map(first) >= s
    return pd.Series({"신규 고객 수": w[new].customer_id.nunique(), "기존 고객 수": w[~new].customer_id.nunique(),
                      "신규 판매": w[new].line_amount.sum(), "기존 판매": w[~new].line_amount.sum(),
                      "신규 주문 수": w[new].invoice.nunique(), "기존 주문 수": w[~new].invoice.nunique()})


ct = pd.DataFrame({"prev": split(w0, PREV), "base": split(w1, BASE)})
ct["diff"] = ct.base - ct.prev
ct.round(2).to_csv("outputs/tables/kpi_drill_new_vs_existing_0919.csv", encoding="utf-8-sig")


def orders(w):
    return w.groupby(["invoice", "customer_id"]).agg(amount=("line_amount", "sum"), qty=("quantity", "sum")).reset_index()


o0, o1 = orders(w0), orders(w1)
bands = [0, 500, 1000, 5000, np.inf]
bl = ["<500", "500~1,000", "1,000~5,000", "5,000+"]


def band(o):
    return (o.assign(b=pd.cut(o.amount, bands, labels=bl, right=False))
            .groupby("b", observed=False).agg(orders=("amount", "size"), sales=("amount", "sum")))


bd = band(o0).join(band(o1), lsuffix="_prev", rsuffix="_base")
bd["sales_diff"] = bd.sales_base - bd.sales_prev
bd["orders_diff"] = bd.orders_base - bd.orders_prev
bd.round(2).to_csv("outputs/tables/kpi_drill_order_size_0919.csv", encoding="utf-8-sig")
top_orders = o1.nlargest(5, "amount")

pr = pd.DataFrame({"prev": w0.groupby("stock_code").line_amount.sum(),
                   "base": w1.groupby("stock_code").line_amount.sum()}).fillna(0)
pr["description"] = p.groupby("stock_code").description.agg(lambda x: x.mode().iat[0])
pr["diff"] = pr.base - pr.prev
pr = pr.sort_values("diff", ascending=False)
pr.round(2).to_csv("outputs/tables/kpi_drill_products_0919.csv", encoding="utf-8-sig")
assert abs(pr["diff"].sum() - dS) < 1e-6 and abs(bd.sales_diff.sum() - dS) < 1e-6
assert abs(ct.loc[["신규 판매", "기존 판매"], "diff"].sum() - dS) < 1e-6

# ---- 그림 1: KPI 트리 ----
fig, ax = plt.subplots(figsize=(16, 8.5))
ax.set_xlim(0, 16); ax.set_ylim(0, 9.1); ax.axis("off")
pos = {"sales": (1.9, 4.25), "orders": (6.4, 6.4), "aov": (6.4, 2.1), "customers": (11.6, 7.4),
       "opc": (11.6, 5.4), "qpo": (11.6, 3.1), "price": (11.6, 1.1)}


def fmt(key, v):
    return {"sales": f"{v:,.0f}", "orders": f"{v:,.0f}", "customers": f"{v:,.0f}", "opc": f"{v:.3f}",
            "aov": f"{v:,.2f}", "qpo": f"{v:,.2f}", "price": f"{v:.3f}"}[key]


def node(key):
    x, y = pos[key]
    ch = (k1[key] / k0[key] - 1) * 100
    cn = dS if key == "sales" else c[key]
    face = "#fde9d9" if key == "sales" else ("#dce6f1" if key in ("orders", "aov") else "#ebf1de")
    ax.add_patch(FancyBboxPatch((x - 1.75, y - .8), 3.5, 1.6, boxstyle="round,pad=0.03", fc=face, ec="#555", lw=1.3))
    ax.text(x, y + .52, names[key], ha="center", fontsize=12, fontweight="bold")
    ax.text(x, y + .12, f"{fmt(key, k0[key])} → {fmt(key, k1[key])}", ha="center", fontsize=10.5)
    ax.text(x, y - .22, f"{ch:+.1f}%", ha="center", fontsize=10.5, color="#c0392b" if ch > 0 else "#1f4e79")
    ax.text(x, y - .55, f"증가액 {dS:+,.0f} (100%)" if key == "sales" else f"증가 기여 {cn:+,.0f} ({cn / dS * 100:.0f}%)",
            ha="center", fontsize=10, fontweight="bold")


def link(a, b):
    (x0, y0), (x1, y1) = pos[a], pos[b]
    ax.plot([x0 + 1.78, (x0 + x1) / 2, (x0 + x1) / 2, x1 - 1.78], [y0, y0, y1, y1], color="#888", lw=1.3)


for a, b in [("sales", "orders"), ("sales", "aov"), ("orders", "customers"), ("orders", "opc"), ("aov", "qpo"), ("aov", "price")]:
    link(a, b)
for k in pos:
    node(k)
ax.text(8, 8.85, "판매 금액 KPI 트리: 직전 주(09-12~09-18) → 기준 주(09-19~09-25)   |   판매 금액 = 고객 수 × 고객당 주문 수 × 주문당 수량 × 평균 단가",
        ha="center", fontsize=12)
ax.text(8, .15, "기여도는 LMDI 분해(항목 합계 = 증가액). 주문 수 = 고객 수 × 고객당 주문 수, 주문당 평균 금액 = 주문당 수량 × 평균 단가",
        ha="center", fontsize=9, color="#555")
plt.tight_layout()
fig.savefig("outputs/figures/kpi_tree_0919.png", dpi=130)

# ---- 그림 2: 기여도·드릴다운 ----
fig, ax = plt.subplots(1, 3, figsize=(18, 5.8))
a = ax[0]
lab = ["구매 고객 수", "고객당 주문 수", "주문당 수량", "평균 단가"]
val = [c["customers"], c["opc"], c["qpo"], c["price"]]
cum = np.cumsum([0] + val[:-1])
a.bar(lab, val, bottom=cum, color=["#4f81bd" if v >= 0 else "#c0504d" for v in val])
a.bar(["합계"], [dS], color="#7f7f7f")
for i, v in enumerate(val):
    a.text(i, cum[i] + v, f"{v:+,.0f}\n({v / dS * 100:.0f}%)", ha="center", va="bottom", fontsize=9)
a.text(4, dS, f"{dS:+,.0f}", ha="center", va="bottom", fontsize=9)
a.set_title("판매 금액 증가 기여도 (워터폴, LMDI)"); a.grid(axis="y", alpha=.3)
a.tick_params(axis="x", labelsize=9); a.set_ylim(0, dS * 1.2)
a = ax[1]
x = np.arange(len(bl))
a.bar(x - .2, bd.sales_prev, .4, label="직전 주", color="#bfbfbf")
a.bar(x + .2, bd.sales_base, .4, label="기준 주", color="#4f81bd")
for i in range(len(bl)):
    a.text(i + .2, bd.sales_base.iloc[i], f"{bd.sales_diff.iloc[i]:+,.0f}", ha="center", va="bottom", fontsize=9)
a.set_xticks(x, [f"{b}\n(주문 {int(p_)}→{int(b_)})" for b, p_, b_ in zip(bl, bd.orders_prev, bd.orders_base)], fontsize=9)
a.set_title("주문 금액 구간별 판매 금액 (막대 위: 증감액)"); a.legend(); a.grid(axis="y", alpha=.3)
a.yaxis.set_major_formatter(lambda y, _: f"{y:,.0f}")
a = ax[2]
tp = pd.concat([pr.head(6), pr.tail(4)])
a.barh([d[:26] for d in tp.description], tp["diff"], color=["#4f81bd" if v >= 0 else "#c0504d" for v in tp["diff"]])
a.invert_yaxis(); a.set_title("상품별 판매 금액 증감 (상위 6 증가 / 하위 4 감소)"); a.grid(axis="x", alpha=.3)
a.tick_params(axis="y", labelsize=8)
for i, v in enumerate(tp["diff"]):
    a.text(v, i, f" {v:+,.0f}", va="center", fontsize=8)
plt.tight_layout()
fig.savefig("outputs/figures/kpi_drilldown_0919.png", dpi=130)

pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30)
print(tree.to_string(index=False)); print(ct.round(2).to_string()); print(bd.round(2).to_string())
print("기준 주 상위 5 주문:\n", top_orders.round(2).to_string(index=False))
print("직전 주 최대 주문:", round(o0.amount.max(), 2), "| 5,000+ 주문 합계 직전/기준:",
      round(o0[o0.amount >= 5000].amount.sum()), round(o1[o1.amount >= 5000].amount.sum()))
print(pr.head(8).round(0).to_string()); print(pr.tail(5).round(0).to_string())
e0, e1 = o0[o0.amount < 5000], o1[o1.amount < 5000]
print("5,000 미만 주문만: 주문 수", len(e0), "->", len(e1), "평균", round(e0.amount.mean(), 1), "->", round(e1.amount.mean(), 1),
      "합계", round(e0.amount.sum()), "->", round(e1.amount.sum()))
print("주문 금액 중앙값:", o0.amount.median(), o1.amount.median())
