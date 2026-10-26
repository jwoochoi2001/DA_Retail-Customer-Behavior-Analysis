"""고객별 주문 구성 지표(상품 종류 수, 상품 종류당 수량)와 총 주문 금액 기반 K-Means 군집.

실행: python analysis/20_order_basket_clustering.py  (프로젝트 루트 기준, 01·02 먼저 실행)
대상: 02와 동일(2011-03-05~2011-08-31 구매 고객, 취소 제외).
파생 컬럼
  avg_products_per_order : 주문 1건에 담은 서로 다른 상품(stock_code) 수의 고객별 평균
  qty_per_product        : 상품 종류 1개당 구매 수량 = 총 수량 / 주문 내 상품 행 수(주문x상품)
  total_amount           : 총 주문 금액(M)
  (참고) total_distinct_products: 기간 내 구매한 서로 다른 상품 수, orders, total_qty
군집: log1p -> 표준화한 (avg_products_per_order, qty_per_product, total_amount), k=2~8 실루엣, k>=3 중 최대 선택.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False

REF = pd.Timestamp("2011-09-01"); START = REF - pd.Timedelta(days=180)
df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
w = df[(df.invoice_date >= START) & (df.invoice_date < REF) & (~df.is_cancel)]
lines = w.groupby(["customer_id", "invoice", "stock_code"]).quantity.sum().reset_index()      # 주문x상품 행
per_order = lines.groupby(["customer_id", "invoice"]).agg(n_prod=("stock_code", "nunique"), qty=("quantity", "sum")).reset_index()
f = per_order.groupby("customer_id").agg(orders=("invoice", "nunique"), avg_products_per_order=("n_prod", "mean"), total_qty=("qty", "sum"))
f["lines"] = per_order.groupby("customer_id").n_prod.sum()
f["qty_per_product"] = f.total_qty / f.lines
f["total_amount"] = w.groupby("customer_id").line_amount.sum()
f["total_distinct_products"] = lines.groupby("customer_id").stock_code.nunique()
f = f.reset_index()
assert abs(f.total_amount.sum() - w.line_amount.sum()) < 0.01 and f.orders.sum() == w.invoice.nunique()
assert (f.avg_products_per_order >= 1).all() and (f.qty_per_product > 0).all()

rfm = pd.read_csv("outputs/tables/rfm_customers.csv")[["customer_id", "R", "F", "cluster"]]
f = f.merge(rfm, on="customer_id", validate="1:1")
f = f.merge(pd.read_csv("outputs/tables/repurchase_customers.csv")[["customer_id", "rp30", "rp90"]], on="customer_id", validate="1:1")  # 15번 결과(기준일 이후 재구매, 평가용)
f["aov"] = f.total_amount / f.orders
f.drop(columns=["lines"]).round(3).to_csv("data/processed/customer_order_features.csv", index=False, encoding="utf-8-sig")

FEATS = ["avg_products_per_order", "qty_per_product", "total_amount"]
X = StandardScaler().fit_transform(np.log1p(f[FEATS]))
rows, labs = [], {}
for k in range(2, 9):
    km = KMeans(k, n_init=20, random_state=42).fit(X); labs[k] = km.labels_
    other = [KMeans(k, n_init=20, random_state=s).fit_predict(X) for s in (1, 2, 3)]
    rows.append(dict(k=k, silhouette=silhouette_score(X, km.labels_), inertia=km.inertia_,
                     min_group_pct=pd.Series(km.labels_).value_counts(normalize=True).min() * 100,
                     stability_ARI=np.mean([adjusted_rand_score(km.labels_, o) for o in other])))
ev = pd.DataFrame(rows); ev.round(4).to_csv("outputs/tables/basket_kmeans_k_evaluation.csv", index=False, encoding="utf-8-sig")
k = int(ev[ev.k >= 3].set_index("k").silhouette.idxmax())
f["basket_cluster"] = labs[k]

g = f.groupby("basket_cluster")
prof = g.agg(customers=("customer_id", "size"), products_per_order_median=("avg_products_per_order", "median"),
             qty_per_product_median=("qty_per_product", "median"), total_amount_median=("total_amount", "median"),
             total_amount_sum=("total_amount", "sum"), orders_median=("orders", "median"), aov_median=("aov", "median"),
             R_median=("R", "median"), distinct_products_median=("total_distinct_products", "median"),
             total_qty_median=("total_qty", "median"), repurchase_30d=("rp30", "mean"), repurchase_90d=("rp90", "mean"))
prof[["repurchase_30d", "repurchase_90d"]] *= 100
prof["customer_share_%"] = prof.customers / len(f) * 100; prof["amount_share_%"] = prof.total_amount_sum / f.total_amount.sum() * 100
prof = prof.sort_values("total_amount_median", ascending=False)
# 군집 이름: 총 금액 순서대로 번호만 붙이고, 해석은 아래 프로파일 기준으로 별도 부여
prof.round(2).to_csv("outputs/tables/basket_cluster_profile.csv", encoding="utf-8-sig")
f.drop(columns=["lines"]).round(3).to_csv("outputs/tables/basket_cluster_customers.csv", index=False, encoding="utf-8-sig")
ct = pd.crosstab(f.basket_cluster, f.cluster).reindex(prof.index)
ct.to_csv("outputs/tables/basket_vs_rfm_crosstab.csv", encoding="utf-8-sig")
corr = np.log1p(f[FEATS]).corr(method="spearman")

order = prof.index.tolist(); pal = dict(zip(order, ["#d94801", "#4f81bd", "#9bbb59", "#8064a2", "#f2c14e", "#7f7f7f"]))
fig, ax = plt.subplots(2, 2, figsize=(16, 11.5))
a = ax[0, 0]; a.plot(ev.k, ev.silhouette, "o-"); a.axvline(k, color="#c0504d", ls="--", label=f"선택 k={k}")
for kk, v in zip(ev.k, ev.silhouette): a.text(kk, v + .004, f"{v:.3f}", ha="center", fontsize=8)
a.set_title("실루엣 점수 (k=2~8)"); a.set_xlabel("k"); a.legend(); a.grid(alpha=.3)
a = ax[0, 1]
for c in order:
    d = f[f.basket_cluster == c]; a.scatter(d.avg_products_per_order, d.qty_per_product, s=np.clip(d.total_amount / 80, 6, 260),
                                            alpha=.55, color=pal[c], label=f"군집 {c} (n={len(d)})", edgecolor="white", linewidth=.3)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("주문당 평균 상품 종류 수 (로그)"); a.set_ylabel("상품 종류당 구매 수량 (로그)")
a.set_title("상품 종류 수 × 종류당 수량 (점 크기 = 총 주문 금액)"); a.legend(fontsize=8); a.grid(alpha=.3)
a = ax[1, 0]; z = (np.log1p(prof[["products_per_order_median", "qty_per_product_median", "total_amount_median"]]).copy())
z.columns = ["주문당 상품 종류 수", "종류당 수량", "총 주문 금액"]
zs = (z - np.log1p(f[["avg_products_per_order", "qty_per_product", "total_amount"]]).median().values) / np.log1p(f[["avg_products_per_order", "qty_per_product", "total_amount"]]).std().values
im = a.imshow(zs.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
for i in range(zs.shape[0]):
    raw = [prof.products_per_order_median.iloc[i], prof.qty_per_product_median.iloc[i], prof.total_amount_median.iloc[i]]
    for j in range(3): a.text(j, i, f"{raw[j]:,.1f}" if j < 2 else f"{raw[j]:,.0f}", ha="center", va="center", fontsize=10)
a.set_xticks(range(3), zs.columns); a.set_yticks(range(len(order)), [f"군집 {c}" for c in order]); a.set_title("군집별 중앙값 (색: 전체 중앙값 대비 표준화 편차)")
a = ax[1, 1]; x = np.arange(len(order))
a.bar(x - .2, prof["customer_share_%"], .4, label="고객 비중", color="#9ecae1"); a.bar(x + .2, prof["amount_share_%"], .4, label="금액 비중", color="#d94801")
for i in range(len(order)):
    a.text(i - .2, prof["customer_share_%"].iloc[i] + .7, f"{prof['customer_share_%'].iloc[i]:.0f}", ha="center", fontsize=9)
    a.text(i + .2, prof["amount_share_%"].iloc[i] + .7, f"{prof['amount_share_%'].iloc[i]:.0f}", ha="center", fontsize=9)
a.set_xticks(x, [f"군집 {c}" for c in order]); a.set_title("군집별 고객 비중 vs 금액 비중 (%)"); a.legend(); a.grid(axis="y", alpha=.3)
plt.tight_layout(); fig.savefig("outputs/figures/basket_clusters.png", dpi=130)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("고객", len(f)); print(f[["avg_products_per_order", "qty_per_product", "total_amount", "orders", "total_distinct_products"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
print(ev.round(3).to_string(index=False)); print("선택 k:", k)
print(prof.round(2).T.to_string()); print("RFM 군집과 교차:\n", ct.to_string()); print("로그 순위상관:\n", corr.round(2).to_string())
print(f.head(5).round(2)[["customer_id", "orders", "avg_products_per_order", "qty_per_product", "total_amount", "total_distinct_products"]].to_string(index=False))
