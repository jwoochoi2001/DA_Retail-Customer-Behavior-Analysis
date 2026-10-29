"""핵심 우량 군집(K-Means k=4) 산포도와 분위(10/25/50/75/90%) 구간 분석.

실행: python analysis/16_core_cluster_scatter.py  (프로젝트 루트 기준, 02·15 먼저 실행)
- 핵심 우량 = 02의 군집 중 M 평균이 가장 큰 군집.
- 구간: 구매금액(M) 분위 경계 P10, P25, P50(중앙값), P75, P90으로 6개 구간(하위 10% / 10~25 / 25~50 / 50~75 / 75~90 / 상위 10%).
- 산포도: F x M, R x M, R x F (F, M은 로그 축)에 R/F/M의 P10·P25·중앙값·P75·P90 기준선 표시.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _plot_style  # 한글 글꼴·마이너스 기호 설정 (analysis/_plot_style.py)

rfm = pd.read_csv("outputs/tables/rfm_customers.csv")
core_id = rfm.groupby("cluster").M.mean().idxmax()
c = rfm[rfm.cluster == core_id][["customer_id", "R", "F", "M"]].copy()
c["AOV"] = c.M / c.F
rp = pd.read_csv("outputs/tables/repurchase_customers.csv")[["customer_id", "rp30", "rp60", "rp90", "orders_90", "amount_90"]]
c = c.merge(rp, on="customer_id", validate="1:1")
n = len(c)

QS = [.10, .25, .50, .75, .90]; QN = ["P10", "P25", "중앙값", "P75", "P90"]
pt = pd.DataFrame({v: c[v].quantile(QS).values for v in ["R", "F", "M", "AOV"]}, index=QN)
pt["_"] = None; pt = pt.drop(columns="_")
pt.loc["평균"] = c[["R", "F", "M", "AOV"]].mean(); pt.loc["최소"] = c[["R", "F", "M", "AOV"]].min(); pt.loc["최대"] = c[["R", "F", "M", "AOV"]].max()
pt.round(1).to_csv("outputs/tables/core_cluster_percentiles.csv", encoding="utf-8-sig")

edges = [-np.inf] + c.M.quantile(QS).tolist() + [np.inf]
labels = ["하위 10%", "10~25%", "25~50%", "50~75%", "75~90%", "상위 10%"]
c["band"] = pd.cut(c.M, edges, labels=labels, right=False)
# 경계값과 같은 금액 때문에 구간 크기가 비정상적으로 달라지지 않는지 확인
assert c.band.notna().all() and len(c) == n
b = c.groupby("band", observed=True).agg(customers=("M", "size"), M_min=("M", "min"), M_max=("M", "max"), M_median=("M", "median"),
                                          R_median=("R", "median"), F_median=("F", "median"), AOV_median=("AOV", "median"),
                                          M_sum=("M", "sum"), rp30=("rp30", "mean"), rp90=("rp90", "mean"), amount_90=("amount_90", "mean"))
b["customer_share_%"] = b.customers / n * 100; b["amount_share_%"] = b.M_sum / c.M.sum() * 100
b[["rp30", "rp90"]] *= 100
b.round(2).to_csv("outputs/tables/core_cluster_bands.csv", encoding="utf-8-sig")
c.to_csv("outputs/tables/core_cluster_customers.csv", index=False, encoding="utf-8-sig")
corr = c[["R", "F", "M", "AOV"]].apply(np.log1p).corr(method="spearman")

# ---- 그림 ----
colors = ["#bdd7e7", "#6baed6", "#3182bd", "#08519c", "#fd8d3c", "#d94801"]
cm = dict(zip(labels, colors))
fig, ax = plt.subplots(2, 2, figsize=(17, 12.5))
qv = {v: c[v].quantile(QS).values for v in ["R", "F", "M"]}
ls = {0: ":", 1: "--", 2: "-", 3: "--", 4: ":"}; lw = {0: .9, 1: .9, 2: 1.6, 3: .9, 4: .9}


def panel(a, x, y, logx, logy, title):
    for lab in labels:
        d = c[c.band == lab]; a.scatter(d[x], d[y], s=22, alpha=.8, color=cm[lab], edgecolor="white", linewidth=.3, label=f"{lab} (n={len(d)})")
    for i, v in enumerate(qv[x]):
        a.axvline(v, color="#555", ls=ls[i], lw=lw[i]); a.text(v, 1.0, QN[i], transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(qv[y]):
        a.axhline(v, color="#555", ls=ls[i], lw=lw[i]); a.text(1.0, v, " " + QN[i], transform=a.get_yaxis_transform(), ha="left", va="center", fontsize=8)
    if logx: a.set_xscale("log")
    if logy: a.set_yscale("log")
    a.set_xlabel(x + (" (로그)" if logx else "")); a.set_ylabel(y + (" (로그)" if logy else "")); a.set_title(title, pad=22); a.grid(alpha=.25)


panel(ax[0, 0], "F", "M", True, True, "주문 수(F) × 구매금액(M)")
panel(ax[0, 1], "R", "M", False, True, "최근성(R, 일) × 구매금액(M)")
panel(ax[1, 0], "R", "F", False, True, "최근성(R, 일) × 주문 수(F)")
ax[0, 0].legend(fontsize=8, title="M 분위 구간", loc="upper left")
a = ax[1, 1]; x = np.arange(len(labels)); a.bar(x - .2, b["customer_share_%"], .4, color="#9ecae1", label="고객 비중(%)")
a.bar(x + .2, b["amount_share_%"], .4, color="#d94801", label="구매금액 비중(%)")
for i in range(len(labels)):
    a.text(i - .2, b["customer_share_%"].iloc[i] + .8, f"{b['customer_share_%'].iloc[i]:.0f}", ha="center", fontsize=9)
    a.text(i + .2, b["amount_share_%"].iloc[i] + .8, f"{b['amount_share_%'].iloc[i]:.1f}", ha="center", fontsize=9)
a.set_xticks(x, [f"{l}\nM {lo:,.0f}~{hi:,.0f}" for l, lo, hi in zip(labels, b.M_min, b.M_max)], fontsize=8)
a.set_title("M 분위 구간별 고객 비중 vs 구매금액 비중"); a.legend(); a.grid(axis="y", alpha=.3)
fig.suptitle(f"핵심 우량 군집 {n}명: 산포도와 분위 구간 (선: 점선 P10/P90, 파선 P25/P75, 실선 중앙값)", fontsize=13)
plt.tight_layout(); fig.savefig("outputs/figures/core_cluster_scatter.png", dpi=130)

pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
print("핵심 우량 군집 번호:", core_id, "| 고객", n, "| 금액 합계", round(c.M.sum()))
print(pt.round(1).to_string()); print(b.round(1).to_string())
print("로그 순위상관(Spearman):\n", corr.round(2).to_string())
top = c.nlargest(5, "M")[["customer_id", "R", "F", "M", "AOV"]]; print(top.round(1).to_string(index=False))
print("상위 1명 비중 %:", round(c.M.max() / c.M.sum() * 100, 1), "| 상위 10명 비중 %:", round(c.M.nlargest(10).sum() / c.M.sum() * 100, 1),
      "| 변동계수 M:", round(c.M.std() / c.M.mean(), 2), "| 왜도 M:", round(c.M.skew(), 2))
print("F 분포:", c.F.value_counts().sort_index().head(12).to_dict(), "| R<=7:", int((c.R <= 7).sum()), "R>=30:", int((c.R >= 30).sum()))
