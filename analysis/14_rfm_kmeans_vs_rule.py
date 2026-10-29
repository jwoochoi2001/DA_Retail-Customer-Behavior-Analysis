"""K-Means RFM(k=2~10)와 규칙 기반 RFM(5분위 점수 -> 6개 세그먼트) 비교, 실루엣 기준 최적 k 추천.

실행: python analysis/14_rfm_kmeans_vs_rule.py  (프로젝트 루트 기준, 02 먼저 실행)
입력: outputs/tables/rfm_customers.csv (02에서 만든 R/F/M, 기준일 2011-09-01, 직전 180일 구매 고객)
- 공통 비교 공간: log1p(R,F,M) 표준화. K-Means는 이 공간에서 군집화, 규칙 기반은 분위 점수 규칙으로 라벨링.
- 평가: 실루엣(주 지표), Calinski-Harabasz(높을수록), Davies-Bouldin(낮을수록), 군집 간 분산 설명 비율(R^2),
  K-Means는 시드를 바꿔 다시 돌렸을 때의 안정성(ARI)도 확인.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_samples, silhouette_score)
from sklearn.preprocessing import StandardScaler
import _plot_style  # 한글 글꼴·마이너스 기호 설정 (analysis/_plot_style.py)

rfm = pd.read_csv("outputs/tables/rfm_customers.csv")[["customer_id", "R", "F", "M"]]
X = StandardScaler().fit_transform(np.log1p(rfm[["R", "F", "M"]]))
tss = ((X - X.mean(0)) ** 2).sum()


def metrics(lab):
    u = np.unique(lab)
    bss = sum((lab == c).sum() * ((X[lab == c].mean(0) - X.mean(0)) ** 2).sum() for c in u)
    return dict(groups=len(u), silhouette=silhouette_score(X, lab), calinski_harabasz=calinski_harabasz_score(X, lab),
                davies_bouldin=davies_bouldin_score(X, lab), r2_between=bss / tss,
                min_group_share_pct=pd.Series(lab).value_counts(normalize=True).min() * 100)


# ---- K-Means: k=2~10 ----
rows, labs = [], {}
for k in range(2, 11):
    km = KMeans(k, n_init=20, random_state=42).fit(X)
    labs[k] = km.labels_
    other = [KMeans(k, n_init=20, random_state=s).fit_predict(X) for s in (1, 2, 3, 4, 5)]
    rows.append({"k": k, **metrics(km.labels_), "inertia": km.inertia_,
                 "stability_ARI": np.mean([adjusted_rand_score(km.labels_, o) for o in other])})
kev = pd.DataFrame(rows)
kev.round(4).to_csv("outputs/tables/rfm_kmeans_k_evaluation.csv", index=False, encoding="utf-8-sig")

# ---- 규칙 기반 RFM: 5분위 점수 ----
rb = rfm.copy()
rb["R_s"] = pd.qcut(rb.R.rank(method="first", ascending=False), 5, labels=[1, 2, 3, 4, 5]).astype(int)  # 최근일수록 5
rb["F_s"] = pd.qcut(rb.F.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rb["M_s"] = pd.qcut(rb.M.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rb["FM"] = (rb.F_s + rb.M_s) / 2


def seg(r):
    if r.R_s >= 4:
        return "챔피언" if r.FM >= 4 else ("잠재 충성" if r.FM >= 3 else "신규·저가 최근")
    if r.R_s == 3:
        return "충성" if r.FM >= 4 else "관심 필요"
    return "이탈 위험(고가치)" if r.FM >= 3.5 else "휴면"


rb["segment"] = rb.apply(seg, axis=1)
order = ["챔피언", "충성", "잠재 충성", "신규·저가 최근", "관심 필요", "이탈 위험(고가치)", "휴면"]
rule_lab = rb.segment.map({s: i for i, s in enumerate(order)}).values
assert rb.segment.notna().all() and set(rb.segment) <= set(order)

prof = (rb.groupby("segment").agg(customers=("R", "size"), R_mean=("R", "mean"), F_mean=("F", "mean"), M_mean=("M", "mean"),
                                  M_sum=("M", "sum")).reindex(order))
prof["customer_share_%"] = prof.customers / prof.customers.sum() * 100; prof["amount_share_%"] = prof.M_sum / prof.M_sum.sum() * 100
prof.round(2).to_csv("outputs/tables/rfm_rule_segments_profile.csv", encoding="utf-8-sig")
rb.to_csv("outputs/tables/rfm_rule_segments.csv", index=False, encoding="utf-8-sig")

# ---- 방법 비교 ----
n_rule = len(np.unique(rule_lab))
cmp_rows = [("규칙 기반 (5분위 점수, %d개 세그먼트)" % n_rule, metrics(rule_lab), np.nan)]
rec_pool = kev[kev.k >= 3]
k_best_sil = int(kev.loc[kev.silhouette.idxmax(), "k"]); k_rec = int(rec_pool.loc[rec_pool.silhouette.idxmax(), "k"])
for k in sorted({k_rec, n_rule, 3}):
    cmp_rows.append((f"K-Means k={k}", metrics(labs[k]), kev.loc[kev.k == k, "stability_ARI"].iloc[0]))
cmp = pd.DataFrame([{"method": n, **m, "stability_ARI": s} for n, m, s in cmp_rows])
cmp.round(4).to_csv("outputs/tables/rfm_method_comparison.csv", index=False, encoding="utf-8-sig")

# 규칙 기반 vs K-Means(추천 k) 일치도
ct = pd.crosstab(rb.segment, labs[k_rec]).reindex(order)
ct.to_csv("outputs/tables/rfm_rule_vs_kmeans_crosstab.csv", encoding="utf-8-sig")
ari, nmi = adjusted_rand_score(rule_lab, labs[k_rec]), normalized_mutual_info_score(rule_lab, labs[k_rec])

# K-Means 추천 k 프로파일
kp = rfm.assign(cluster=labs[k_rec]).groupby("cluster").agg(customers=("R", "size"), R_mean=("R", "mean"), F_mean=("F", "mean"),
                                                            M_mean=("M", "mean"), M_sum=("M", "sum"))
kp["customer_share_%"] = kp.customers / kp.customers.sum() * 100; kp["amount_share_%"] = kp.M_sum / kp.M_sum.sum() * 100
kp = kp.sort_values("M_mean", ascending=False).round(2)
kp.to_csv("outputs/tables/rfm_kmeans_recommended_profile.csv", encoding="utf-8-sig")

# ---- 그림 ----
fig, ax = plt.subplots(2, 2, figsize=(15, 10.5))
a = ax[0, 0]; a.plot(kev.k, kev.silhouette, "o-", color="#4f81bd")
a.axvline(k_rec, color="#c0504d", ls="--", label=f"추천 k={k_rec} (k≥3 중 최대)")
for k_, v in zip(kev.k, kev.silhouette): a.text(k_, v + .004, f"{v:.3f}", ha="center", fontsize=8)
a.set_title("K-Means 실루엣 점수 (k=2~10)"); a.set_xlabel("k"); a.set_ylabel("실루엣"); a.legend(); a.grid(alpha=.3)
a = ax[0, 1]; a.plot(kev.k, kev.inertia, "o-", color="#9bbb59"); a.set_xlabel("k"); a.set_ylabel("inertia (군집 내 제곱합)")
a2 = a.twinx(); a2.plot(kev.k, kev.davies_bouldin, "s--", color="#8064a2"); a2.set_ylabel("Davies-Bouldin (낮을수록 좋음)")
a.set_title("엘보(inertia)와 Davies-Bouldin"); a.grid(alpha=.3)
a = ax[1, 0]
names = ["규칙 기반\n(%d개)" % n_rule] + [f"K-Means\nk={k}" for k in sorted({3, k_rec, n_rule})]
vals = [cmp.silhouette.iloc[0]] + [cmp.loc[cmp.method == f"K-Means k={k}", "silhouette"].iloc[0] for k in sorted({3, k_rec, n_rule})]
b = a.bar(names, vals, color=["#c0504d"] + ["#4f81bd"] * (len(vals) - 1))
for bb, v in zip(b, vals): a.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
a.set_title("방법별 실루엣 비교 (로그·표준화 RFM 공간)"); a.set_ylim(0, max(vals) * 1.2); a.grid(axis="y", alpha=.3)
a = ax[1, 1]; im = a.imshow(ct.div(ct.sum(1), axis=0).values * 100, cmap="Blues", aspect="auto", vmin=0, vmax=100)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]): a.text(j, i, f"{ct.iloc[i, j]}", ha="center", va="center", fontsize=9)
a.set_xticks(range(ct.shape[1]), [f"군집 {c}" for c in ct.columns]); a.set_yticks(range(ct.shape[0]), ct.index)
a.set_title(f"규칙 기반 세그먼트 × K-Means(k={k_rec}) 고객 수 (색: 행 기준 %)\nARI={ari:.2f}, NMI={nmi:.2f}")
plt.tight_layout(); fig.savefig("outputs/figures/rfm_kmeans_vs_rule.png", dpi=130)

pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
print("고객", len(rfm)); print(kev.round(3).to_string(index=False))
print("실루엣 최대 k(전체):", k_best_sil, "| k>=3 중 최대:", k_rec)
print(cmp.round(3).to_string(index=False)); print(prof.round(2).to_string()); print(kp.to_string()); print(ct.to_string())
print(f"ARI={ari:.3f} NMI={nmi:.3f}")
for k in (k_best_sil, k_rec):
    s = silhouette_samples(X, labs[k]); print(f"k={k} 군집별 평균 실루엣:", pd.Series(s).groupby(labs[k]).mean().round(3).to_dict(),
                                           "| 음수 실루엣 비율 %:", round((s < 0).mean() * 100, 1))
