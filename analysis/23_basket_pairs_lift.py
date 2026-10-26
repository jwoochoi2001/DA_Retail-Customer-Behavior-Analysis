"""공동 구매 비율 1% 이상 상품 쌍의 향상도(lift) 분석, 향상도 상위 vs 공동 구매 건수 상위 비교, 진열 추천 조합.

실행: python analysis/23_basket_pairs_lift.py  (프로젝트 루트 기준, 01·22 먼저 실행)
- 기본(전체 고객): retail_clean.csv 구매 주문. 민감도 확인: retail_clean_general.csv(일반 고객만)로 같은 계산.
- 지지도 = 함께 구매된 주문 수 / 전체 주문 수 >= 1% 쌍만 대상.
- 향상도 = P(A,B) / (P(A) P(B)). 신뢰도 = P(B|A), P(A|B).
- 같은 계열 판정: 색상·디자인 단어를 뺀 핵심 단어의 Jaccard >= 0.5 (같은 상품의 변형으로 간주).
- 추천 점수: 지지도>=1% 이고 향상도>=5, 두 방향 신뢰도 중 작은 값(min_conf) 기준으로 정렬.
"""
import re
import numpy as np, pandas as pd, matplotlib
from scipy import sparse
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False

SUP = 0.01
GENERIC = set("RED BLUE GREEN PINK YELLOW WHITE BLACK PURPLE ORANGE IVORY CREAM SILVER GOLD BROWN GREY GRAY TURQUOISE RETROSPOT "
              "POLKADOT SPOTTY STRAWBERRY SPACEBOY SUKI DESIGN SKULL DOLLY GIRL ANTIQUE VINTAGE SET OF PACK AND IN THE".split())


def pairs_of(path):
    df = pd.read_csv(path)
    b = df[~df.is_cancel][["invoice", "stock_code"]].drop_duplicates()
    name = df.groupby("stock_code").description.agg(lambda s: s.mode().iat[0])
    ic, inv = pd.factorize(b.invoice); sc_, sc = pd.factorize(b.stock_code)
    X = sparse.csr_matrix((np.ones(len(b), dtype=np.int32), (ic, sc_)), shape=(len(inv), len(sc)))
    n = X.shape[0]; C = (X.T @ X).tocoo(); io = np.asarray(X.sum(0)).ravel()
    m = (C.row < C.col) & (C.data >= SUP * n)
    p = pd.DataFrame({"a": sc[C.row[m]], "b": sc[C.col[m]], "orders_together": C.data[m]})
    p["orders_a"] = io[C.row[m]]; p["orders_b"] = io[C.col[m]]
    p["support_%"] = p.orders_together / n * 100
    p["conf_a_to_b_%"] = p.orders_together / p.orders_a * 100; p["conf_b_to_a_%"] = p.orders_together / p.orders_b * 100
    p["min_conf_%"] = p[["conf_a_to_b_%", "conf_b_to_a_%"]].min(axis=1)
    p["lift"] = p.orders_together * n / (p.orders_a * p.orders_b)
    p["desc_a"] = p.a.map(name); p["desc_b"] = p.b.map(name)
    return p, n


def core_tokens(s):
    return set(re.sub(r"[^A-Z ]+", " ", str(s).upper()).split()) - GENERIC


def same_family(a, b):
    ta, tb = core_tokens(a), core_tokens(b)
    return len(ta & tb) / max(1, len(ta | tb)) >= 0.5


P, N = pairs_of("data/processed/retail_clean.csv")
P["same_family"] = [same_family(x, y) for x, y in zip(P.desc_a, P.desc_b)]
P = P.sort_values("lift", ascending=False).reset_index(drop=True); P["rank_lift"] = np.arange(1, len(P) + 1)
P["rank_count"] = P.orders_together.rank(ascending=False, method="min").astype(int)
G, NG = pairs_of("data/processed/retail_clean_general.csv")
# 두 데이터셋에서 쌍의 a/b 순서가 다를 수 있어 순서와 무관한 키를 사용
key = lambda d: pd.Series(["|".join(sorted(t)) for t in zip(d.a, d.b)], index=d.index)
G["key"] = key(G); P["key"] = key(P)
gl = G.set_index("key")
P["support_general_%"] = P.key.map(gl["support_%"]); P["lift_general"] = P.key.map(gl.lift)
P.drop(columns="key").round(3).to_csv("outputs/tables/basket_pairs_support1pct.csv", index=False, encoding="utf-8-sig")

K = 10
top_lift, top_cnt = P.nsmallest(K, "rank_lift"), P.nsmallest(K, "rank_count")
overlap = set(top_lift.index) & set(top_cnt.index)
corr = P[["orders_together", "lift", "min_conf_%"]].corr(method="spearman")

rec = P[(P.lift >= 5)].sort_values(["min_conf_%", "lift"], ascending=False).copy()
rec["tier"] = np.where(rec.same_family, "A 같은 계열 변형 인접 진열", "B 교차 진열(다른 상품군)")
rec_cols = ["tier", "a", "desc_a", "b", "desc_b", "orders_together", "support_%", "conf_a_to_b_%", "conf_b_to_a_%", "min_conf_%", "lift",
            "support_general_%", "lift_general", "rank_lift", "rank_count"]
rec[rec_cols].round(2).to_csv("outputs/tables/basket_display_recommendations.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [1.15, 1]})
a = ax[0]
a.scatter(P["support_%"], P.lift, s=34, color="#9ecae1", edgecolor="white", linewidth=.4, label=f"전체 {len(P)}쌍")
a.scatter(top_cnt["support_%"], top_cnt.lift, s=70, color="#4f81bd", label="공동 구매 건수 상위 10")
a.scatter(top_lift["support_%"], top_lift.lift, s=70, marker="D", color="#d94801", label="향상도 상위 10")
for _, r in top_lift.head(5).iterrows(): a.annotate(f"{r.desc_a[:16]}+{r.desc_b[:16]}", (r["support_%"], r.lift), fontsize=7, xytext=(4, 4), textcoords="offset points")
a.axhline(5, color="gray", ls=":", lw=1); a.axvline(SUP * 100, color="gray", ls=":", lw=1)
a.set_xlabel("지지도 (함께 구매된 주문 / 전체 주문, %)"); a.set_ylabel("향상도 (lift)"); a.set_title("지지도 × 향상도 (지지도 1% 이상 쌍)"); a.legend(fontsize=8); a.grid(alpha=.3)
a = ax[1]; r10 = rec.head(10); y = np.arange(len(r10))[::-1]
a.barh(y, r10.lift, color=np.where(r10.same_family, "#9bbb59", "#d94801"))
for yy, (_, r) in zip(y, r10.iterrows()): a.text(r.lift, yy, f"  lift {r.lift:.1f} · 건수 {int(r.orders_together):,} · 최소신뢰도 {r['min_conf_%']:.0f}%", va="center", fontsize=8)
a.set_yticks(y, [f"{x[:20]} + {z[:20]}" for x, z in zip(r10.desc_a, r10.desc_b)], fontsize=8); a.set_xlim(0, r10.lift.max() * 1.7)
a.set_title("진열 추천 상위 10 (녹색=같은 계열, 주황=교차)"); a.grid(axis="x", alpha=.3)
plt.tight_layout(); fig.savefig("outputs/figures/basket_pairs_lift.png", dpi=130)

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 34)
print("전체 주문", N, "| 일반 고객 주문", NG, "| 지지도>=1% 쌍:", len(P), "(일반 고객:", len(G), ") | 같은 계열 쌍:", int(P.same_family.sum()))
c = ["desc_a", "desc_b", "orders_together", "support_%", "min_conf_%", "lift", "same_family", "rank_count"]
print("[향상도 상위 10]"); print(top_lift[c].round(2).to_string(index=False))
c2 = ["desc_a", "desc_b", "orders_together", "support_%", "min_conf_%", "lift", "same_family", "rank_lift"]
print("[공동 구매 건수 상위 10]"); print(top_cnt[c2].round(2).to_string(index=False))
print("두 목록 겹침:", len(overlap), "| 순위상관(건수-향상도):", round(corr.loc["orders_together", "lift"], 2), "| 건수-최소신뢰도:", round(corr.loc["orders_together", "min_conf_%"], 2))
print("lift 분포:", P.lift.describe(percentiles=[.25, .5, .75, .9]).round(2).to_dict())
print("같은 계열 평균 lift:", round(P[P.same_family].lift.mean(), 2), "| 교차 평균 lift:", round(P[~P.same_family].lift.mean(), 2))
print("[교차 진열(다른 상품군) lift 상위 12]")
print(P[~P.same_family].nlargest(12, "lift")[["desc_a", "desc_b", "orders_together", "support_%", "min_conf_%", "lift", "support_general_%"]].round(2).to_string(index=False))
print("[추천 후보 lift>=5]", len(rec), "| A:", int(rec.same_family.sum()), "B:", int((~rec.same_family).sum()))
print(rec[["tier", "desc_a", "desc_b", "orders_together", "support_%", "min_conf_%", "lift", "support_general_%", "lift_general"]].head(25).round(2).to_string(index=False))
print("일반 고객에서도 지지도 1% 이상 유지 비율:", round(P.support_general_pct.notna().mean() * 100, 1) if "support_general_pct" in P else round(P["support_general_%"].notna().mean() * 100, 1))
