"""첫 구매 월 기준 월별 구매 유지율 코호트 히트맵.

실행: python analysis/13_cohort_retention.py  (프로젝트 루트 기준, 01 먼저 실행)
- 코호트 = 고객의 첫 구매(취소 제외) 월. 유지율(k개월차) = 코호트 고객 중 첫 구매 후 k개월째 달에 1건 이상 구매한 고객 비율.
- 데이터가 2011-12-09에 끝나 2011-12는 부분 월이므로 분석에서 제외(코호트·경과 월 모두).
- 2009-12 코호트는 데이터 시작 월이라 기존 고객이 섞여 있어 '실제 신규'가 아님(그래프에 표시).
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
p = df[~df.is_cancel].copy()
p["month"] = p.invoice_date.dt.to_period("M")
LAST = pd.Period("2011-11", "M")                         # 마지막 완전한 월
p = p[p.month <= LAST]
first = p.groupby("customer_id").month.min().rename("cohort")
a = p[["customer_id", "month"]].drop_duplicates().merge(first, on="customer_id")
a["k"] = (a.month - a.cohort).apply(lambda x: x.n)

cnt = a.groupby(["cohort", "k"]).customer_id.nunique().unstack("k")
size = cnt[0]
ret = cnt.div(size, axis=0) * 100
# 관측 불가능한 칸(코호트 월 + k > 마지막 월)은 NaN으로 두고, 관측 가능한데 0명인 칸은 0%
for coh in ret.index:
    maxk = (LAST - coh).n
    ret.loc[coh, [k for k in ret.columns if k > maxk]] = np.nan
    ret.loc[coh, [k for k in range(maxk + 1) if k not in ret.columns]] = 0
ret = ret.reindex(columns=range(ret.columns.max() + 1))
for coh in ret.index:
    maxk = (LAST - coh).n
    ret.loc[coh, [k for k in ret.columns if k <= maxk]] = ret.loc[coh, [k for k in ret.columns if k <= maxk]].fillna(0)
assert (ret[0] == 100).all() and size.sum() == p.customer_id.nunique()

out = ret.round(1); out.insert(0, "cohort_size", size)
out.index = out.index.astype(str); out.index.name = "cohort"
out.to_csv("outputs/tables/cohort_retention.csv", encoding="utf-8-sig")
cnt.fillna(0).astype(int).rename_axis(None).set_axis(cnt.index.astype(str)).to_csv("outputs/tables/cohort_active_customers.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(19, 10))
M = ret.values
im = ax.imshow(np.ma.masked_invalid(M), cmap="YlGnBu", vmin=0, vmax=60, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if (M[i, j] > 38 or j == 0) else "black")
ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"{k}" for k in ret.columns], fontsize=9)
ylab = [f"{c}  (n={int(size[c]):,})" + ("  *" if str(c) == "2009-12" else "") for c in ret.index]
ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(ylab, fontsize=9)
ax.set_xlabel("첫 구매 후 경과 개월 (0 = 첫 구매 월)"); ax.set_ylabel("첫 구매 월 코호트 (고객 수)")
ax.set_title("첫 구매 월 기준 월별 구매 유지율 (%)  —  2011-12는 부분 월이라 제외, * 2009-12 코호트는 기존 고객 포함", fontsize=12)
cb = fig.colorbar(im, ax=ax, pad=.01); cb.set_label("유지율 (%)  (색상 상한 60%)")
plt.tight_layout(); fig.savefig("outputs/figures/cohort_retention_heatmap.png", dpi=130)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("기간:", p.invoice_date.min().date(), "~", p.invoice_date.max().date(), "| 고객", len(size), "| 코호트", len(size))
print(out.iloc[:, :14].to_string())
col = ret.iloc[:, 1:]
print("가중평균 유지율(코호트 규모 가중): 1,2,3,6,12개월차")
for k in (1, 2, 3, 6, 12):
    ok = ret[k].notna(); print(k, round((cnt[k].fillna(0)[ok].sum() / size[ok].sum()) * 100, 1), "| 2010-01 이후 코호트만:",
                                round((cnt[k].fillna(0)[ok & (ret.index >= pd.Period('2010-01', 'M'))].sum() / size[ok & (ret.index >= pd.Period('2010-01', 'M'))].sum()) * 100, 1))
print("코호트 규모:", size.astype(int).to_dict())
