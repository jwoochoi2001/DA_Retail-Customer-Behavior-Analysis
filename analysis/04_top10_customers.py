"""구매금액(M) 상위 10% 고객의 금액 비중, 상위 5명, 상위 10% vs 나머지의 R/F/M·구매당 금액.

실행: python analysis/04_top10_customers.py  (프로젝트 루트 기준, 02 먼저 실행)
대상: 02와 동일(2011-03-05~2011-08-31 구매 고객). 상위 10% = M 내림차순 상위 ceil(고객 수*10%)명.
"""
import math
import pandas as pd

rfm = pd.read_csv("outputs/tables/rfm_customers.csv", parse_dates=["last_purchase"])
rfm["AOV"] = rfm["M"] / rfm["F"]
rfm = rfm.sort_values(["M", "customer_id"], ascending=[False, True]).reset_index(drop=True)
n_top = math.ceil(len(rfm) * 0.10)
rfm["group"] = ["상위 10%" if i < n_top else "나머지 90%" for i in range(len(rfm))]
top = rfm[rfm.group == "상위 10%"]
assert rfm.M.sum() > 0 and len(top) == n_top

share = top.M.sum() / rfm.M.sum() * 100
top5 = top.head(5)[["customer_id", "F", "M", "R", "AOV", "last_purchase", "cluster"]].copy()
top5.insert(2, "repeat_orders(F-1)", top5.F - 1); top5["M_share_%"] = top5.M / rfm.M.sum() * 100
top5 = top5.round(2)

def agg(d):
    return pd.Series({"customers": len(d), "M_sum": d.M.sum(), "M_share_%": d.M.sum() / rfm.M.sum() * 100,
        "orders": d.F.sum(), "R_mean": d.R.mean(), "R_median": d.R.median(), "F_mean": d.F.mean(), "F_median": d.F.median(),
        "M_mean": d.M.mean(), "M_median": d.M.median(), "AOV_total(M/F)": d.M.sum() / d.F.sum(),
        "AOV_cust_median": d.AOV.median(), "repeat_cust_%(F>=2)": (d.F >= 2).mean() * 100})
cmp_ = pd.DataFrame({"상위 10%": agg(top), "나머지 90%": agg(rfm[rfm.group != "상위 10%"]), "전체": agg(rfm)}).T.round(2)
top_dist = top[["R", "F", "M", "AOV"]].describe(percentiles=[.25, .5, .75]).round(2)
comp = pd.crosstab(top.cluster, "n").assign(**{"%": lambda x: x.n / x.n.sum() * 100}).round(1)
cum = {p: round(rfm.head(math.ceil(len(rfm) * p / 100)).M.sum() / rfm.M.sum() * 100, 1) for p in (1, 5, 10, 20, 50)}

rfm.drop(columns="last_purchase").assign(last_purchase=rfm.last_purchase).to_csv("outputs/tables/rfm_top10_flag.csv", index=False, encoding="utf-8-sig")
cmp_.to_csv("outputs/tables/top10_vs_rest.csv", encoding="utf-8-sig"); top5.to_csv("outputs/tables/top5_customers.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(f"전체 {len(rfm):,}명, 상위 10% = {n_top}명 | M 컷오프(최소) {top.M.min():,.2f} | 상위10% 금액 비중 {share:.2f}% ({top.M.sum():,.2f} / {rfm.M.sum():,.2f})")
print("상위 p% 고객의 누적 금액 비중:", cum)
print(top5.to_string(index=False)); print(cmp_.to_string()); print(top_dist.to_string()); print(comp.to_string())
