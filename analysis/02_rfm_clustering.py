"""2011-09-01 이전 180일 구매 고객 RFM + K-Means 군집, 군집별 주문당 평균 금액.

실행: python analysis/02_rfm_clustering.py   (프로젝트 루트 기준)
필요: pandas, scikit-learn (입력: data/processed/retail_clean.csv)
- 기준일: 2011-09-01, 분석 기간: [기준일-180일, 기준일)
- 구매만 사용(is_cancel=False). R=기준일-마지막 구매일(일), F=주문 수, M=금액 합계
- 군집: log1p 변환 후 표준화한 R/F/M에 K-Means, k는 실루엣 점수로 3~6 중 선택
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

REF = pd.Timestamp("2011-09-01"); START = REF - pd.Timedelta(days=180)
df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
w = df[(df["invoice_date"] >= START) & (df["invoice_date"] < REF) & (~df["is_cancel"])]

rfm = (w.groupby("customer_id")
       .agg(last_purchase=("invoice_date", "max"), F=("invoice", "nunique"), M=("line_amount", "sum"))
       .reset_index())
rfm["R"] = (REF - rfm["last_purchase"].dt.normalize()).dt.days
rfm["M"] = rfm["M"].round(2)
assert abs(rfm["M"].sum() - w["line_amount"].sum()) < 1 and rfm["F"].sum() == w["invoice"].nunique()

X = StandardScaler().fit_transform(np.log1p(rfm[["R", "F", "M"]]))
scores = {k: silhouette_score(X, KMeans(k, n_init=20, random_state=42).fit_predict(X)) for k in range(3, 7)}
k = max(scores, key=scores.get)
rfm["cluster"] = KMeans(k, n_init=20, random_state=42).fit_predict(X)

g = rfm.groupby("cluster")
summ = g.agg(customers=("customer_id", "size"), R_mean=("R", "mean"), F_mean=("F", "mean"),
             M_mean=("M", "mean"), orders=("F", "sum"), amount=("M", "sum")).reset_index()
summ["avg_order_value"] = summ["amount"] / summ["orders"]               # 군집 총금액 / 군집 총주문수
summ["avg_order_value_cust_mean"] = g.apply(lambda d: (d["M"] / d["F"]).mean(), include_groups=False).values
summ["customer_share_%"] = summ["customers"] / summ["customers"].sum() * 100
summ["amount_share_%"] = summ["amount"] / summ["amount"].sum() * 100
summ = summ.sort_values("M_mean", ascending=False).reset_index(drop=True)
tot = {"cluster": "전체", "customers": len(rfm), "R_mean": rfm.R.mean(), "F_mean": rfm.F.mean(),
       "M_mean": rfm.M.mean(), "orders": rfm.F.sum(), "amount": rfm.M.sum(),
       "avg_order_value": rfm.M.sum() / rfm.F.sum(), "avg_order_value_cust_mean": (rfm.M / rfm.F).mean(),
       "customer_share_%": 100.0, "amount_share_%": 100.0}
summ = pd.concat([summ, pd.DataFrame([tot])], ignore_index=True).round(2)

Path("outputs/tables").mkdir(parents=True, exist_ok=True)
rfm.drop(columns="last_purchase").assign(last_purchase=rfm["last_purchase"]).to_csv(
    "outputs/tables/rfm_customers.csv", index=False, encoding="utf-8-sig")
summ.to_csv("outputs/tables/rfm_cluster_summary.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(f"기간: {START.date()} ~ {(REF - pd.Timedelta(days=1)).date()} (기준일 {REF.date()}) | 구매 행 {len(w):,} | 고객 {len(rfm):,}")
print("실루엣:", {k_: round(v, 3) for k_, v in scores.items()}, "-> k =", k)
print(rfm.head(8).to_string(index=False))
print(rfm[["R", "F", "M"]].describe().round(2).to_string())
print(summ.to_string(index=False))
