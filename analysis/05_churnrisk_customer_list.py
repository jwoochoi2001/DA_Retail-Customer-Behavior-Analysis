"""이탈 위험 고객 관리표: 사용자 ID, 최근 방문일(마지막 구매일), 누적 구매 횟수.

실행: python analysis/05_churnrisk_customer_list.py  (프로젝트 루트 기준, 02 먼저 실행)
- 이탈 위험 = R 평균이 가장 큰 군집(02 결과). 누적 구매 횟수 = 기준일(2011-09-01) 이전 전체 이력의 구매 주문 수
  (취소 제외, 기준일 이후 정보 미사용). 정렬: 누적 구매 횟수 내림차순, 같으면 최근 방문일 오래된 순.
"""
import pandas as pd

REF = pd.Timestamp("2011-09-01")
rfm = pd.read_csv("outputs/tables/rfm_customers.csv", parse_dates=["last_purchase"])
risk = rfm.groupby("cluster")["R"].mean().idxmax()
r = rfm[rfm.cluster == risk]

df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
hist = df[(df.invoice_date < REF) & (~df.is_cancel) & df.customer_id.isin(r.customer_id)]
cum = hist.groupby("customer_id").agg(cumulative_orders=("invoice", "nunique"), first_purchase_date=("invoice_date", "min"),
                                      last_hist=("invoice_date", "max")).reset_index()
out = (r[["customer_id", "last_purchase", "R", "F"]].merge(cum, on="customer_id", validate="1:1")
       .rename(columns={"last_purchase": "last_visit_date", "R": "days_since_last_visit", "F": "orders_in_180d"}))
assert len(out) == len(r) == out.customer_id.nunique() and (out.last_visit_date == out.last_hist).all()
assert (out.cumulative_orders >= out.orders_in_180d).all()
out["last_visit_date"] = out.last_visit_date.dt.date
out = (out[["customer_id", "last_visit_date", "cumulative_orders", "days_since_last_visit", "orders_in_180d"]]
       .sort_values(["cumulative_orders", "last_visit_date"], ascending=[False, True]).reset_index(drop=True))
out.to_csv("outputs/tables/churnrisk_customer_list.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200)
print("이탈 위험 군집:", risk, "| 고객 수:", len(out)); print(out.head(15).to_string(index=False))
print(out.cumulative_orders.value_counts().sort_index().to_string())
print("방문 경과일:", out.days_since_last_visit.describe().round(1).to_dict())
