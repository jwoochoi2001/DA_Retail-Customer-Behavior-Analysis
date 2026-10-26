"""이탈 위험 군집 내부 세부 분류: 경과일(R) 30일 기준 x 누적 구매 횟수 3회 기준.

실행: python analysis/06_churn_segments.py  (프로젝트 루트 기준, 02 먼저 실행)
대상: 02에서 R 평균이 가장 큰 군집(이탈 위험)의 고객만. 군집은 새로 묶지 않는다.
경과일 = 기준일(2011-09-01) - 마지막 구매일. 누적 구매 횟수 = 기준일 이전 전체 이력의 구매 주문 수(취소 제외).
"""
import pandas as pd

REF = pd.Timestamp("2011-09-01")
rfm = pd.read_csv("outputs/tables/rfm_customers.csv", parse_dates=["last_purchase"])
risk = rfm.groupby("cluster")["R"].mean().idxmax()
r = rfm[rfm.cluster == risk]

df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
hist = df[(df.invoice_date < REF) & (~df.is_cancel) & df.customer_id.isin(r.customer_id)]
cum = hist.groupby("customer_id").invoice.nunique().rename("cumulative_orders")
c = r.merge(cum, on="customer_id", validate="1:1")
assert len(c) == len(r) and (c.cumulative_orders >= c.F).all()

c["recency"] = (c.R > 30).map({True: "경과 30일 초과", False: "경과 30일 이하"})
c["loyalty"] = (c.cumulative_orders >= 3).map({True: "누적 3회 이상", False: "누적 3회 미만"})
c["segment"] = c.recency + " / " + c.loyalty
order = ["경과 30일 초과 / 누적 3회 이상", "경과 30일 초과 / 누적 3회 미만",
         "경과 30일 이하 / 누적 3회 이상", "경과 30일 이하 / 누적 3회 미만"]

g = c.groupby("segment")
s = g.agg(customers=("customer_id", "size"), R_mean=("R", "mean"), R_median=("R", "median"),
          cum_orders_mean=("cumulative_orders", "mean"), cum_orders_median=("cumulative_orders", "median"),
          F_window_mean=("F", "mean"), M_sum=("M", "sum"), orders=("F", "sum")).reindex(order)
s["customer_share_%"] = s.customers / s.customers.sum() * 100; s["M_share_%"] = s.M_sum / s.M_sum.sum() * 100
s["avg_order_value"] = s.M_sum / s.orders
s.loc["이탈 위험 전체"] = [len(c), c.R.mean(), c.R.median(), c.cumulative_orders.mean(), c.cumulative_orders.median(),
                      c.F.mean(), c.M.sum(), c.F.sum(), 100, 100, c.M.sum() / c.F.sum()]
s = s.round(2)
assert s.loc[order, "customers"].sum() == len(r) and abs(s.loc[order, "M_sum"].sum() - r.M.sum()) < 1

out = c[["customer_id", "last_purchase", "R", "cumulative_orders", "F", "M", "recency", "loyalty", "segment"]].copy()
out["last_purchase"] = out.last_purchase.dt.date
out = out.rename(columns={"last_purchase": "last_visit_date", "R": "days_since_last_visit",
                          "F": "orders_in_180d", "M": "amount_180d"})
out["_o"] = out.segment.map({k: i for i, k in enumerate(order)})
out = out.sort_values(["_o", "cumulative_orders", "days_since_last_visit"], ascending=[True, False, False]).drop(columns="_o")
out.to_csv("outputs/tables/churn_segment_customers.csv", index=False, encoding="utf-8-sig")
s.to_csv("outputs/tables/churn_segment_summary.csv", encoding="utf-8-sig")

pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30)
print("이탈 위험 군집:", risk, "| 고객:", len(c)); print(s.to_string())
print(pd.crosstab(c.cumulative_orders.clip(upper=10), c.recency).to_string())
