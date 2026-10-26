"""대량 구매(도매·선물가게형) 고객을 표시하고 일반 고객만 남긴 분석용 데이터셋을 만든다.

실행: python analysis/22_general_customer_dataset.py  (프로젝트 루트 기준, 01 먼저 실행)
대량 구매 고객 = 아래 중 하나라도 해당(구매 주문 기준, 전체 기간 이력, 취소 제외).
  1) 주문당 수량(주문 합계 개수)의 고객별 중앙값 >= UNITS_PER_ORDER
  2) 상품 행당 수량의 고객별 중앙값 >= QTY_PER_LINE   (12개입 단위 도매 구매 흔적: 24 = 2다스)
중앙값 사용 이유: 한 번의 큰 주문에 휘둘리지 않고 반복되는 구매 습관을 본다.
임계값은 분포에 뚜렷한 분리점이 없어(로그 척도 단봉) 상위 약 10% 경계와 24개로 정한 가정값. 상수만 바꾸면 재실행 가능.
제외 기록: 고객 수·행 수·금액을 summary에 남김. 취소 행은 같은 고객 기준으로 함께 제외.
"""
import numpy as np, pandas as pd

UNITS_PER_ORDER, QTY_PER_LINE = 400, 24
df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
buy = df[~df.is_cancel]
o = buy.groupby(["customer_id", "invoice"]).agg(units=("quantity", "sum"), amt=("line_amount", "sum")).reset_index()
c = o.groupby("customer_id").agg(orders=("invoice", "size"), med_units_per_order=("units", "median"),
                                 med_order_amount=("amt", "median"), total_amount=("amt", "sum"))
c["med_qty_per_line"] = buy.groupby("customer_id").quantity.median()
c["bulk_rule_units"] = c.med_units_per_order >= UNITS_PER_ORDER
c["bulk_rule_qty"] = c.med_qty_per_line >= QTY_PER_LINE
c["is_bulk"] = c.bulk_rule_units | c.bulk_rule_qty
cust_all = df.groupby("customer_id").size().index
c = c.reindex(cust_all)                                     # 취소 행만 있는 고객은 NaN -> 일반 고객으로 취급
c[["bulk_rule_units", "bulk_rule_qty", "is_bulk"]] = c[["bulk_rule_units", "bulk_rule_qty", "is_bulk"]].fillna(False).astype(bool)
c.round(2).to_csv("outputs/tables/bulk_customer_flags.csv", encoding="utf-8-sig")

df["is_bulk"] = df.customer_id.map(c.is_bulk)
gen, blk = df[~df.is_bulk], df[df.is_bulk]
assert len(gen) + len(blk) == len(df) and abs(gen.line_amount.sum() + blk.line_amount.sum() - df.line_amount.sum()) < 0.01
gen.drop(columns="is_bulk").to_csv("data/processed/retail_clean_general.csv", index=False, encoding="utf-8-sig")


def stat(d, label):
    b = d[~d.is_cancel]
    oo = b.groupby("invoice").agg(u=("quantity", "sum"), a=("line_amount", "sum"), n=("stock_code", "nunique"))
    return {"group": label, "customers": d.customer_id.nunique(), "rows": len(d), "purchase_amount": b.line_amount.sum(),
            "orders": b.invoice.nunique(), "units_per_order_median": oo.u.median(), "order_amount_median": oo.a.median(),
            "order_amount_mean": oo.a.mean(), "products_per_order_median": oo.n.median(),
            "orders_per_customer_median": b.groupby("customer_id").invoice.nunique().median(),
            "cancel_row_rate_%": d.is_cancel.mean() * 100}


s = pd.DataFrame([stat(df, "전체"), stat(gen, "일반 고객"), stat(blk, "대량 구매 고객")])
s["customer_share_%"] = s.customers / s.customers.iloc[0] * 100; s["amount_share_%"] = s.purchase_amount / s.purchase_amount.iloc[0] * 100
s["order_share_%"] = s.orders / s.orders.iloc[0] * 100
s.round(2).to_csv("outputs/tables/general_vs_bulk_summary.csv", index=False, encoding="utf-8-sig")

rfm = pd.read_csv("outputs/tables/rfm_customers.csv")
rfm["is_bulk"] = rfm.customer_id.map(c.is_bulk)
x = rfm.groupby("cluster").agg(customers=("is_bulk", "size"), bulk=("is_bulk", "sum"), amount=("M", "sum"))
x["bulk_amount"] = rfm[rfm.is_bulk].groupby("cluster").M.sum(); x = x.fillna(0)
x["bulk_%"] = x.bulk / x.customers * 100; x["bulk_amount_%"] = x.bulk_amount / x.amount * 100
x.round(1).to_csv("outputs/tables/bulk_share_by_rfm_cluster.csv", encoding="utf-8-sig")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("규칙: 주문당 수량 중앙값 >=", UNITS_PER_ORDER, "또는 행당 수량 중앙값 >=", QTY_PER_LINE)
print("대량 고객:", int(c.is_bulk.sum()), "| 규칙1만:", int((c.bulk_rule_units & ~c.bulk_rule_qty).sum()), "규칙2만:",
      int((~c.bulk_rule_units & c.bulk_rule_qty).sum()), "둘 다:", int((c.bulk_rule_units & c.bulk_rule_qty).sum()))
print(s.round(1).T.to_string()); print(x.round(1).to_string())
print(c[c.is_bulk].sort_values("total_amount", ascending=False).head(6).round(0)[["orders", "med_units_per_order", "med_qty_per_line", "total_amount"]].to_string())
