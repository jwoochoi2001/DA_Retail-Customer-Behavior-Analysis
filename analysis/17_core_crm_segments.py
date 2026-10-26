"""핵심 우량 군집을 R/M 기준으로 4개 CRM 타깃으로 나누고, 전략 수립에 필요한 지표를 계산.

실행: python analysis/17_core_crm_segments.py  (프로젝트 루트 기준, 02·15·16 먼저 실행)
타깃(위에서부터 먼저 맞는 규칙 하나만 적용, 중복 없음):
  A 이탈 조짐   : R >= 30일 (핵심 우량 R의 P90 근처)
  B VIP         : M >= P90 (핵심 우량 M의 상위 10%)  [A 제외]
  C 핵심 단골   : P50 <= M < P90                      [A 제외]
  D 성장 후보   : M < P50 (중앙값 미만)                [A 제외]
지표: 고객 수·금액 비중, R/F/M 중앙값, 30/90일 재구매율(기준일 2011-09-01 이후, 15번 결과), 평균 구매 간격(기간 내 주문 간격),
      구매 간격 대비 경과(R / 간격), 주문당 금액, 선호 요일, 90일 내 미재구매 고객의 직전 180일 금액(=이탈 시 위험 금액).
"""
import numpy as np, pandas as pd

REF = pd.Timestamp("2011-09-01"); START = REF - pd.Timedelta(days=180)
core = pd.read_csv("outputs/tables/core_cluster_customers.csv")[["customer_id", "R", "F", "M", "AOV", "rp30", "rp90", "amount_90"]]
n = len(core)
p90_m, p50_m = core.M.quantile(.90), core.M.quantile(.50)


def seg(r):
    if r.R >= 30: return "A 이탈 조짐"
    if r.M >= p90_m: return "B VIP"
    if r.M >= p50_m: return "C 핵심 단골"
    return "D 성장 후보"


core["target"] = core.apply(seg, axis=1)
order = ["A 이탈 조짐", "B VIP", "C 핵심 단골", "D 성장 후보"]
assert core.target.isin(order).all() and len(core) == n

df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
w = df[(df.invoice_date >= START) & (df.invoice_date < REF) & (~df.is_cancel) & df.customer_id.isin(core.customer_id)]
o = w.groupby(["customer_id", "invoice"]).agg(d=("invoice_date", "min"), amt=("line_amount", "sum"), qty=("quantity", "sum")).reset_index().sort_values(["customer_id", "d"])
# 구매 간격: 같은 날 여러 주문은 하루로 보고, 서로 다른 구매일 사이의 간격(일)의 고객별 중앙값
days = o.assign(day=o.d.dt.normalize()).drop_duplicates(["customer_id", "day"]).sort_values(["customer_id", "day"])
days["gap"] = days.groupby("customer_id").day.diff().dt.days
gap = days.groupby("customer_id").gap.median().rename("gap_days")
wd = o.assign(wd=o.d.dt.weekday).merge(core[["customer_id", "target"]], on="customer_id")
core = core.merge(gap, on="customer_id", how="left")
core["overdue_ratio"] = core.R / core.gap_days.clip(lower=1)          # 1 초과 = 평소 간격보다 오래 쉬는 중
core["at_risk_amount"] = np.where(core.rp90 == 0, core.M, 0)                 # 90일 내 재구매하지 않은 고객의 직전 180일 금액

g = core.groupby("target")
t = g.agg(customers=("M", "size"), R_median=("R", "median"), F_median=("F", "median"), M_median=("M", "median"), M_sum=("M", "sum"),
          AOV_median=("AOV", "median"), gap_median_days=("gap_days", "median"), overdue_median=("overdue_ratio", "median"),
          rp30=("rp30", "mean"), rp90=("rp90", "mean"), amount_90_per_cust=("amount_90", "mean"), amount_90_total=("amount_90", "sum"),
          at_risk_amount=("at_risk_amount", "sum"), not_rp90_customers=("rp90", lambda s: int((s == 0).sum()))).reindex(order)
t["customer_share_%"] = t.customers / n * 100; t["amount_share_%"] = t.M_sum / core.M.sum() * 100
t[["rp30", "rp90"]] *= 100
for tg in order:
    d = wd[wd.target == tg]; t.loc[tg, "top_weekday"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d.wd.value_counts().idxmax()]
    t.loc[tg, "orders_per_cust_180d"] = d.groupby("customer_id").invoice.nunique().mean()
    t.loc[tg, "items_per_order"] = d["qty"].mean()
# 조기 경보: 평소 구매 간격의 1.5배를 넘긴 고객 수(타깃 A 제외 포함 전체)
core["early_warning"] = core.overdue_ratio >= 1.5
t["early_warning_customers"] = core.groupby("target").early_warning.sum().reindex(order)
t.round(2).to_csv("outputs/tables/core_crm_segments.csv", encoding="utf-8-sig")
core.round(2).to_csv("outputs/tables/core_crm_target_list.csv", index=False, encoding="utf-8-sig")

# 상품: 타깃별 상위 품목(판매금액)
prod = w.merge(core[["customer_id", "target"]], on="customer_id")
topprod = {tg: prod[prod.target == tg].groupby("description").line_amount.sum().nlargest(5).round(0) for tg in order}

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
print("핵심 우량", n, "| M P50", round(p50_m), "P90", round(p90_m), "| 재구매율 전체 30/90:", round(core.rp30.mean() * 100, 1), round(core.rp90.mean() * 100, 1))
print(t.round(1).T.to_string())
for tg in order: print(tg, topprod[tg].to_dict())
print("간격 결측(구매일 1일뿐):", int(core.gap_days.isna().sum()))
print("조기 경보(R>=간격x1.5) 전체:", int(core.early_warning.sum()), "| 간격 중앙값 전체:", round(core.gap_days.median(), 1),
      "| 간격 결측:", int(core.gap_days.isna().sum()))
print("A 타깃 M 구성: M>=P90:", int(((core.target == 'A 이탈 조짐') & (core.M >= p90_m)).sum()), "| M 중앙값 이상:", int(((core.target == 'A 이탈 조짐') & (core.M >= p50_m)).sum()))
a = core[core.target == "A 이탈 조짐"].nlargest(5, "M")[["customer_id", "R", "F", "M", "gap_days", "overdue_ratio", "rp90"]]; print(a.round(1).to_string(index=False))
print("재구매 안 한 고객 수(90일):", int((core.rp90 == 0).sum()), "위험 금액:", round(core.at_risk_amount.sum()))
