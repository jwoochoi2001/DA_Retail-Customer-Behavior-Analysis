"""규칙 기반 RFM 세그먼트와 K-Means 군집(k=4)별 30/60/90일 재구매율.

실행: python analysis/15_repurchase_rate.py  (프로젝트 루트 기준, 01·02·14 먼저 실행)
- 대상: 02/14의 RFM 고객(기준일 2011-09-01 이전 180일 구매 고객).
- 재구매 = 기준일 이후 N일 안에 구매(취소 제외) 1건 이상. 30일=[09-01, 10-01), 60일=[09-01, 10-31), 90일=[09-01, 11-30).
  데이터가 2011-12-09까지라 90일 창이 모두 관측됨. 기준일 이전 정보로 만든 그룹을 기준일 이후 행동으로 평가(누수 없음).
- 비율의 95% 신뢰구간은 Wilson 방식. 구분력 비교용으로 그룹 재구매율을 점수로 쓴 AUC와 그룹 간 분산 설명 비율(eta^2)을 계산(표본 내 평가).
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False

REF = pd.Timestamp("2011-09-01"); WINDOWS = (30, 60, 90)
df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
assert df.invoice_date.max() >= REF + pd.Timedelta(days=max(WINDOWS))
rfm = pd.read_csv("outputs/tables/rfm_customers.csv")[["customer_id", "R", "F", "M", "cluster"]]
rule = pd.read_csv("outputs/tables/rfm_rule_segments.csv")[["customer_id", "segment"]]
base = rfm.merge(rule, on="customer_id", validate="1:1")
assert len(base) == len(rfm) == len(rule)

# K-Means 군집 이름: R 평균 최대=이탈 위험, 나머지 중 M 평균 최대=핵심 우량, 남은 둘 중 R 평균이 작은 쪽=최근 소액, 다른 쪽=중간 활성
cs = base.groupby("cluster").agg(R=("R", "mean"), M=("M", "mean"))
nm = {cs.R.idxmax(): "이탈 위험"}
rest = cs.drop(index=list(nm)); nm[rest.M.idxmax()] = "핵심 우량"
rest = rest.drop(index=[k for k in nm if k in rest.index]); nm[rest.R.idxmin()] = "최근 소액"; nm[rest.R.idxmax()] = "중간 활성"
base["kmeans"] = base.cluster.map(nm)

fut = df[(~df.is_cancel) & (df.invoice_date >= REF) & (df.invoice_date < REF + pd.Timedelta(days=max(WINDOWS)))]
first = fut.groupby("customer_id").invoice_date.min().dt.normalize()
base["days_to_first"] = (base.customer_id.map(first) - REF).dt.days
for w in WINDOWS:
    base[f"rp{w}"] = (base.days_to_first.notna() & (base.days_to_first < w)).astype(int)
f90 = fut.groupby("customer_id").agg(orders_90=("invoice", "nunique"), amount_90=("line_amount", "sum"))
base = base.merge(f90, on="customer_id", how="left").fillna({"orders_90": 0, "amount_90": 0})
assert (base.rp30 <= base.rp60).all() and (base.rp60 <= base.rp90).all()


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d * 100, (c + s) / d * 100


rule_order = ["챔피언", "충성", "잠재 충성", "신규·저가 최근", "관심 필요", "이탈 위험(고가치)", "휴면"]
km_order = ["핵심 우량", "중간 활성", "최근 소액", "이탈 위험"]
rows = []
for method, col, order in (("규칙 기반", "segment", rule_order), ("K-Means(k=4)", "kmeans", km_order)):
    for g in order + ["전체"]:
        d = base if g == "전체" else base[base[col] == g]
        r = {"method": method, "group": g, "customers": len(d)}
        for w in WINDOWS:
            k = int(d[f"rp{w}"].sum()); lo, hi = wilson(k, len(d))
            r.update({f"repurchased_{w}d": k, f"rate_{w}d_%": k / len(d) * 100, f"ci95_{w}d_low": lo, f"ci95_{w}d_high": hi})
        rep = d[d.rp90 == 1]
        r.update({"median_days_to_first_90d": rep.days_to_first.median(), "orders_90d_per_customer": d.orders_90.mean(),
                  "amount_90d_per_customer": d.amount_90.mean(), "amount_90d_total": d.amount_90.sum()})
        rows.append(r)
res = pd.DataFrame(rows)
res.round(2).to_csv("outputs/tables/repurchase_rates_by_group.csv", index=False, encoding="utf-8-sig")
base.drop(columns="cluster").to_csv("outputs/tables/repurchase_customers.csv", index=False, encoding="utf-8-sig")

# 방법별 구분력
cmp_rows = []
for method, col in (("규칙 기반", "segment"), ("K-Means(k=4)", "kmeans")):
    r = {"method": method, "groups": base[col].nunique()}
    for w in WINDOWS:
        y = base[f"rp{w}"]; gr = base.groupby(col)[f"rp{w}"].transform("mean")
        rates = base.groupby(col)[f"rp{w}"].mean() * 100
        r.update({f"AUC_{w}d": roc_auc_score(y, gr), f"eta2_{w}d": ((gr - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                  f"spread_{w}d_pp": rates.max() - rates.min()})
    cmp_rows.append(r)
cmp = pd.DataFrame(cmp_rows)
cmp.round(4).to_csv("outputs/tables/repurchase_method_comparison.csv", index=False, encoding="utf-8-sig")

# ---- 그림 ----
cols = ["#9ecae1", "#4292c6", "#08519c"]
fig, ax = plt.subplots(2, 2, figsize=(17, 10.5))
for j, (method, col, order) in enumerate((("규칙 기반", "segment", rule_order), ("K-Means(k=4)", "kmeans", km_order))):
    a = ax[0, j]; d = res[res.method == method].set_index("group").loc[order]; x = np.arange(len(order)); wd = .26
    for i, w in enumerate(WINDOWS):
        b = a.bar(x + (i - 1) * wd, d[f"rate_{w}d_%"], wd, label=f"{w}일", color=cols[i],
                  yerr=[d[f"rate_{w}d_%"] - d[f"ci95_{w}d_low"], d[f"ci95_{w}d_high"] - d[f"rate_{w}d_%"]], error_kw=dict(lw=.8, capsize=2))
        for bb, v in zip(b, d[f"rate_{w}d_%"]): a.text(bb.get_x() + bb.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    a.axhline(res[(res.method == method) & (res.group == "전체")]["rate_90d_%"].iloc[0], color="gray", ls=":", lw=1)
    a.set_xticks(x, [f"{g}\n(n={int(n)})" for g, n in zip(order, d.customers)], fontsize=9); a.set_ylim(0, 105)
    a.set_title(f"{method}: 30/60/90일 재구매율 (%, 막대 선: 95% 신뢰구간, 점선: 전체 90일)"); a.legend(); a.grid(axis="y", alpha=.3)
    a = ax[1, j]
    for g in order:
        dd = base[base[col] == g]; days = np.arange(0, 91)
        a.plot(days, [(dd.days_to_first.notna() & (dd.days_to_first < t + 1)).mean() * 100 for t in days], label=f"{g} (n={len(dd)})", lw=2)
    for w in WINDOWS: a.axvline(w, color="gray", ls=":", lw=.8)
    a.set_xlabel("기준일(2011-09-01) 이후 경과 일수"); a.set_ylabel("누적 재구매율 (%)"); a.set_title(f"{method}: 누적 재구매 곡선"); a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); fig.savefig("outputs/figures/repurchase_rates.png", dpi=130)

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
show = ["method", "group", "customers", "rate_30d_%", "rate_60d_%", "rate_90d_%", "ci95_90d_low", "ci95_90d_high",
        "median_days_to_first_90d", "orders_90d_per_customer", "amount_90d_per_customer", "amount_90d_total"]
print(res[show].round(1).to_string(index=False)); print(cmp.round(3).to_string(index=False))
print("K-Means 군집 이름:", nm)
print("교차: ", pd.crosstab(base.segment, base.kmeans).reindex(rule_order)[km_order].to_string())
