"""2011-09-19 주(월~일) 기준 이전 2주·이후 2주 판매량(수량)·판매 금액 그래프.

실행: python analysis/11_week_0919_window.py  (프로젝트 루트 기준, 01 먼저 실행)
대상: retail_clean.csv 구매 행(is_cancel=False). 판매량=수량 합계.
"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False

BASE = pd.Timestamp("2011-09-19")                       # 기준 주 시작(월요일)
assert BASE.weekday() == 0
df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
p = df[~df.is_cancel].copy()
p["date"] = p.invoice_date.dt.normalize()
p["week_start"] = p.date - pd.to_timedelta(p.date.dt.weekday, unit="D")
s = p[(p.week_start >= BASE - pd.Timedelta(weeks=2)) & (p.week_start <= BASE + pd.Timedelta(weeks=2))]
assert s.date.max() < BASE + pd.Timedelta(weeks=3)

wk = s.groupby("week_start").agg(quantity=("quantity", "sum"), sales=("line_amount", "sum"),
                                 orders=("invoice", "nunique"), customers=("customer_id", "nunique")).reset_index()
wk["rel"] = ((wk.week_start - BASE).dt.days // 7)
wk["label"] = wk.apply(lambda r: f"{r.week_start:%m-%d}~{r.week_start + pd.Timedelta(days=6):%m-%d}\n({'기준 주' if r.rel == 0 else (f'{r.rel:+d}주')})", axis=1)
wk["qty_wow"] = wk.quantity.diff(); wk["qty_wow_%"] = wk.quantity.pct_change() * 100
wk["sales_wow"] = wk.sales.diff()
day = s.groupby("date").agg(quantity=("quantity", "sum"), sales=("line_amount", "sum")).reindex(
    pd.date_range(BASE - pd.Timedelta(weeks=2), BASE + pd.Timedelta(days=6 + 14))).fillna(0)
assert wk.quantity.sum() == s.quantity.sum() and day.quantity.sum() == s.quantity.sum() and len(wk) == 5

fig, ax = plt.subplots(3, 1, figsize=(12, 13))
cols = ["#c9c9c9" if r != 0 else "#4f81bd" for r in wk.rel]
for a, c, t in [(ax[0], "quantity", "주간 판매량 (수량)"), (ax[1], "sales", "주간 판매 금액")]:
    b = a.bar(wk.label, wk[c], color=cols)
    for i, (bb, v) in enumerate(zip(b, wk[c])):
        chg = "" if i == 0 else f"\n({wk[c].pct_change().iloc[i] * 100:+.1f}%)"
        a.text(bb.get_x() + bb.get_width() / 2, v, f"{v:,.0f}{chg}", ha="center", va="bottom", fontsize=10)
    a.set_title(t + " - 2011-09-19 주 기준 전후 2주 (괄호: 전주 대비)"); a.set_ylim(0, wk[c].max() * 1.2)
    a.yaxis.set_major_formatter(lambda y, _: f"{y:,.0f}"); a.grid(axis="y", alpha=.3)
a = ax[2]; names = "월화수목금토일"
a.plot(day.index, day.quantity, marker="o", color="#c0504d")
a.axvspan(BASE, BASE + pd.Timedelta(days=7), color="#4f81bd", alpha=.15, label="기준 주 (09-19~09-25)")
for d, v in day.quantity.items():
    if v == 0: a.plot(d, 0, "x", color="gray")
a.set_xticks(day.index); a.set_xticklabels([f"{d:%m-%d}\n{names[d.weekday()]}" for d in day.index], fontsize=7)
a.set_title("일별 판매량 (수량, x = 판매 없음)"); a.legend(); a.grid(alpha=.3); a.yaxis.set_major_formatter(lambda y, _: f"{y:,.0f}")
plt.tight_layout(); fig.savefig("outputs/figures/week_0919_window.png", dpi=130)

out = wk.drop(columns="label").assign(week_end=lambda x: x.week_start + pd.Timedelta(days=6)).round(2)
out.to_csv("outputs/tables/week_0919_window.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print(out.to_string(index=False))
print(day.assign(wd=[names[d.weekday()] for d in day.index]).to_string())
