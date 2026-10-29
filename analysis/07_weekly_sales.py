"""주간(월~일) 판매 금액·주문 수·주문당 평균 금액, 전주 대비 판매 증가액이 가장 큰 주.

실행: python analysis/07_weekly_sales.py  (프로젝트 루트 기준, 01 먼저 실행)
대상: data/processed/retail_clean.csv 중 구매 행(is_cancel=False). 주 시작=월요일.
데이터 첫 주(2009-12-01 화)와 마지막 주(~2011-12-09 금)는 월~일이 모두 포함되지 않은 부분 주 -> 증가액 비교에서 제외.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import _plot_style  # 한글 글꼴·마이너스 기호 설정 (analysis/_plot_style.py)

df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
p = df[~df.is_cancel].copy()
p["week_start"] = p.invoice_date.dt.normalize() - pd.to_timedelta(p.invoice_date.dt.weekday, unit="D")
w = p.groupby("week_start").agg(sales=("line_amount", "sum"), orders=("invoice", "nunique")).reset_index()
w = w.set_index("week_start").reindex(pd.date_range(w.week_start.min(), w.week_start.max(), freq="7D")).rename_axis("week_start").reset_index()
w[["sales", "orders"]] = w[["sales", "orders"]].fillna(0)
w["aov"] = (w.sales / w.orders.replace(0, np.nan))
w["week_end"] = w.week_start + pd.Timedelta(days=6)
first_d, last_d = p.invoice_date.min().normalize(), p.invoice_date.max().normalize()
w["partial"] = (w.week_start < first_d) | (w.week_end > last_d)
w["wow_change"] = w.sales.diff()
w["prev_sales"] = w.sales.shift(1)
# 비교 가능한 주: 두 주 모두 완전한 주이고, 전주 판매가 0(연말 휴무)이 아닌 경우만 증가액 순위에 사용
w["wow_valid"] = w.wow_change.where(~w.partial & ~w.partial.shift(1, fill_value=True) & (w.prev_sales > 0))
w["week_start"] = w.week_start; w["week_end"] = w.week_end
assert abs(w.sales.sum() - p.line_amount.sum()) < 0.01 and w.orders.sum() == p.groupby(p.invoice).ngroups

best = w.loc[w.wow_valid.idxmax()]; prev = w.loc[w.wow_valid.idxmax() - 1]
worst = w.loc[w.wow_valid.idxmin()]
top3 = w.nlargest(5, "wow_valid")[["week_start", "week_end", "prev_sales", "sales", "wow_change", "orders"]]

fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
cols = [("sales", "판매 금액", "#4f81bd"), ("orders", "주문 수", "#9bbb59"), ("aov", "주문당 평균 금액", "#c0504d")]
for a, (c, t, col) in zip(ax, cols):
    full = w[~w.partial]; a.plot(full.week_start, full[c], color=col, lw=1.8, marker="o", ms=3)
    a.plot(w[w.partial].week_start, w[w.partial][c], "o", color="gray", ms=5, label="부분 주(월~일 미충족)")
    a.set_ylabel(t); a.set_title(f"주간 {t} (월~일)"); a.grid(alpha=.3)
ax[0].axvspan(best.week_start, best.week_end, color="gold", alpha=.5, label=f"전주 대비 증가액 최대 주: {best.week_start:%Y-%m-%d}~{best.week_end:%m-%d}")
ax[0].annotate(f"+{best.wow_change:,.0f}", (best.week_start, best.sales), xytext=(10, 12), textcoords="offset points", fontweight="bold")
ax[0].legend(loc="upper left"); ax[0].yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
ax[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout(); fig.savefig("outputs/figures/weekly_sales.png", dpi=130)

o = w.assign(week_start=w.week_start.dt.date, week_end=w.week_end.dt.date).round(2)
o.to_csv("outputs/tables/weekly_sales.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print("주 수:", len(w), "| 부분 주:", w[w.partial].week_start.dt.date.tolist(), "| 판매 0인 주:", int((w.sales == 0).sum()))
print(f"최대 증가: {best.week_start:%Y-%m-%d}~{best.week_end:%Y-%m-%d} 판매 {best.sales:,.2f} (전주 {prev.sales:,.2f}) 증가액 {best.wow_change:,.2f} "
      f"({best.wow_change/prev.sales*100:.1f}%) 주문 {best.orders:.0f} (전주 {prev.orders:.0f}) AOV {best.aov:.2f} (전주 {prev.aov:.2f})")
print("최대 감소:", f"{worst.week_start:%Y-%m-%d} {worst.wow_valid:,.2f}"); print(top3.round(2).to_string(index=False))
print(w[["sales", "orders", "aov"]].describe().round(2).to_string())
print(w[w.sales == 0].week_start.dt.date.tolist(), "| 판매 최대 주:", w.loc[w.sales.idxmax(), ["week_start", "sales"]].tolist())
print(w.loc[best.name - 3:best.name + 2, ["week_start", "sales", "orders", "aov"]].round(1).to_string(index=False))
