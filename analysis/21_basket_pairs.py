"""같은 주문에서 함께 구매된 상품 쌍 상위 10개 (함께 산 주문 수, 전체 주문 대비 비율, 지지도·신뢰도·리프트).

실행: python analysis/21_basket_pairs.py  (프로젝트 루트 기준, 01 먼저 실행)
- 대상: 전처리 데이터(retail_clean.csv) 중 구매 주문(is_cancel=False). 주문 번호(invoice) 하나가 장바구니 하나.
- 상품 = stock_code. 한 주문에 같은 상품이 여러 번 나와도 1번으로 센다. 쌍은 순서 없는 조합(A,B = B,A).
- 비율(%) = 두 상품이 함께 들어간 주문 수 / 전체 주문 수.
"""
import numpy as np, pandas as pd, matplotlib
from scipy import sparse
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _plot_style  # 한글 글꼴·마이너스 기호 설정 (analysis/_plot_style.py)

df = pd.read_csv("data/processed/retail_clean.csv")
b = df[~df.is_cancel][["invoice", "stock_code"]].drop_duplicates()
name = df.groupby("stock_code").description.agg(lambda s: s.mode().iat[0])
inv_codes, inv = pd.factorize(b.invoice); sc_codes, sc = pd.factorize(b.stock_code)
X = sparse.csr_matrix((np.ones(len(b), dtype=np.int32), (inv_codes, sc_codes)), shape=(len(inv), len(sc)))
n_orders = X.shape[0]
C = (X.T @ X).tocoo()
item_orders = np.asarray(X.sum(0)).ravel()
m = C.row < C.col
pairs = pd.DataFrame({"a": sc[C.row[m]], "b": sc[C.col[m]], "orders_together": C.data[m]})
pairs["orders_a"] = item_orders[C.row[m]]; pairs["orders_b"] = item_orders[C.col[m]]
pairs["share_of_all_orders_%"] = pairs.orders_together / n_orders * 100
pairs["confidence_a_to_b_%"] = pairs.orders_together / pairs.orders_a * 100
pairs["confidence_b_to_a_%"] = pairs.orders_together / pairs.orders_b * 100
pairs["lift"] = pairs.orders_together * n_orders / (pairs.orders_a * pairs.orders_b)
pairs["desc_a"] = pairs.a.map(name); pairs["desc_b"] = pairs.b.map(name)
pairs = pairs.sort_values(["orders_together", "lift"], ascending=False).reset_index(drop=True)
pairs.insert(0, "rank", np.arange(1, len(pairs) + 1))
# 검증: 상위 쌍 하나를 원본 주문 집합 교집합으로 직접 재계산
t = pairs.iloc[0]
chk = len(set(b[b.stock_code == t.a].invoice) & set(b[b.stock_code == t.b].invoice))
assert chk == t.orders_together and abs(item_orders[list(sc).index(t.a)] - len(set(b[b.stock_code == t.a].invoice))) == 0
pairs.head(500).round(3).to_csv("outputs/tables/basket_pairs_top500.csv", index=False, encoding="utf-8-sig")
top = pairs.head(10)

fig, ax = plt.subplots(figsize=(14, 6.5)); y = np.arange(10)[::-1]
ax.barh(y, top.orders_together, color="#4f81bd")
for yy, o, s, l in zip(y, top.orders_together, top["share_of_all_orders_%"], top.lift):
    ax.text(o, yy, f"  {o:,}건 ({s:.2f}%)  lift {l:.1f}", va="center", fontsize=9)
ax.set_yticks(y, [f"{r}. {a[:24]} + {b_[:24]}" for r, a, b_ in zip(top["rank"], top.desc_a, top.desc_b)], fontsize=9)
ax.set_xlim(0, top.orders_together.max() * 1.3); ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
ax.set_xlabel("함께 구매된 주문 수"); ax.grid(axis="x", alpha=.3)
ax.set_title(f"함께 구매된 상품 쌍 상위 10개 (전체 주문 {n_orders:,}건 기준 비율)")
plt.tight_layout(); fig.savefig("outputs/figures/basket_pairs_top10.png", dpi=130)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 40)
print("전체 주문", n_orders, "| 상품", len(sc), "| 함께 구매된 쌍 수", len(pairs))
print(top[["rank", "a", "desc_a", "b", "desc_b", "orders_together", "share_of_all_orders_%", "orders_a", "orders_b", "confidence_a_to_b_%", "confidence_b_to_a_%", "lift"]].round(2).to_string(index=False))
print("전체 주문 중 상품 2개 이상 주문 %:", round((np.asarray(X.sum(1)).ravel() >= 2).mean() * 100, 1), "| 쌍 최대 orders_together:", pairs.orders_together.max())
