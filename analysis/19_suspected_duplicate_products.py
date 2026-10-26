"""상품 코드·상품명을 검토해 '같은 상품인데 구성만 바뀌어 다른 코드가 된' 의심 쌍을 찾는다.

실행: python analysis/19_suspected_duplicate_products.py  (프로젝트 루트 기준, 01 먼저 실행)
방법
1) 코드별 대표 상품명(가장 많이 쓰인 Description)을 정규화(대문자, 특수문자 제거, 공백 정리).
2) 문자 n-gram TF-IDF 코사인 유사도 >= 0.85인 서로 다른 코드 쌍을 후보로 수집.
3) 두 이름의 토큰 차이로 유형 분류: 동일 / 숫자만 다름(수량·구성) / 색상만 다름 / 기타 표현 차이.
4) 근거 점수: 이름 동일(+3), 숫자만 다름(+2), 기타 차이이며 유사도>=0.9(+1), 판매 기간이 겹치지 않는 교체형(+2, 색상 쌍은 +1),
   같은 코드 숫자부(접미 문자만 다름)(+1), 단가 비율 0.8~1.25(+1). 점수 5 이상을 '강한 의심'으로 분류.
색상만 다른 쌍은 보통 정상 변형 상품(SKU)이라 점수에서 가산하지 않음(교체형일 때만 +1).
"""
import re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
buy = df[~df.is_cancel]
g = buy.groupby("stock_code").agg(sales=("line_amount", "sum"), qty=("quantity", "sum"), orders=("invoice", "nunique"),
                                  first=("invoice_date", "min"), last=("invoice_date", "max"))
g["price"] = g.sales / g.qty
med_price = buy.groupby("stock_code").price.median()
g["description"] = df.groupby("stock_code").description.agg(lambda s: s.mode().iat[0])
g["n_desc"] = df.groupby("stock_code").description.nunique()


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", str(s).upper())).strip()


g["norm"] = g.description.map(norm)
g["base"] = g.index.to_series().str.replace(r"[A-Z]+$", "", regex=True)
COLORS = set("RED BLUE GREEN PINK YELLOW WHITE BLACK PURPLE ORANGE IVORY CREAM SILVER GOLD BROWN GREY GRAY TURQUOISE LILAC "
             "PEACH CLEAR AQUA MAGENTA CHARTREUSE LIME MINT BEIGE CORAL TEAL NAVY".split())

X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(g.norm)
S = (X @ X.T).toarray(); np.fill_diagonal(S, 0)
ii, jj = np.where(np.triu(S >= 0.85, 1))

toks = {c: set(n.split()) for c, n in g.norm.items()}
codes = g.index.to_numpy()
rows = []
for i, j in zip(ii, jj):
    a, b = codes[i], codes[j]
    d = toks[a] ^ toks[b]
    if not d: typ = "설명 동일"
    elif all(t.isdigit() for t in d): typ = "숫자만 다름(수량·구성)"
    elif all(t in COLORS for t in d): typ = "색상만 다름"
    elif all(t.isdigit() or t in COLORS for t in d): typ = "숫자·색상만 다름"
    else: typ = "기타 표현 차이"
    ga, gb = g.loc[a], g.loc[b]
    first, last = (ga["first"], ga["last"]), (gb["first"], gb["last"])
    overlap = (min(first[1], last[1]) - max(first[0], last[0])).days
    old, new = (a, b) if first[0] <= last[0] else (b, a)            # 먼저 팔린 코드 / 나중에 팔린 코드
    gap_days = (g.loc[new, "first"] - g.loc[old, "last"]).days      # 음수면 판매 기간이 겹침
    succ = gap_days >= -30                                           # 겹침이 30일 이하이면 교체형
    ratio = med_price[a] / med_price[b]
    score = {"설명 동일": 3, "숫자만 다름(수량·구성)": 2, "기타 표현 차이": 1 if S[i, j] >= 0.9 else 0}.get(typ, 0)
    score += (1 if typ in ("색상만 다름", "숫자·색상만 다름") else 2) if succ else 0
    score += 1 if g.loc[a, "base"] == g.loc[b, "base"] else 0
    score += 1 if 0.8 <= ratio <= 1.25 else 0
    rows.append(dict(code_a=a, desc_a=ga.description, code_b=b, desc_b=gb.description, similarity=round(S[i, j], 3), type=typ,
                     diff_tokens=" ".join(sorted(d)), old_code=old, new_code=new, old_last=g.loc[old, "last"].date(),
                     new_first=g.loc[new, "first"].date(), price_old=round(med_price[old], 2), price_new=round(med_price[new], 2),
                     pattern="교체형(순차 판매)" if succ else "병행 판매(동시 판매)", gap_days=gap_days, replace_pattern=succ,
                     price_ratio=round(ratio, 2), sales_a=round(ga.sales), sales_b=round(gb.sales), same_base=g.loc[a, "base"] == g.loc[b, "base"],
                     score=score))
pairs = pd.DataFrame(rows).sort_values(["score", "similarity"], ascending=False)
pairs["suspicion"] = np.where(pairs.score >= 5, "강한 의심", np.where(pairs.score >= 3, "의심", "참고"))
pairs.to_csv("outputs/tables/suspected_duplicate_products.csv", index=False, encoding="utf-8-sig")

# 코드 하나에 상품명이 여러 개인 경우(같은 코드의 이름 변경)
multi = df.groupby("stock_code").description.nunique(); multi = multi[multi > 1]
multi_desc = (df[df.stock_code.isin(multi.index)].groupby(["stock_code", "description"]).line_amount.agg(["size", "sum"]).reset_index())
multi_desc.round(2).to_csv("outputs/tables/product_code_multiple_descriptions.csv", index=False, encoding="utf-8-sig")

# 이름 동일 코드군을 합치면 판매 상위 10개가 바뀌는지
dup_groups = g[g.norm.duplicated(keep=False)]
merged = g.groupby("norm").sales.sum().sort_values(ascending=False)
top_orig = g.sales.nlargest(10).index.map(g.norm).tolist(); top_merged = merged.head(10).index.tolist()

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 38)
print("상품 코드", len(g), "| 후보 쌍(유사도>=0.85)", len(pairs), "| 코드 하나에 이름 여러 개:", len(multi))
print(pairs.groupby(["type"]).size().to_string()); print(pairs.groupby(["suspicion", "type"]).size().unstack(fill_value=0).to_string())
print("이름 동일 코드군:", dup_groups.norm.nunique(), "그룹 /", len(dup_groups), "코드 | 판매 금액", round(dup_groups.sales.sum()),
      f"({dup_groups.sales.sum() / g.sales.sum() * 100:.2f}%)")
cols = ["code_a", "desc_a", "code_b", "desc_b", "type", "diff_tokens", "gap_days", "price_ratio", "sales_a", "sales_b", "score"]
print("\n[강한 의심 상위]"); print(pairs[pairs.suspicion == "강한 의심"][cols].head(30).to_string(index=False))
print("\n[교체형 (판매 기간 비겹침, 색상 제외)]")
rp = pairs[(pairs.replace_pattern) & (~pairs.type.isin(["색상만 다름"]))].copy(); rp["tot"] = rp.sales_a + rp.sales_b
print(rp.sort_values("tot", ascending=False)[cols].head(15).to_string(index=False))
print("\n판매 상위 10 정규화명 동일:", top_orig == top_merged, "| 병합 후 상위 10:"); print(merged.head(10).round(0).to_string())
print("코드 하나에 이름 여러 개 상위(판매 기준):")
tm = multi_desc.groupby("stock_code")["sum"].sum().sort_values(ascending=False).head(6).index
print(multi_desc[multi_desc.stock_code.isin(tm)].sort_values(["stock_code", "sum"], ascending=[True, False]).round(0).to_string(index=False))
