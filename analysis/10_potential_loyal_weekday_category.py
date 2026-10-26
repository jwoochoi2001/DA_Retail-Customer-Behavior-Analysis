"""잠재 충성 고객의 요일별 지표 + 상품 카테고리별 지표.

실행: python analysis/10_potential_loyal_weekday_category.py  (프로젝트 루트 기준, 02 먼저 실행)
잠재 충성 고객 = 02 군집 중 최근성이 높고(R 평균 30일 이하) 구매금액이 핵심 우량보다 낮은 군집(최근·소액 구매, 재구매 여지).
대상 기간·구매 기준은 02와 동일(2011-03-05~08-31, 취소 제외).
상품 카테고리: 원본에 카테고리 열이 없어 상품명(Description)의 키워드로 추정한 분류(아래 CATS, 위에서부터 먼저 맞는 규칙 적용).
"""
import re
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False

REF = pd.Timestamp("2011-09-01"); START = REF - pd.Timedelta(days=180)
CATS = [  # (카테고리, 키워드 정규식) - 순서가 우선순위
    ("크리스마스·시즌", r"CHRISTMAS|XMAS|EASTER|HALLOWEEN|SANTA|SNOWMAN|REINDEER|ADVENT|NATIVITY|SNOWFLAKE|STOCKING"),
    ("조명·캔들", r"LIGHT|LANTERN|CANDLE|LAMP|T-LIGHT|TEALIGHT|NIGHTLIGHT|INCENSE|WARMER|LAMPSHADE"),
    ("가방·지갑", r"\bBAG\b|\bBAGS\b|PURSE|WALLET|SHOPPER|SATCHEL|RUCKSACK|BACKPACK|POUCH"),
    ("주방·식기", r"CAKE|MUG|CUP|PLATE|BOWL|TEAPOT|\bTEA\b|COFFEE|LUNCH|CUTLERY|SPOON|FORK|\bJAM\b|KITCHEN|APRON|BAKING|BISCUIT|\bJUG\b|PICNIC|SPICE|DISH|TRAY|COASTER|GLASS|TUMBLER|CORKSCREW|BOTTLE OPENER|EGG|PANTRY|TEASET|SAUCER|PLACEMAT|TEA TOWEL|OVEN"),
    ("수납·정리", r"\bBOX\b|BOXES|\bTIN\b|\bTINS\b|STORAGE|\bJAR\b|JARS|BASKET|DRAWER|CABINET|TRUNK|CRATE|ORGANI[SZ]ER|CASE|HOLDER|RACK|SHELF|HOOK|HANGER|BIN\b"),
    ("정원·야외", r"GARDEN|PLANT|FLOWER|PARASOL|BIRD|WATERING|PLANTER|SEED|HERB|POTS|OUTDOOR|WINDMILL|FEEDER|BUCKET|SPADE|TROWEL"),
    ("장난감·게임", r"\bTOY|GAME|DOLL|PUZZLE|PLAYING|\bBALL\b|SPACEBOY|ROBOT|DOMINO|SKIPPING|YOYO|KIT\b|SKITTLE|CRAYON|COLOURING|BUBBLE"),
    ("패션·주얼리", r"NECKLACE|BRACELET|EARRING|JEWEL|\bRING\b|HAIR|SCARF|BROOCH|BEAD|BANGLE|PENDANT|CHARM|SUNGLASS|SLIPPER|FLIP FLOP|GLOVE"),
    ("문구·카드·포장", r"WRAP|CARD|PARTY|BIRTHDAY|BUNTING|BALLOON|NAPKIN|CRACKER|RIBBON|TISSUE|PAPER|STICKER|TAG\b|PENCIL|NOTEBOOK|ERASER|\bPEN\b|BOOK|STATIONERY|ENVELOPE|GIFT|BADGE|CONFETTI|STAMP|RUBBER"),
    ("패브릭·생활", r"HOT WATER|BLANKET|TOWEL|CURTAIN|QUILT|PILLOW|COVER|CUSHION|DOORMAT|RUG\b|HANDKERCHIEF|SOAP|SPONGE|BATH|DOORSTOP|FELT"),
    ("홈데코·인테리어", r"FRAME|CLOCK|SIGN|WALL|MIRROR|HANGING|HEART|DECORATION|DOOR|VASE|WREATH|MOBILE|STAR|ORNAMENT|FIGURE|CHIME|SUNCATCHER|PLAQUE|WINDOW|CHAIN|STAND|CERAMIC|METAL|WOODEN|WOOD|GLASS|DECO"),
]
def categorize(desc):
    d = str(desc).upper()
    for name, pat in CATS:
        if re.search(pat, d): return name
    return "기타"

rfm = pd.read_csv("outputs/tables/rfm_customers.csv")
cs = rfm.groupby("cluster").agg(R=("R", "mean"), M=("M", "mean"))
cand = cs[cs.R <= 30]; cand = cand.drop(cand.M.idxmax()) if len(cand) > 1 else cand
pl = cand.M.idxmin(); ids = rfm[rfm.cluster == pl].customer_id
df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["invoice_date"])
w = df[(df.invoice_date >= START) & (df.invoice_date < REF) & (~df.is_cancel)].copy()
s = w[w.customer_id.isin(ids)].copy()
assert s.customer_id.nunique() == len(ids)

# ---- 요일별 ----
s["date"] = s.invoice_date.dt.normalize(); s["wd"] = s.invoice_date.dt.weekday
names = ["월", "화", "수", "목", "금", "토", "일"]
bdays = w.assign(d=w.invoice_date.dt.normalize(), wd=w.invoice_date.dt.weekday).groupby("wd").d.nunique().reindex(range(7)).fillna(0)
wd = s.groupby("wd").agg(sales=("line_amount", "sum"), orders=("invoice", "nunique"), customers=("customer_id", "nunique")).reindex(range(7)).fillna(0)
wd["aov"] = wd.sales / wd.orders.replace(0, float("nan"))
wd["sales_share_%"] = wd.sales / wd.sales.sum() * 100; wd["orders_share_%"] = wd.orders / wd.orders.sum() * 100
wd["business_days"] = bdays; wd["sales_per_day"] = wd.sales / bdays.replace(0, float("nan"))
wd.index = pd.Index(names, name="weekday")
assert abs(wd.sales.sum() - s.line_amount.sum()) < .01 and wd.orders.sum() == s.invoice.nunique()
wd.round(2).to_csv("outputs/tables/potential_loyal_weekday.csv", encoding="utf-8-sig")

# ---- 카테고리별 ----
cat_of = w.groupby("stock_code").description.agg(lambda x: x.mode().iat[0]).map(categorize)
for d_ in (s, w): d_["category"] = d_.stock_code.map(cat_of)
g = s.groupby("category").agg(sales=("line_amount", "sum"), quantity=("quantity", "sum"), orders=("invoice", "nunique"),
                              customers=("customer_id", "nunique"), items=("stock_code", "nunique"))
g["sales_share_%"] = g.sales / g.sales.sum() * 100; g["avg_unit_price"] = g.sales / g.quantity
g["aov_in_category"] = g.sales / g.orders           # 해당 카테고리가 포함된 주문당 카테고리 소계
g["customer_reach_%"] = g.customers / s.customer_id.nunique() * 100
g["all_customers_share_%"] = w.groupby("category").line_amount.sum() / w.line_amount.sum() * 100
g["share_diff_pp"] = g["sales_share_%"] - g["all_customers_share_%"]
g = g.sort_values("sales", ascending=False)
assert abs(g.sales.sum() - s.line_amount.sum()) < .01
g.round(2).to_csv("outputs/tables/potential_loyal_category.csv", encoding="utf-8-sig")
# 분류 점검용: 전체 구매 기준 카테고리 커버리지와 기타 상위 상품
other = w[w.category == "기타"].groupby("description").line_amount.sum().sort_values(ascending=False)

# ---- 그림 ----
def bars(a, x, v, title, color, fmt, sh=None, dec=0):
    b = a.bar(x, v.fillna(0), color=color)
    for i, (bb, val) in enumerate(zip(b, v.fillna(0))):
        t = "-" if val == 0 else f"{val:,.{dec}f}" + (f"\n({sh.iloc[i]:.1f}%)" if sh is not None else "")
        a.text(bb.get_x() + bb.get_width() / 2, val, t, ha="center", va="bottom", fontsize=9)
    a.set_title(title); a.set_ylim(0, max(v.max(), 1) * 1.2); a.yaxis.set_major_formatter(lambda y, _: f"{y:,.0f}"); a.grid(axis="y", alpha=.3)
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
bars(ax[0, 0], names, wd.sales, "요일별 판매 금액", "#4f81bd", "", wd["sales_share_%"])
bars(ax[0, 1], names, wd.orders, "요일별 주문 수", "#9bbb59", "", wd["orders_share_%"])
bars(ax[1, 0], names, wd.aov, "요일별 주문당 평균 금액", "#c0504d", "")
bars(ax[1, 1], names, wd.sales_per_day, "요일별 영업일당 판매 금액", "#8064a2", "")
fig.suptitle(f"잠재 충성 고객 {len(ids):,}명 요일별 지표 ({START.date()}~{(REF - pd.Timedelta(days=1)).date()})", fontsize=13)
plt.tight_layout(); fig.savefig("outputs/figures/potential_loyal_weekday.png", dpi=130)

fig, ax = plt.subplots(2, 2, figsize=(15, 10)); c = g.index.tolist()
a = ax[0, 0]; y = range(len(c))
a.barh([i + .2 for i in y], g["sales_share_%"], .4, label="잠재 충성", color="#4f81bd")
a.barh([i - .2 for i in y], g["all_customers_share_%"], .4, label="전체 고객", color="#bfbfbf")
a.set_yticks(list(y), c); a.invert_yaxis(); a.set_title("카테고리별 판매 금액 비중 (%)"); a.legend(); a.grid(axis="x", alpha=.3)
for i, v in enumerate(g["sales_share_%"]): a.text(v, i + .2, f" {v:.1f}", va="center", fontsize=8)
def hb(a, v, t, col, dec=0):
    a.barh(c, v, color=col); a.invert_yaxis(); a.set_title(t); a.grid(axis="x", alpha=.3)
    for i, val in enumerate(v): a.text(val, i, f" {val:,.{dec}f}", va="center", fontsize=8)
hb(ax[0, 1], g.orders, "카테고리 포함 주문 수", "#9bbb59")
hb(ax[1, 0], g.aov_in_category, "카테고리 포함 주문당 평균 금액", "#c0504d")
hb(ax[1, 1], g.avg_unit_price, "평균 단가 (판매금액 ÷ 수량)", "#8064a2", 2)
fig.suptitle(f"잠재 충성 고객 상품 카테고리별 지표 (키워드 기반 추정 분류)", fontsize=13)
plt.tight_layout(); fig.savefig("outputs/figures/potential_loyal_category.png", dpi=130)

pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20)
print("잠재 충성 군집:", pl, cs.loc[pl].round(1).to_dict(), "| 고객", len(ids), "| 주문", s.invoice.nunique(), "| 판매", round(s.line_amount.sum(), 2))
print(wd.round(2).to_string()); print(g.round(2).to_string())
print("전체(기간 내 모든 고객) 기타 비중 %:", round(w[w.category == '기타'].line_amount.sum() / w.line_amount.sum() * 100, 1),
      "| 상품 수 기준 분류:", cat_of.value_counts().to_dict())
print(other.head(25).round(0).to_string())
