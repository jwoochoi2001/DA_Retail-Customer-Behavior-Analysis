"""그래프 한글 글꼴 설정. 분석 스크립트가 `import _plot_style` 한 줄로 불러온다.

설치된 글꼴 중 Windows(Malgun Gothic) -> macOS(AppleGothic) -> Linux(NanumGothic, Noto Sans CJK KR) 순으로 첫 번째를 사용하고,
없으면 DejaVu Sans(한글은 네모로 표시됨)로 대체한다.
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CANDIDATES = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
_installed = {f.name for f in font_manager.fontManager.ttflist}
_font = next((f for f in _CANDIDATES if f in _installed), "DejaVu Sans")
plt.rcParams["font.family"] = _font
plt.rcParams["axes.unicode_minus"] = False
if _font == "DejaVu Sans":
    print("경고: 한글 글꼴을 찾지 못했습니다. 그래프의 한글이 깨질 수 있습니다(NanumGothic 등 설치 권장).")
