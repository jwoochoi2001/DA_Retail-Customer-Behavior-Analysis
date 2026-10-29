"""analysis/ 의 스크립트를 번호 순서대로 모두 실행한다. (프로젝트 루트에서 실행)

사용법
  python run_all.py              # 01 ~ 23 전체 실행
  python run_all.py --from 15    # 15번부터 실행(앞 단계 결과 파일이 이미 있을 때)
  python run_all.py --only 01 02 # 지정한 번호만 실행
첫 단계(01)는 data/raw/online_retail_II.xlsx 를 읽으므로 약 1~3분 걸린다. 한 단계라도 실패하면 즉시 멈춘다.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

root = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", type=int, default=1)
ap.add_argument("--only", nargs="*")
args = ap.parse_args()

scripts = sorted(p for p in (root / "analysis").glob("[0-9][0-9]_*.py"))
if args.only:
    scripts = [p for p in scripts if p.name[:2] in args.only]
else:
    scripts = [p for p in scripts if int(p.name[:2]) >= args.start]

env = {**os.environ, "PYTHONUTF8": "1"}
t0 = time.time()
for p in scripts:
    t = time.time()
    print(f"[실행] {p.name}", flush=True)
    r = subprocess.run([sys.executable, str(p)], cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"[실패] {p.name}\n{r.stderr[-1500:]}")
        sys.exit(r.returncode)
    print(f"[완료] {p.name} ({time.time() - t:.0f}초)", flush=True)
print(f"전체 완료: {len(scripts)}개 스크립트, {time.time() - t0:.0f}초")
