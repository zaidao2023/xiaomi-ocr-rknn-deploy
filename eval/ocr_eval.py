#!/usr/bin/env python3
"""OCR accuracy evaluation framework for Xiaomi-OCR-0 on-device deployment.

Scores on-device OCR output against ground-truth text with character-level
metrics: CER (character error rate), edit distance and exact match.

Two modes:
  from-file : score captured model outputs against ground truth (offline)
  run       : drive the on-board demo over SSH, capture output, score

Usage:
  # offline, validate against captured outputs
  python ocr_eval.py --mode from-file --cases test_cases.json --fixtures fixtures

  # on-board
  python ocr_eval.py --mode run --cases test_cases.json \
      --ssh root@10.62.9.139 --workdir /root/xiaomi_ocr
"""
import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass, asdict


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance (substitutions + insertions + deletions)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize(text: str, lower: bool = False, collapse_ws: bool = True) -> str:
    """Normalize text for fair OCR comparison."""
    if lower:
        text = text.lower()
    if collapse_ws:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_code_fences(text: str) -> str:
    """Remove markdown/latex code fences the model emits around its answer."""
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```", "", text)
    return text.strip()


def parse_demo_output(raw: str) -> str:
    """Extract the model answer from demo 'user: robot: ...' framing.

    The demo prints the generated text after a 'robot:' marker. Take the first
    complete answer block, strip code fences.
    """
    if "robot:" in raw:
        raw = raw.split("robot:", 1)[1]
    return strip_code_fences(raw)


def cer(pred: str, gt: str) -> float:
    """Character error rate: edit_distance / len(gt). Lower is better."""
    if not gt:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, gt) / len(gt)


@dataclass
class CaseResult:
    name: str
    gt_chars: int
    pred_chars: int
    edit_distance: int
    cer: float
    accuracy: float
    exact_match: bool
    pred_snippet: str
    gt_snippet: str


@dataclass
class Case:
    name: str
    image: str
    prompt: str
    gt_file: str
    pred_file: str = ""


def load_cases(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Case(**c) for c in data]


def score_case(case: Case, pred_text: str, normalize_kwargs=None) -> CaseResult:
    nk = normalize_kwargs or {}
    gt_raw = _read(case.gt_file)
    pred_raw = parse_demo_output(pred_text)
    gt_n = normalize(gt_raw, **nk)
    pred_n = normalize(pred_raw, **nk)
    ed = edit_distance(pred_n, gt_n)
    c = cer(pred_n, gt_n)
    return CaseResult(
        name=case.name,
        gt_chars=len(gt_n),
        pred_chars=len(pred_n),
        edit_distance=ed,
        cer=round(c, 4),
        accuracy=round(1 - c, 4),
        exact_match=(pred_n == gt_n),
        pred_snippet=pred_n[:120],
        gt_snippet=gt_n[:120],
    )


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def run_demo(ssh_host, workdir, image, vision_model, llm_model,
             prompt, max_new_tokens=256, max_context=1024, core=1, platform="rv1126b"):
    """Drive the on-board demo non-interactively and return raw output."""
    cmd = (
        f"cd {workdir} && export LD_LIBRARY_PATH={workdir}/lib && "
        f"printf '%s\\nexit\\n' \"{prompt}\" | "
        f"./bin/demo {image} {vision_model} \"\" \"\" {llm_model} "
        f"{max_new_tokens} {max_context} {core} {platform}"
    )
    out = subprocess.run(["ssh", ssh_host, cmd], capture_output=True, text=True, timeout=600)
    return out.stdout + out.stderr


def aggregate(results: list) -> dict:
    if not results:
        return {}
    n = len(results)
    mean_cer = sum(r.cer for r in results) / n
    mean_acc = sum(r.accuracy for r in results) / n
    return {
        "cases": n,
        "mean_cer": round(mean_cer, 4),
        "mean_accuracy": round(mean_acc, 4),
        "exact_matches": sum(r.exact_match for r in results),
    }


def print_report(results: list, agg: dict):
    print("=" * 72)
    print(f"{'case':<16}{'gt':>7}{'pred':>7}{'edit':>7}{'CER':>9}{'acc':>9}{'exact':>7}")
    print("-" * 72)
    for r in results:
        print(f"{r.name:<16}{r.gt_chars:>7}{r.pred_chars:>7}{r.edit_distance:>7}"
              f"{r.cer:>9}{r.accuracy:>9}{str(r.exact_match):>7}")
    print("-" * 72)
    print(f"{'AGGREGATE':<16}{'':>7}{'':>7}{'':>7}{agg['mean_cer']:>9}"
          f"{agg['mean_accuracy']:>9}{agg['exact_matches']:>7}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description="OCR accuracy evaluation framework")
    ap.add_argument("--mode", choices=["from-file", "run"], default="from-file")
    ap.add_argument("--cases", required=True, help="test cases json")
    ap.add_argument("--fixtures", default="fixtures", help="dir with captured pred files (from-file)")
    ap.add_argument("--ssh", help="board ssh host (run mode)")
    ap.add_argument("--workdir", default="/root/xiaomi_ocr", help="demo workdir on board (run mode)")
    ap.add_argument("--vision", default="models/qwen3.5_vision_rv1126b.rknn")
    ap.add_argument("--llm", default="models/xiaomi-ocr-0_w8a8_rv1126b.rkllm")
    ap.add_argument("--lower", action="store_true", help="case-insensitive normalization")
    ap.add_argument("--out", help="write json report to this path")
    args = ap.parse_args()

    cases = load_cases(args.cases)
    nk = {"lower": args.lower}
    results = []

    for case in cases:
        if args.mode == "from-file":
            pred_file = os.path.join(args.fixtures, case.pred_file or f"{case.name}.pred.txt")
            if not os.path.exists(pred_file):
                print(f"[{case.name}] no captured output at {pred_file}, skip")
                continue
            pred_text = _read(pred_file)
        else:
            if not args.ssh:
                raise SystemExit("--ssh required in run mode")
            pred_text = run_demo(args.ssh, args.workdir, case.image, args.vision,
                                 args.llm, case.prompt)
        r = score_case(case, pred_text, nk)
        results.append(r)
        print(f"[{case.name}] CER={r.cer} acc={r.accuracy}")

    agg = aggregate(results)
    print_report(results, agg)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"aggregate": agg, "results": [asdict(r) for r in results]},
                      f, ensure_ascii=False, indent=2)
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
