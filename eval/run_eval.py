"""
Clinical RAG evaluation and quality gate.

    python -m eval.run_eval                 # full run, writes reports/, exits 1 if a gate fails
    python -m eval.run_eval --limit 10      # quick smoke run
    python -m eval.run_eval --patient P002  # one patient

Metrics
  status_accuracy        pipeline returned the expected status (answered / not_documented / out_of_scope)
  refusal_accuracy       out-of-scope questions (advice, other patient, off-topic) were refused
  retrieval_hit_rate     an expected source chunk was retrieved (hit@k)
  retrieval_mrr          mean reciprocal rank of the first expected source
  keyword_recall         answer contains every required fact (deterministic string check)
  answer_correctness     LLM judge vs. the clinician-written reference answer
  faithfulness           share of claims the verifier found supported by the sources
  citation_coverage      share of factual sentences carrying a citation
  cross_patient_leakage  retrieved chunks belonging to another patient (must be 0)
"""
import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from chartrag import config, llm
from chartrag.pipeline import ChartAssistant
from chartrag.prompts import JUDGE_SYSTEM

ROOT = Path(__file__).resolve().parent.parent

GATES = {  # metric: (minimum, maximum). Tune deliberately and record why in the PR.
    "status_accuracy": (0.90, None),
    "refusal_accuracy": (0.875, None),
    "retrieval_hit_rate": (0.85, None),
    "keyword_recall": (0.85, None),
    "answer_correctness": (0.80, None),
    "faithfulness": (0.90, None),
    "cross_patient_leakage": (None, 0),
}


class Judgement(BaseModel):
    correct: bool
    explanation: str = ""


def judge(question, reference, candidate) -> Judgement | None:
    try:
        return llm.chat_json([
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nReference answer: {reference}\nCandidate answer: {candidate}"},
        ], Judgement, model=config.JUDGE_MODEL)
    except Exception:
        return None


def contains_all(answer: str, required: list[str]) -> bool:
    text = answer.lower()
    return all(any(alt.strip().lower() in text for alt in item.split("|")) for item in required)


def run_item(assistant: ChartAssistant, item: dict) -> dict:
    result = assistant.ask(item["patient_id"], item["question"], item.get("history"))
    row = {
        "id": item["id"], "patient_id": item["patient_id"], "category": item["category"],
        "question": item["question"], "expected_status": item["expected_status"],
        "status": result.status, "reason": result.reason, "answer": result.answer,
        "status_ok": result.status == item["expected_status"],
        "retrieved": [c.id for c in result.retrieved],
        "cited": [c.id for c in result.cited],
        "citation_coverage": result.citation_coverage,
        "latency_s": result.latency_s,
        "leaked": sum(1 for c in result.retrieved if c.patient_id != item["patient_id"]) + result.blocked_cross_patient,
    }
    expected = item.get("expected_sources")
    if expected and result.retrieved:
        ranks = [i for i, c in enumerate(result.retrieved, 1) if c.id in expected]
        row["retrieval_hit"] = bool(ranks)
        row["reciprocal_rank"] = 1.0 / ranks[0] if ranks else 0.0
    elif expected:
        row["retrieval_hit"], row["reciprocal_rank"] = False, 0.0

    if item["expected_status"] == "answered":
        answered = result.status == "answered"
        if item.get("must_include"):
            row["keywords_ok"] = answered and contains_all(result.answer, item["must_include"])
        if item.get("reference_answer"):
            j = judge(item["question"], item["reference_answer"], result.answer) if answered else None
            row["correct"] = bool(j and j.correct)
            row["judge_note"] = j.explanation if j else ("not answered" if not answered else "judge error")
        if answered and result.verification and result.verification.get("score") is not None:
            row["faithfulness"] = result.verification["score"]
            row["unsupported_claims"] = result.verification["unsupported"]
    return row


def mean(values):
    values = [float(v) for v in values]
    return round(statistics.mean(values), 3) if values else None


def summarize(rows: list[dict]) -> dict:
    refusals = [r for r in rows if r["expected_status"] == "out_of_scope"]
    lat = sorted(r["latency_s"] for r in rows)
    return {
        "n": len(rows),
        "status_accuracy": mean(r["status_ok"] for r in rows),
        "refusal_accuracy": mean(r["status_ok"] for r in refusals),
        "retrieval_hit_rate": mean(r["retrieval_hit"] for r in rows if "retrieval_hit" in r),
        "retrieval_mrr": mean(r["reciprocal_rank"] for r in rows if "reciprocal_rank" in r),
        "keyword_recall": mean(r["keywords_ok"] for r in rows if "keywords_ok" in r),
        "answer_correctness": mean(r["correct"] for r in rows if "correct" in r),
        "faithfulness": mean(r["faithfulness"] for r in rows if "faithfulness" in r),
        "citation_coverage": mean(r["citation_coverage"] for r in rows if r["citation_coverage"] is not None),
        "cross_patient_leakage": sum(r["leaked"] for r in rows),
        "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "latency_p95_s": lat[max(0, int(round(0.95 * len(lat))) - 1)] if lat else None,
    }


def check_gates(summary: dict) -> dict:
    checks = {}
    for metric, (lo, hi) in GATES.items():
        v = summary.get(metric)
        ok = v is not None and (lo is None or v >= lo) and (hi is None or v <= hi)
        checks[metric] = {"value": v, "min": lo, "max": hi, "pass": ok}
    return checks


def by_category(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r)
    return {c: {"n": len(g), "status_accuracy": mean(r["status_ok"] for r in g),
                "keyword_recall": mean(r["keywords_ok"] for r in g if "keywords_ok" in r),
                "answer_correctness": mean(r["correct"] for r in g if "correct" in r)}
            for c, g in sorted(groups.items())}


def fmt(v):
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def write_report(summary, checks, cats, rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"chat_model": config.CHAT_MODEL, "judge_model": config.JUDGE_MODEL, "embed_model": config.EMBED_MODEL,
            "pipeline_version": config.PIPELINE_VERSION, "min_similarity": config.MIN_SIMILARITY,
            "top_k": config.TOP_K, "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "eval_report.json").write_text(json.dumps(
        {"meta": meta, "summary": summary, "gates": checks, "categories": cats, "rows": rows}, indent=2))

    md = ["# Evaluation report", "",
          f"Run {meta['run_at']} with `{meta['chat_model']}` (answers), `{meta['judge_model']}` (judge and verifier), "
          f"`{meta['embed_model']}` (embeddings), pipeline {meta['pipeline_version']}, {summary['n']} questions.", "",
          "## Quality gates", "", "| Metric | Value | Gate | Result |", "|---|---|---|---|"]
    for m, c in checks.items():
        gate = f"≥ {c['min']}" if c["min"] is not None else f"≤ {c['max']}"
        md.append(f"| {m.replace('_', ' ')} | {fmt(c['value'])} | {gate} | {'Pass' if c['pass'] else 'Fail'} |")
    md += ["", "## Other metrics", "", "| Metric | Value |", "|---|---|"]
    for m in ("retrieval_mrr", "citation_coverage", "latency_p50_s", "latency_p95_s"):
        md.append(f"| {m.replace('_', ' ')} | {fmt(summary[m])} |")
    md += ["", "## By question category", "", "| Category | n | Status accuracy | Keyword recall | Correctness |",
           "|---|---|---|---|---|"]
    for c, s in cats.items():
        md.append(f"| {c.replace('_', ' ')} | {s['n']} | {fmt(s['status_accuracy'])} | {fmt(s['keyword_recall'])} | {fmt(s['answer_correctness'])} |")
    failures = [r for r in rows if not r["status_ok"] or r.get("keywords_ok") is False or r.get("correct") is False
                or r.get("retrieval_hit") is False or (r.get("faithfulness") is not None and r["faithfulness"] < 1)]
    md += ["", f"## Items to review ({len(failures)})", ""]
    for r in failures:
        issues = []
        if not r["status_ok"]:
            issues.append(f"expected {r['expected_status']}, got {r['status']} ({r['reason']})")
        if r.get("retrieval_hit") is False:
            issues.append("expected source not retrieved")
        if r.get("keywords_ok") is False:
            issues.append("missing required facts")
        if r.get("correct") is False:
            issues.append(f"judge: {r.get('judge_note', '')}")
        if r.get("unsupported_claims"):
            issues.append("unsupported: " + "; ".join(r["unsupported_claims"])[:200])
        md.append(f"- **{r['id']}** ({r['patient_id']}): {r['question']}  \n  {' | '.join(issues)}")
    (out_dir / "eval_report.md").write_text("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden_set.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--patient")
    ap.add_argument("--no-gate", action="store_true", help="always exit 0")
    args = ap.parse_args()

    from chartrag.ingest import ensure_index
    ensure_index()
    items = [json.loads(l) for l in Path(args.golden).read_text().splitlines() if l.strip()]
    if args.patient:
        items = [i for i in items if i["patient_id"] == args.patient]
    if args.limit:
        items = items[:args.limit]

    assistant = ChartAssistant()
    rows = []
    for n, item in enumerate(items, 1):
        row = run_item(assistant, item)
        rows.append(row)
        mark = "ok " if row["status_ok"] and row.get("keywords_ok", True) and row.get("correct", True) else "!! "
        print(f"{mark}[{n}/{len(items)}] {item['id']}: {row['status']}")

    summary, cats = summarize(rows), by_category(rows)
    checks = check_gates(summary)
    write_report(summary, checks, cats, rows, Path(args.out))
    print(json.dumps(summary, indent=2))
    failed = [m for m, c in checks.items() if not c["pass"]]
    if failed and not args.no_gate:
        print(f"\nQuality gate FAILED: {', '.join(failed)}. See {args.out}/eval_report.md")
        sys.exit(1)
    print("\nAll quality gates passed." if not failed else "\nGates failed (ignored with --no-gate).")


if __name__ == "__main__":
    main()
