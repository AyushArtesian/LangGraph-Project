# agents/judge.py

from llm import llm
from langsmith import traceable

from utils.tracing import add_trace

APPROVAL_THRESHOLD = 90


def _is_yes(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "yes"
    return False



def _hallucination_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()

@traceable(name = "Judge Agent")
def judge_node(state):
    feedback = state.get("review_feedback", {})

    score  = 100
    issues = []

    # ── CRITICAL failures — immediately disqualify ────────────────────
    if not _is_yes(feedback.get("model_name_present")):
        score = 0
        issues.append("CRITICAL: model name missing from copy")

    if not _is_yes(feedback.get("model_name_correct_spacing")):
        score = 0
        issues.append("CRITICAL: model name spacing incorrect (letters+numbers must have a space)")

    # ── Major deductions ──────────────────────────────────────────────
    if _is_yes(feedback.get("duplicate_trim_detected")):
        score -= 40
        issues.append("duplicate trim name written twice in a row")

    if _is_yes(feedback.get("duplicate_model_detected")):
        score -= 30
        issues.append("model name unnecessarily duplicated")

    if _is_yes(feedback.get("contains_year_date")):
        score -= 40
        issues.append("copy contains a year or date reference")

    if _is_yes(feedback.get("contains_price")):
        score -= 40
        issues.append("copy contains pricing information")

    if _is_yes(feedback.get("contains_plant_location")):
        score -= 30
        issues.append("copy contains plant/factory/manufacturing location")

    if _is_yes(feedback.get("contains_hallucinated_data")):
        hallucinations = _hallucination_text(feedback.get("hallucination_examples"))
        score -= 30
        if hallucinations:
            issues.append(f"hallucinated content: {hallucinations[:300]}")

    score = max(score, 0)

    # Approved only if score meets threshold AND no critical/major issues
    critical_count = sum(1 for i in issues if "CRITICAL" in i)
    approved = (score >= APPROVAL_THRESHOLD) and (critical_count == 0)

    state["quality_score"] = score
    state["approved"]      = approved
    state["iteration"]     += 1

    print(f"\n[judge] Iteration {state['iteration']} | Score: {score}/100 | Approved: {approved}")
    if issues:
        print("[judge] Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("[judge] No issues found.")

    add_trace(
        state,
        agent="Judge Agent",
        details={
            "quality_score": score,
            "approved": approved,
            "issues": issues,
            "iteration": state["iteration"],
        }
    )

    return state