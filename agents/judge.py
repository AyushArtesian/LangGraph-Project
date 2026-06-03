from llm import llm

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
        return ", ".join(
            str(v).strip()
            for v in value
            if str(v).strip()
        )

    return str(value).strip()


def judge_node(state):
    feedback = state.get("review_feedback", {})

    score = 100
    issues = []

    if not _is_yes(feedback.get("model_name_present")):
        score = 0
        issues.append("CRITICAL: model name missing")

    if not _is_yes(feedback.get("model_name_correct_spacing")):
        score = 0
        issues.append("CRITICAL: model spacing incorrect")

    if _is_yes(feedback.get("duplicate_trim_detected")):
        score -= 50
        issues.append("duplicate trim detected")

    if _is_yes(feedback.get("duplicate_model_detected")):
        score -= 50
        issues.append("duplicate model detected")

    if _is_yes(feedback.get("contains_year_date")):
        score -= 40
        issues.append("contains year/date")

    if _is_yes(feedback.get("contains_price")):
        score -= 40
        issues.append("contains price")

    if _is_yes(feedback.get("contains_plant_location")):
        score -= 30
        issues.append("contains plant location")

    if _is_yes(feedback.get("contains_hallucinated_data")):
        hallucinations = _hallucination_text(
            feedback.get("hallucination_examples")
        )

        score -= 40

        if hallucinations:
            issues.append(
                f"Hallucinations: {hallucinations[:200]}"
            )

    score = max(score, 0)

    approved = (
        score >= APPROVAL_THRESHOLD
        and len(issues) == 0
    )

    state["quality_score"] = score
    state["approved"] = approved
    state["iteration"] += 1

    print(f"\n[judge] Iteration {state['iteration']}")
    print(f"[judge] Score: {score}/100")
    print(f"[judge] Approved: {approved}")

    if issues:
        print("[judge] Issues Found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("[judge] No issues found.")

    return state