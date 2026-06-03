import json
import re

from llm import llm


def extract_json(text: str) -> dict | None:
    """Robustly extract JSON from LLM response, handling markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def reviewer_node(state):
    car_json = state["car_json"]
    marketing_copy = state["marketing_copy"]

    prompt = f"""
You are a strict automotive content reviewer.

Your job is to review the marketing copy against the vehicle JSON and identify specific problems.

VEHICLE JSON:
{car_json}

MARKETING COPY:
{marketing_copy}

---

Review the copy across these four areas:

1. FACTUAL ACCURACY
   - Compare every spec in the copy against the vehicle JSON.
   - List any incorrect, exaggerated, or invented specifications.

2. MISSING INFORMATION
   - Check which of these categories are absent or insufficiently covered:
     pricing, engine, transmission, performance, dimensions, suspension, wheels_tyres,
     exterior_features, interior_features, infotainment, safety, off_road, warranty, ratings, colors
   - List only the ones that are genuinely missing or too vague.

3. GRAMMAR & READABILITY
   - Flag any grammatical errors, awkward phrasing, or repetitive sentences.
   - Note if sections are missing headings or hard to scan.

4. MARKETING QUALITY
   - Does it have a strong headline?
   - Is the opening paragraph aspirational and premium?
   - Is there a clear value proposition?
   - Does it end with a strong buying recommendation / CTA?
   - Flag anything weak or generic.

---

Return ONLY valid JSON with no markdown fences, no preamble:

{{
    "factual_errors": [
        "list of specific spec mismatches, e.g. 'Copy says 480 Nm torque but JSON says 500 Nm'"
    ],
    "missing_features": [
        "list of category names that are missing or too vague, e.g. 'suspension', 'colors'"
    ],
    "grammar_issues": [
        "list of specific grammar or readability problems found"
    ],
    "marketing_issues": [
        "list of specific marketing quality problems, e.g. 'No closing CTA', 'Headline is weak'"
    ],
    "improvements": [
        "list of concrete, actionable improvements combining all of the above"
    ]
}}
"""

    response = llm.invoke(prompt)

    result = extract_json(response.content)

    if result is None:
        print(f"[reviewer] WARNING: Could not parse JSON. Raw:\n{response.content[:300]}")
        result = {
            "factual_errors": [],
            "missing_features": [],
            "grammar_issues": [],
            "marketing_issues": [],
            "improvements": ["Reviewer could not parse response — check copy manually."]
        }

    # Merge reviewer findings into review_feedback.
    # The judge will later overwrite this with its own structured feedback,
    # but the refiner can also use this if it runs before the judge.
    state["review_feedback"] = {
        "score": state.get("quality_score", 0),  # carry forward last known score
        "checklist": {},                          # judge will populate this
        "factual_errors": result.get("factual_errors", []),
        "missing_features": result.get("missing_features", []),
        "grammar_issues": result.get("grammar_issues", []),
        "marketing_issues": result.get("marketing_issues", []),
        "improvements": result.get("improvements", [])
    }

    print(f"[reviewer] Found {len(result.get('missing_features', []))} missing sections, "
          f"{len(result.get('factual_errors', []))} factual errors, "
          f"{len(result.get('improvements', []))} improvements.")

    return state