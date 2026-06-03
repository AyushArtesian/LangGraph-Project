# agents/reviewer.py

import json
import re

from llm import llm


def extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response safely."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return None


def reviewer_node(state):
    vehicle = state["vehicle"]
    marketing_copy = state["marketing_copy"]

    make = vehicle.get("make", "")
    model = vehicle.get("model", "")
    trim = vehicle.get("trim", "")

    prompt = f"""
You are an automotive compliance reviewer.

Your job is to validate whether the marketing copy follows VPIC data.

VEHICLE DATA:

Make: {make}
Model: {model}
Trim: {trim}

MARKETING COPY:

{marketing_copy}

VALIDATION RULES:

1. Model name "{model}" must appear.
2. Model spacing must be correct.
3. No duplicated trim names.
4. No duplicated model names.
5. No year references.
6. No dates.
7. No pricing.
8. No plant/factory/manufacturing location.
9. No hallucinated features.
10. No hallucinated specifications.
11. No invented technology.
12. No invented interior features.
13. No invented performance claims.

Return ONLY valid JSON.

{{
    "model_name_present": "yes|no",
    "model_name_correct_spacing": "yes|no",
    "duplicate_trim_detected": "yes|no",
    "duplicate_model_detected": "yes|no",
    "contains_year_date": "yes|no",
    "contains_price": "yes|no",
    "contains_plant_location": "yes|no",
    "contains_hallucinated_data": "yes|no",
    "hallucination_examples": [],
    "summary": ""
}}
"""

    response = llm.invoke(prompt)

    result = extract_json(response.content)

    if result is None:
        result = {
            "model_name_present": "no",
            "model_name_correct_spacing": "no",
            "duplicate_trim_detected": "yes",
            "duplicate_model_detected": "yes",
            "contains_year_date": "yes",
            "contains_price": "yes",
            "contains_plant_location": "yes",
            "contains_hallucinated_data": "yes",
            "hallucination_examples": ["review parse failure"],
            "summary": "review parse failure",
        }

    state["review_feedback"] = result

    print(
        f"[reviewer] "
        f"Model={result.get('model_name_present')} | "
        f"Spacing={result.get('model_name_correct_spacing')} | "
        f"DupTrim={result.get('duplicate_trim_detected')} | "
        f"DupModel={result.get('duplicate_model_detected')} | "
        f"Hallucinations={result.get('contains_hallucinated_data')}"
    )

    return state