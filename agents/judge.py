import json
import re

from llm import llm


APPROVAL_THRESHOLD = 85


def extract_json(text: str) -> dict | None:
    """Robustly extract JSON, stripping markdown fences."""
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


# ─────────────────────────────────────────────
# STAGE 1 — Objective checklist (no scoring)
# Just yes/no per item. LLM cannot game this.
# ─────────────────────────────────────────────
CHECKLIST_PROMPT = """
You are a content verification assistant.

Read the MARKETING COPY below and answer yes or no for each item.
Do not score anything. Just check presence.

MARKETING COPY:
{marketing_copy}

For each item, answer "yes" if it is clearly present in the copy, "no" if absent or too vague.

Return ONLY valid JSON, no markdown fences:

{{
  "pricing_exshowroom": "yes_or_no",
  "pricing_onroad": "yes_or_no",
  "engine_displacement": "yes_or_no",
  "engine_power_torque": "yes_or_no",
  "transmission_gearbox": "yes_or_no",
  "drive_type_4wd": "yes_or_no",
  "top_speed": "yes_or_no",
  "acceleration_0_100": "yes_or_no",
  "mileage": "yes_or_no",
  "fuel_tank": "yes_or_no",
  "dimensions_length_width_height": "yes_or_no",
  "wheelbase": "yes_or_no",
  "ground_clearance": "yes_or_no",
  "boot_space": "yes_or_no",
  "kerb_weight": "yes_or_no",
  "suspension_front_rear": "yes_or_no",
  "brakes_front_rear": "yes_or_no",
  "wheel_size": "yes_or_no",
  "tyre_type": "yes_or_no",
  "exterior_features_min5": "yes_or_no",
  "interior_features_min5": "yes_or_no",
  "comfort_features_cruise_wipers": "yes_or_no",
  "infotainment_screen_size": "yes_or_no",
  "android_auto_carplay": "yes_or_no",
  "jbl_sound_system": "yes_or_no",
  "airbag_count": "yes_or_no",
  "adas_min5_features": "yes_or_no",
  "offroad_terrain_modes": "yes_or_no",
  "water_wading_depth": "yes_or_no",
  "locking_differential": "yes_or_no",
  "warranty_years_km": "yes_or_no",
  "roadside_assistance": "yes_or_no",
  "ncap_rating": "yes_or_no",
  "owner_rating": "yes_or_no",
  "colors_listed": "yes_or_no"
}}
"""

# ─────────────────────────────────────────────
# STAGE 2 — Factual spot-check
# Only checks specs that are actually mentioned.
# ─────────────────────────────────────────────
FACTUAL_PROMPT = """
You are a fact-checking assistant for automotive marketing.

VEHICLE JSON (source of truth):
{car_json}

MARKETING COPY:
{marketing_copy}

Compare only the numeric specifications mentioned in the copy against the JSON.
List any value in the copy that does NOT match the JSON exactly.

Return ONLY valid JSON, no markdown fences:

{{
  "factual_errors": [
    "describe each mismatch, e.g. 'Copy says 480 Nm but JSON says 500 Nm'"
  ]
}}

If everything matches, return: {{"factual_errors": []}}
"""

# ─────────────────────────────────────────────
# STAGE 3 — Marketing quality score (0–100)
# Only scores tone, persuasion, CTA quality.
# Coverage is already verified in stage 1.
# ─────────────────────────────────────────────
QUALITY_PROMPT = """
You are a senior automotive marketing director evaluating copy quality.

The copy has already been verified to contain all required technical specifications.
Your job is ONLY to evaluate writing quality.

MARKETING COPY:
{marketing_copy}

Score these five dimensions. Be honest and varied — do not default to round numbers.

1. Headline strength (0–20): Is it specific, compelling, and premium?
2. Opening paragraph (0–20): Aspirational? Does it create desire?
3. Language quality (0–20): Premium automotive vocabulary? No generic filler phrases?
4. Value proposition clarity (0–20): Does a buyer clearly understand why to choose this car?
5. Closing CTA (0–20): Confident, specific, action-driving?

Return ONLY valid JSON, no markdown fences:

{{
  "headline_score": <0-20>,
  "opening_score": <0-20>,
  "language_score": <0-20>,
  "value_prop_score": <0-20>,
  "cta_score": <0-20>,
  "headline_feedback": "one sentence",
  "opening_feedback": "one sentence",
  "language_feedback": "one sentence",
  "value_prop_feedback": "one sentence",
  "cta_feedback": "one sentence"
}}
"""


def run_checklist(marketing_copy: str) -> dict:
    prompt = CHECKLIST_PROMPT.format(marketing_copy=marketing_copy)
    response = llm.invoke(prompt)
    result = extract_json(response.content)
    if result is None:
        print("[judge/checklist] WARNING: Could not parse checklist JSON.")
        return {}
    return result


def run_factual_check(car_json: dict, marketing_copy: str) -> list:
    prompt = FACTUAL_PROMPT.format(car_json=car_json, marketing_copy=marketing_copy)
    response = llm.invoke(prompt)
    result = extract_json(response.content)
    if result is None:
        print("[judge/factual] WARNING: Could not parse factual check JSON.")
        return []
    return result.get("factual_errors", [])


def run_quality_score(marketing_copy: str) -> dict:
    prompt = QUALITY_PROMPT.format(marketing_copy=marketing_copy)
    response = llm.invoke(prompt)
    result = extract_json(response.content)
    if result is None:
        print("[judge/quality] WARNING: Could not parse quality score JSON.")
        return {
            "headline_score": 10, "opening_score": 10, "language_score": 10,
            "value_prop_score": 10, "cta_score": 10,
            "headline_feedback": "", "opening_feedback": "",
            "language_feedback": "", "value_prop_feedback": "", "cta_feedback": ""
        }
    return result


def compute_final_score(checklist: dict, factual_errors: list, quality: dict) -> tuple[int, list]:
    """
    Score breakdown:
      - Coverage:  35 pts  (1 pt per checklist item present, out of 35)
      - Factual:   30 pts  (30 minus 5 per error, floor 0)
      - Quality:   35 pts  (sum of 5 quality dimension scores, each 0-20, scaled to 35)

    Total: 100 pts
    """
    # Coverage (35 pts)
    total_items = len(checklist)
    present_items = sum(1 for v in checklist.values() if str(v).lower() == "yes")
    missing_items = [k for k, v in checklist.items() if str(v).lower() == "no"]
    coverage_score = round((present_items / total_items) * 35) if total_items > 0 else 0

    # Factual (30 pts)
    factual_score = max(0, 30 - (len(factual_errors) * 5))

    # Quality (35 pts) — sum of five 0-20 scores, scaled to 35
    raw_quality = (
        quality.get("headline_score", 0) +
        quality.get("opening_score", 0) +
        quality.get("language_score", 0) +
        quality.get("value_prop_score", 0) +
        quality.get("cta_score", 0)
    )
    quality_score = round((raw_quality / 100) * 35)

    total = coverage_score + factual_score + quality_score

    print(f"[judge] Coverage: {coverage_score}/35 ({present_items}/{total_items} items) | "
          f"Factual: {factual_score}/30 ({len(factual_errors)} errors) | "
          f"Quality: {quality_score}/35 (raw {raw_quality}/100)")

    return total, missing_items


def judge_node(state):
    marketing_copy = state["marketing_copy"]
    car_json = state["car_json"]

    print(f"\n[judge] Running 3-stage evaluation...")

    # Stage 1: Objective checklist
    checklist = run_checklist(marketing_copy)

    # Stage 2: Factual accuracy
    factual_errors = run_factual_check(car_json, marketing_copy)

    # Stage 3: Marketing quality
    quality = run_quality_score(marketing_copy)

    # Compute score from parts
    score, missing_items = compute_final_score(checklist, factual_errors, quality)
    approved = score >= APPROVAL_THRESHOLD

    # Build improvements list from quality feedback
    improvements = []
    for key in ["headline_feedback", "opening_feedback", "language_feedback",
                "value_prop_feedback", "cta_feedback"]:
        fb = quality.get(key, "")
        if fb:
            improvements.append(fb)

    state["quality_score"] = score
    state["approved"] = approved
    state["review_feedback"] = {
        "score": score,
        "checklist": checklist,
        "missing_features": missing_items,
        "factual_errors": factual_errors,
        "improvements": improvements,
        "quality_breakdown": {
            "headline": quality.get("headline_score", 0),
            "opening": quality.get("opening_score", 0),
            "language": quality.get("language_score", 0),
            "value_prop": quality.get("value_prop_score", 0),
            "cta": quality.get("cta_score", 0),
        }
    }

    state["iteration"] += 1

    print(f"[judge] Iteration {state['iteration']} — Score: {score}/100 — Approved: {approved}")
    if missing_items:
        print(f"[judge] Missing checklist items: {missing_items}")
    if factual_errors:
        print(f"[judge] Factual errors: {factual_errors}")

    return state