# agents/reviewer.py

import json
import re

from llm import llm


def extract_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
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
    v    = state["vehicle"]
    copy = state["marketing_copy"]

    make      = v.get("make", "")
    model     = v.get("model", "")
    trim      = v.get("trim", "")
    trim_raw  = v.get("trim_raw", "")
    full_name = v.get("full_name", "")
    safety    = v.get("safety", {})

    def s(key):
        val = safety.get(key, "")
        return str(val).strip() if val else ""

    # Rear visibility is confirmed if EITHER field has a value
    rear_vis_confirmed = bool(s("backup_camera") or s("rear_visibility_system"))

    prompt = f"""
You are an automotive compliance reviewer.

Validate the vehicle description against the VPIC data below.
Be precise. Only flag genuine problems.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VPIC DATA (confirmed source of truth):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Make              : {make}
Model (exact)     : "{model}"
Trim (exact)      : "{trim}"  (raw VPIC: "{trim_raw}")
Full vehicle name : "{full_name}"
Body Class        : {v.get('body_class', '')}
Doors             : {v.get('doors', '')}
Drive Type        : {v.get('drive_type', '')}
Displacement      : {v.get('displacement_l', '')}L
Cylinders         : {v.get('engine_cylinders', '')}
Engine Config     : {v.get('engine_config', '')}   ("In-Line" = "inline" = same thing)
Turbo             : {v.get('turbo', '')}
Power             : {v.get('engine_power_hp', '')} hp  (only HP is available; kW is NOT provided)
Fuel Primary      : {v.get('fuel_primary', '')}
Fuel Secondary    : {v.get('fuel_secondary', '')}
Electrification   : {v.get('electrification', '')}
Other Engine Info  : {v.get('other_engine_info', '')}

CONFIRMED STANDARD SAFETY (all of these are supported — do NOT flag as hallucination):
  ABS               : {s('abs')}
  ESC               : {s('esc')}
  Traction Control  : {s('traction_control')}
  TPMS              : {s('tpms')}
  Backup Camera     : {s('backup_camera')}
  Rear Visibility   : {s('rear_visibility_system')}
  Rear visibility system CONFIRMED: {rear_vis_confirmed}
  Keyless Ignition  : {s('keyless_ignition')}
  DRL               : {s('drl')}
  Brake System      : {s('brake_system_type')}
  Front Airbags     : {s('front_airbags')}
  Side Airbags      : {s('side_airbags')}
  Curtain Airbags   : {s('curtain_airbags')}
  Knee Airbags      : {s('knee_airbags')}
  Restraint Info    : {s('other_restraint_info')}

CONFIRMED OPTIONAL/AVAILABLE (do NOT flag as hallucination if described as "available"):
  Blind Spot Warning : {s('blind_spot_warning')}
  FCW                : {s('fcw')}
  CIB                : {s('cib')}
  DBS                : {s('dbs')}
  Semi-Auto Headlamps: {s('semi_auto_headlamps')}
  ACC                : {s('acc')}
  LDW                : {s('ldw')}
  LKA                : {s('lka')}

IMPORTANT NOTES:
- "rear visibility system" IS confirmed (Rear Visibility System CONFIRMED: {rear_vis_confirmed}). Do NOT flag it.
- "inline" and "In-Line" are the same engine config. Do NOT flag "inline" as a hallucination.
- ESC/traction control/ABS are confirmed Standard above. Do NOT flag them.
- Optional features described as "available" are NOT hallucinations.
- Standard features described as included/standard are NOT hallucinations.
- HP figure ({v.get('engine_power_hp', '')} hp) is confirmed. Do NOT flag it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESCRIPTION TO VALIDATE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{copy}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECKS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. model_name_present: Does "{model}" appear in the description? yes/no

2. model_name_correct_spacing: Is there a space between every letter-number boundary?
   e.g. "4 XE" is correct, "4XE" is wrong. If model has no numbers, answer yes.

3. duplicate_trim_detected: Does the trim appear twice in a row (e.g. "4 XE 4 XE")? yes/no

4. duplicate_model_detected: Is the model name written twice unnecessarily? yes/no

5. contains_year_date: Does the copy mention a calendar year (e.g. 2023) or date? yes/no

6. contains_price: Does the copy mention any price, MSRP, dollar amount? yes/no

7. contains_plant_location: Does the copy mention a plant city, factory, or manufacturing location? yes/no

8. contains_hallucinated_data: Does the copy claim anything NOT in the VPIC data above?
   ONLY flag actual hallucinations — features with no VPIC entry at all, or wrong numbers.
   Do NOT flag: rear visibility system, inline engine, ABS/ESC/TC, confirmed airbags, HP figure,
   standard features described as standard, optional features described as available.

Return ONLY valid JSON, no markdown fences:

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
    "summary": "one sentence — main issues found, or No issues found"
}}
"""

    response = llm.invoke(prompt)
    result   = extract_json(response.content)

    if result is None:
        print("[reviewer] WARNING: Could not parse JSON. Treating as all-fail.")
        result = {
            "model_name_present":         "no",
            "model_name_correct_spacing": "no",
            "duplicate_trim_detected":    "yes",
            "duplicate_model_detected":   "yes",
            "contains_year_date":         "yes",
            "contains_price":             "yes",
            "contains_plant_location":    "yes",
            "contains_hallucinated_data": "yes",
            "hallucination_examples":     ["review parse failure"],
            "summary":                    "review parse failure",
        }

    state["review_feedback"] = result

    print(
        f"[reviewer] Model={result.get('model_name_present')} | "
        f"Spacing={result.get('model_name_correct_spacing')} | "
        f"DupTrim={result.get('duplicate_trim_detected')} | "
        f"Hallucinations={result.get('contains_hallucinated_data')} | "
        f"{result.get('summary', '')}"
    )

    return state