# agents/reviewer.py

import json
import re

from llm import llm
from langsmith import traceable

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


@traceable(name = "Reviewer Agent")
def reviewer_node(state):
    v = state["vehicle"]
    copy = state["marketing_copy"]

    make = v.get("make", "")
    model = v.get("model", "")
    trim = v.get("trim", "")
    trim_raw = v.get("trim_raw", "")
    full_name = v.get("full_name", "")
    safety = v.get("safety", {})

    def s(key):
        val = safety.get(key, "")
        return str(val).strip() if val else ""

    rear_vis_confirmed = bool(
        s("backup_camera") or s("rear_visibility_system")
    )

    prompt = f"""
You are an automotive compliance reviewer.

Your ONLY job is to verify factual consistency between the marketing copy and VPIC data.

Do NOT invent problems.
Do NOT require exact VPIC wording.
Industry-standard feature names and VPIC abbreviations are equivalent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VPIC DATA (SOURCE OF TRUTH)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Make              : {make}
Model             : "{model}"
Trim              : "{trim}"
Raw Trim          : "{trim_raw}"
Full Vehicle Name : "{full_name}"

Body Class        : {v.get('body_class', '')}
Doors             : {v.get('doors', '')}
Drive Type        : {v.get('drive_type', '')}

Transmission Style : {v.get('transmission', '')}
Transmission Speeds: {v.get('transmission_speeds', '')}

Displacement      : {v.get('displacement_l', '')}L
Cylinders         : {v.get('engine_cylinders', '')}
Engine Config     : {v.get('engine_config', '')}
Turbo             : {v.get('turbo', '')}

Power             : {v.get('engine_power_hp', '')} hp

Fuel Primary      : {v.get('fuel_primary', '')}
Fuel Secondary    : {v.get('fuel_secondary', '')}
Electrification   : {v.get('electrification', '')}

Other Engine Info : {v.get('other_engine_info', '')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIRMED STANDARD FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ABS                         : {s('abs')}
ESC                         : {s('esc')}
Traction Control            : {s('traction_control')}
TPMS                        : {s('tpms')}
Backup Camera               : {s('backup_camera')}
Rear Visibility System      : {s('rear_visibility_system')}
Keyless Ignition            : {s('keyless_ignition')}
Daytime Running Lights      : {s('drl')}

Blind Spot Warning          : {s('blind_spot_warning')}
Forward Collision Warning   : {s('fcw')}
Crash Imminent Braking      : {s('cib')}
Dynamic Brake Support       : {s('dbs')}
Pedestrian AEB              : {s('paeb')}
Rear AEB                    : {s('rear_aeb')}

Lane Departure Warning      : {s('ldw')}

Semi Auto Headlamps         : {s('semi_auto_headlamps')}

Front Airbags               : {s('front_airbags')}
Side Airbags                : {s('side_airbags')}
Curtain Airbags             : {s('curtain_airbags')}
Knee Airbags                : {s('knee_airbags')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIRMED OPTIONAL / AVAILABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Adaptive Cruise Control     : {s('acc')}
Lane Keeping Assistance     : {s('lka')}
Lane Centering Assistance   : {s('lane_centering_assist')}
Blind Spot Intervention     : {s('bsi')}
Adaptive Driving Beam       : {s('adb')}
Rear Cross Traffic Alert    : {s('rear_cross_traffic_alert')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALLOWED FEATURE NAME EQUIVALENTS
NEVER FLAG THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FCW  = Forward Collision Warning

CIB  = Crash Imminent Braking

DBS  = Dynamic Brake Support

PAEB = Pedestrian Automatic Emergency Braking
PAEB = Pedestrian AEB

ACC  = Adaptive Cruise Control

LDW  = Lane Departure Warning

LKA  = Lane Keeping Assistance

BSW  = Blind Spot Warning

BSI  = Blind Spot Intervention

ADB  = Adaptive Driving Beam

Rear AEB = Rear Automatic Emergency Braking

Backup Camera = Rear Visibility System

LED Headlamps = Headlamp Light Source LED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT REVIEW RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NEVER flag a feature if VPIC confirms it.

2. NEVER flag a feature because the copy uses
   the expanded industry name instead of the VPIC abbreviation.

3. NEVER flag:
   - FCW / Forward Collision Warning
   - CIB / Crash Imminent Braking
   - DBS / Dynamic Brake Support
   - PAEB / Pedestrian Automatic Emergency Braking
   - ACC / Adaptive Cruise Control
   - LKA / Lane Keeping Assistance
   - LDW / Lane Departure Warning
   - BSW / Blind Spot Warning
   - BSI / Blind Spot Intervention
   - ADB / Adaptive Driving Beam

4. If VPIC says Standard,
   the copy may describe it as:
   - standard
   - included
   - equipped with

5. If VPIC says Optional,
   the copy may describe it as:
   - available
   - optional

6. Rear Visibility System and Backup Camera
   are equivalent.

7. LED Headlamps are allowed if VPIC confirms
   LED as the headlamp source.

8. Do NOT require exact VPIC wording.

9. Only mark hallucination when the copy
   introduces a feature, specification,
   horsepower value, drivetrain, engine size,
   transmission, or safety technology that
   does NOT exist anywhere in VPIC.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESCRIPTION TO REVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{copy}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. model_name_present

2. model_name_correct_spacing

3. duplicate_trim_detected

4. duplicate_model_detected

5. contains_year_date

6. contains_price

7. contains_plant_location

8. contains_hallucinated_data

Return ONLY valid JSON:

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
  "summary": "brief summary"
}}
"""

    response = llm.invoke(prompt)

    result = extract_json(response.content)

    if result is None:
        print("[reviewer] WARNING: Could not parse JSON.")

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
        f"Hallucinations={result.get('contains_hallucinated_data')} | "
        f"{result.get('summary', '')}"
    )

    return state