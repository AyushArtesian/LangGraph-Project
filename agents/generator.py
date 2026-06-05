# agents/generator.py

from llm import llm
from langsmith import traceable

from utils.tracing import add_trace

import time

start_time = time.time()

def _val(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("", "none", "null", "not applicable", "n/a", "no") else s


def _build_standard_features(safety: dict) -> list:
    return _build_status_features(safety)[0]


def _build_optional_features(safety: dict) -> list:
    """Features confirmed Optional in VPIC — described as 'available'."""
    return _build_status_features(safety)[1]


def _build_status_features(safety: dict) -> tuple[list, list]:
    """Collect safety features by VPIC status so all confirmed Standard items are usable."""
    features = []
    optional = []
    seen = set()

    label_overrides = {
        "keyless_ignition": "keyless ignition",
        "esc": "electronic stability control",
        "traction_control": "traction control",
        "abs": "anti-lock brakes",
        "drl": "daytime running lights",
        "blind_spot_warning": "blind spot warning",
        "bsi": "blind spot intervention",
        "dbs": "dynamic brake support",
        "fcw": "forward collision warning",
        "cib": "crash imminent braking",
        "acc": "adaptive cruise control",
        "ldw": "lane departure warning",
        "lka": "lane keeping assistance",
        "lane_centering_assist": "lane centering assistance",
        "parking_assist": "parking assist",
        "rcta": "rear cross traffic alert",
        "rear_aeb": "rear automatic emergency braking",
        "paeb": "pedestrian automatic emergency braking",
        "semi_auto_headlamps": "semi-automatic headlamp beam switching",
        "adb": "adaptive driving beam",
    }

    status_keys_to_skip = {
        "tpms", "backup_camera", "rear_visibility_system",
        "front_airbags", "side_airbags", "curtain_airbags", "knee_airbags",
        "seat_belt_type", "other_restraint_info", "brake_system_type", "headlamp_light_source",
    }

    for key, raw_val in safety.items():
        if key in status_keys_to_skip:
            continue

        val = _val(raw_val).lower()
        if val not in {"standard", "optional"}:
            continue

        label = label_overrides.get(key, key.replace("_", " "))
        if label in seen:
            continue

        if val == "standard":
            features.append(label)
        else:
            optional.append(label)
        seen.add(label)

    # TPMS values are often type strings like "Direct" rather than Standard/Optional.
    if _val(safety.get("tpms")):
        features.append("tire pressure monitoring")

    # Rear visibility system can be represented by either field.
    rear_vis_value = _val(safety.get("backup_camera")) or _val(safety.get("rear_visibility_system"))
    if rear_vis_value:
        rear_vis_status = rear_vis_value.lower()
        if rear_vis_status == "optional":
            if "a rear visibility system" not in seen:
                optional.append("a rear visibility system")
                seen.add("a rear visibility system")
        else:
            if "a rear visibility system" not in seen:
                features.append("a rear visibility system")
                seen.add("a rear visibility system")

    headlamp_source = _val(safety.get("headlamp_light_source"))
    if headlamp_source:
        headlamp_label = f"{headlamp_source} headlamps"
        if headlamp_label not in seen:
            features.append(headlamp_label)
            seen.add(headlamp_label)

    # Airbags and restraint details are direct facts, not optional statuses.
    airbag_parts = []
    if _val(safety.get("front_airbags")):   airbag_parts.append("front")
    if _val(safety.get("side_airbags")):    airbag_parts.append("side")
    if _val(safety.get("curtain_airbags")): airbag_parts.append("curtain")
    if _val(safety.get("knee_airbags")):    airbag_parts.append("knee")
    if airbag_parts:
        features.append(f"{' and '.join(airbag_parts)} airbags for added peace of mind")

    return features, optional


def _list_to_prose(items: list) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"

@traceable(name = "Generator Agent")
def generator_node(state):

    duration = round(time.time() - start_time, 2)

    v      = state["vehicle"]
    safety = v.get("safety", {})

    make       = _val(v.get("make"))
    model      = _val(v.get("model"))
    trim       = _val(v.get("trim"))
    trim_raw   = _val(v.get("trim_raw"))
    full_name  = _val(v.get("full_name"))
    doors      = _val(v.get("doors"))
    body_class = _val(v.get("body_class"))
    drive_type = _val(v.get("drive_type"))
    hp         = _val(v.get("engine_power_hp"))
    transmission = _val(v.get("transmission"))
    transmission_speeds = _val(v.get("transmission_speeds"))
    disp_l     = _val(v.get("displacement_l"))
    cylinders  = _val(v.get("engine_cylinders"))
    engine_cfg = _val(v.get("engine_config"))
    turbo      = _val(v.get("turbo"))
    fuel_pri   = _val(v.get("fuel_primary"))
    fuel_sec   = _val(v.get("fuel_secondary"))
    elec       = _val(v.get("electrification"))
    brake_type = _val(safety.get("brake_system_type"))
    restraint  = _val(safety.get("other_restraint_info"))
    headlamp_light_source = _val(safety.get("headlamp_light_source"))

    turbo_prefix = "turbocharged " if turbo.lower() == "yes" else ""
    inline_cfg   = "inline " if "in-line" in engine_cfg.lower() else (engine_cfg.lower() + " " if engine_cfg else "")
    engine_desc  = f"{turbo_prefix}{disp_l}L {inline_cfg}{cylinders}-cylinder".strip() if disp_l and cylinders else ""

    is_phev = "phev" in elec.lower() if elec else False

    std_features = _build_standard_features(safety)
    opt_features = _build_optional_features(safety)

    std_prose = _list_to_prose(std_features)
    opt_prose = _list_to_prose(opt_features)

    prompt = f"""
You are a professional automotive marketing writer.

Write a dealership-quality vehicle description following the EXACT structure of the EXAMPLE below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE — study the structure and tone:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This 2023 Jeep Wrangler 4XE combines legendary Jeep capability with advanced plug-in hybrid
technology, delivering an exciting blend of efficiency, power, and off-road confidence. Powered
by a turbocharged 2.0L inline 4-cylinder engine paired with an electric drive system, the
Wrangler 4XE produces an impressive 375 horsepower and features a 4WD drivetrain for outstanding
traction and control across a variety of driving conditions.

Designed with Jeep's iconic SUV styling, this four-door Wrangler 4XE offers the versatility and
rugged character drivers expect while adding the benefits of plug-in hybrid performance. Key
features include keyless ignition, electronic stability control, traction control, anti-lock
brakes, daytime running lights, a rear visibility system, tire pressure monitoring, and front
and side airbags for added peace of mind.

This vehicle is equipped with a hydraulic braking system and features standard safety technologies
engineered to enhance driver confidence. Available driver-assistance technologies include blind
spot monitoring, dynamic brake support, forward collision warning, collision intervention braking,
and semi-automatic headlamp beam switching.

Finished in Black and showing approximately 65,108 miles, this Wrangler 4XE delivers the
open-air freedom, adventure-ready capability, and innovative hybrid technology that make it a
standout choice for drivers seeking both efficiency and authentic Jeep performance.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT: Exactly 4 paragraphs. Flowing prose only. No bullet points. No bold. No headers.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VEHICLE DATA (VPIC — single source of truth):
  Full name       : {full_name}
  Make            : {make}
  Model           : {model}
  Trim            : {trim} (raw VPIC value: {trim_raw})
  Body Class      : {body_class}
  Doors           : {doors}
  Drive Type      : {drive_type}
  Engine          : {engine_desc if engine_desc else "see VPIC"}
  Power           : {hp + " hp" if hp else "see VPIC"}
    Transmission    : {((transmission_speeds + "-speed ") if transmission_speeds else "") + transmission if transmission else "see VPIC"}
  Fuel Primary    : {fuel_pri}
  Fuel Secondary  : {fuel_sec}
  Electrification : {elec}
    Headlamps       : {headlamp_light_source if headlamp_light_source else "see VPIC"}
  Brake System    : {brake_type}
  Restraint Info  : {restraint}

STANDARD FEATURES (confirmed Standard in VPIC — write as included):
  {std_prose if std_prose else "standard safety systems"}

OPTIONAL FEATURES (confirmed Optional in VPIC — write as "available"):
  {opt_prose if opt_prose else "none confirmed"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. MODEL NAME: Always write with space between letters and numbers.
   Model "{model}" + Trim "{trim}" → full name: {full_name}
   NEVER write: "{trim_raw}" (no space). ALWAYS write: "{trim}" (with space).

2. Do NOT include: any year, any date, plant location, list price, base price.

3. Do NOT add "-Class" to the model name.

4. Do NOT invent features. Use ONLY what is listed in VPIC data above.

5. Do NOT mention kW — use only HP ({hp} hp).

6. Paragraph structure:
   Para 1: Opening sentence introducing the vehicle + powertrain description
   Para 2: Body style description + standard safety/tech features list
   Para 3: Brake system + optional/available ADAS features
   Para 4: Closing value statement (no color or mileage — those are not in VPIC)

7. If transmission details are present, include them in the powertrain description.
8. Output ONLY the 4-paragraph description. No preamble, no notes.
"""

    response = llm.invoke(prompt)
    state["marketing_copy"] = response.content.strip()

    add_trace(
        state,
        agent="Generator Agent",
        details={
            "vehicle": state["vehicle"]["full_name"],
            "duration_seconds": duration,
        })   

    return state