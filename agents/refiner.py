# agents/refiner.py

from llm import llm


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
        return "; ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def _val(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("", "none", "null", "not applicable", "n/a", "no") else s


def _list_to_prose(items: list) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def refiner_node(state):
    v        = state["vehicle"]
    feedback = state.get("review_feedback", {})
    current  = state["marketing_copy"]

    make      = _val(v.get("make"))
    model     = _val(v.get("model"))
    trim      = _val(v.get("trim"))
    trim_raw  = _val(v.get("trim_raw"))    # ← defined here, fixes the NameError
    full_name = _val(v.get("full_name"))
    safety    = v.get("safety", {})

    def s(key):
        val = safety.get(key, "")
        return str(val).strip() if val else ""

    # Standard features
    std_features = []
    for key, label in [
        ("keyless_ignition",    "keyless ignition"),
        ("esc",                 "electronic stability control"),
        ("traction_control",    "traction control"),
        ("abs",                 "anti-lock brakes"),
        ("drl",                 "daytime running lights"),
        ("tpms",                "tire pressure monitoring"),
    ]:
        if s(key):
            std_features.append(label)
    if s("backup_camera") or s("rear_visibility_system"):
        std_features.append("a rear visibility system")
    airbag_parts = []
    if s("front_airbags"):   airbag_parts.append("front")
    if s("side_airbags"):    airbag_parts.append("side")
    if s("curtain_airbags"): airbag_parts.append("curtain")
    if s("knee_airbags"):    airbag_parts.append("knee")
    if airbag_parts:
        std_features.append(f"{' and '.join(airbag_parts)} airbags for added peace of mind")

    # Optional features
    opt_features = []
    for key, label in [
        ("blind_spot_warning",  "blind spot monitoring"),
        ("dbs",                 "dynamic brake support"),
        ("fcw",                 "forward collision warning"),
        ("cib",                 "collision intervention braking"),
        ("semi_auto_headlamps", "semi-automatic headlamp beam switching"),
        ("acc",                 "adaptive cruise control"),
        ("ldw",                 "lane departure warning"),
        ("lka",                 "lane keeping assistance"),
    ]:
        val = s(key)
        if val and val.lower() == "optional":
            opt_features.append(label)

    # Build fix instructions from reviewer feedback
    fixes = []

    if not _is_yes(feedback.get("model_name_present")):
        fixes.append(
            f"ADD the model name. It must appear as: {full_name}\n"
            f"   Model '{model}' must be explicitly written in the description."
        )

    if not _is_yes(feedback.get("model_name_correct_spacing")):
        fixes.append(
            f"FIX model name spacing:\n"
            f"   WRONG: '{trim_raw}' (letters and numbers touching)\n"
            f"   CORRECT: '{trim}' (space between letters and numbers)\n"
            f"   Full vehicle name must appear as: {full_name}"
        )

    if _is_yes(feedback.get("duplicate_trim_detected")):
        fixes.append(
            f"REMOVE duplicate trim. The trim appears twice in a row.\n"
            f"   WRONG: '{model} {trim} {trim}'\n"
            f"   CORRECT: '{full_name}'"
        )

    if _is_yes(feedback.get("duplicate_model_detected")):
        fixes.append("REMOVE the duplicate model name — it appears more than once unnecessarily.")

    if _is_yes(feedback.get("contains_year_date")):
        fixes.append(
            "REMOVE all year and date references.\n"
            "   Do not write any year (e.g. 2023, 2026) anywhere in the description."
        )

    if _is_yes(feedback.get("contains_price")):
        fixes.append("REMOVE all pricing — no dollar amounts, MSRP, list price, or base price.")

    if _is_yes(feedback.get("contains_plant_location")):
        fixes.append("REMOVE all plant, factory, or manufacturing location references.")

    if _is_yes(feedback.get("contains_hallucinated_data")):
        hallucinations = _hallucination_text(feedback.get("hallucination_examples"))
        fixes.append(
            f"REMOVE hallucinated content — only use confirmed VPIC features:\n"
            f"   {hallucinations}"
        )

    fixes_block = "\n\n".join(f"{i+1}. {fix}" for i, fix in enumerate(fixes)) \
                  if fixes else "Improve overall quality and flow."

    prompt = f"""
You are correcting an automotive marketing description.

VEHICLE DATA (VPIC — single source of truth):
  Make            : {make}
  Model           : {model}
  Trim            : {trim}  (raw VPIC: {trim_raw})
  Full name       : {full_name}
  Body Class      : {_val(v.get('body_class'))}
  Doors           : {_val(v.get('doors'))}
  Drive Type      : {_val(v.get('drive_type'))}
  Displacement    : {_val(v.get('displacement_l'))}L
  Cylinders       : {_val(v.get('engine_cylinders'))}
  Engine Config   : {_val(v.get('engine_config'))}
  Turbo           : {_val(v.get('turbo'))}
  Power           : {_val(v.get('engine_power_hp'))} hp  (use HP only, not kW)
  Fuel Primary    : {_val(v.get('fuel_primary'))}
  Fuel Secondary  : {_val(v.get('fuel_secondary'))}
  Electrification : {_val(v.get('electrification'))}
  Brake System    : {s('brake_system_type')}
  Restraint Info  : {s('other_restraint_info')}

STANDARD FEATURES (confirmed — write as standard/included):
  {_list_to_prose(std_features) or "standard safety systems"}

OPTIONAL FEATURES (confirmed — write as available):
  {_list_to_prose(opt_features) or "none confirmed"}

CURRENT COPY (score {state.get('quality_score', 0)}/100):
{current}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED FIXES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{fixes_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Model name must appear as: {full_name}
   Space between letters and numbers — NEVER write "{trim_raw}", ALWAYS write "{trim}".
2. No years, no dates, no pricing, no plant locations.
3. No "-Class" suffix on model name.
4. Only use VPIC-confirmed features.
5. Exactly 4 paragraphs. No bullet points. No bold. No headers. Flowing prose only.
6. Output ONLY the corrected description. No preamble, no notes.
"""

    response = llm.invoke(prompt)
    state["marketing_copy"] = response.content.strip()
    return state