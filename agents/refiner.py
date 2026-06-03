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
        return ", ".join(
            str(v).strip()
            for v in value
            if str(v).strip()
        )

    return str(value).strip()


def refiner_node(state):
    vehicle = state["vehicle"]
    feedback = state.get("review_feedback", {})
    current_copy = state["marketing_copy"]

    make = vehicle.get("make", "")
    model = vehicle.get("model", "")
    trim = vehicle.get("trim", "")

    normalized_model = model.lower().replace(" ", "")
    normalized_trim = trim.lower().replace(" ", "")

    if trim and normalized_trim not in normalized_model:
        full_name = f"{make} {model} {trim}".strip()
    else:
        full_name = f"{make} {model}".strip()

    fixes = []

    if not _is_yes(
        feedback.get("model_name_present")
    ):
        fixes.append(
            f"Add exact model name '{model}'"
        )

    if not _is_yes(
        feedback.get("model_name_correct_spacing")
    ):
        fixes.append(
            f"Use exact spacing '{model}'"
        )

    if _is_yes(
        feedback.get("duplicate_trim_detected")
    ):
        fixes.append(
            "Remove duplicated trim names"
        )

    if _is_yes(
        feedback.get("duplicate_model_detected")
    ):
        fixes.append(
            "Remove duplicated model names"
        )

    if _is_yes(
        feedback.get("contains_year_date")
    ):
        fixes.append(
            "Remove all year/date references"
        )

    if _is_yes(
        feedback.get("contains_price")
    ):
        fixes.append(
            "Remove all pricing references"
        )

    if _is_yes(
        feedback.get("contains_plant_location")
    ):
        fixes.append(
            "Remove all plant/factory/manufacturing locations"
        )

    if _is_yes(
        feedback.get("contains_hallucinated_data")
    ):
        hallucinations = _hallucination_text(
            feedback.get("hallucination_examples")
        )

        fixes.append(
            f"Remove hallucinated content: {hallucinations}"
        )

    fixes_text = "\n".join(
        f"- {fix}"
        for fix in fixes
    )

    prompt = f"""
You are correcting an automotive marketing description.

VEHICLE DATA (ONLY SOURCE OF TRUTH):

Make: {make}
Model: {model}
Trim: {trim}

Body Class: {vehicle.get('body_class', '')}
Doors: {vehicle.get('doors', '')}
Drive Type: {vehicle.get('drive_type', '')}

Transmission: {vehicle.get('transmission', '')}
Transmission Speeds: {vehicle.get('transmission_speeds', '')}

Engine Cylinders: {vehicle.get('engine_cylinders', '')}
Displacement: {vehicle.get('displacement_l', '')}
Horsepower: {vehicle.get('engine_power_hp', '')}
Power kW: {vehicle.get('engine_power_kw', '')}

Fuel Primary: {vehicle.get('fuel_primary', '')}
Fuel Secondary: {vehicle.get('fuel_secondary', '')}
Electrification: {vehicle.get('electrification', '')}
Turbo: {vehicle.get('turbo', '')}

CURRENT COPY:

{current_copy}

REQUIRED FIXES:

{fixes_text}

MANDATORY RULES:

1. Use ONLY VPIC data.
2. No hallucinations.
3. No invented features.
4. No invented specifications.
5. No dates.
6. No years.
7. No pricing.
8. No plant location.
9. No factory location.
10. No manufacturing location.
11. No duplicate trim names.
12. No duplicate model names.
13. Vehicle name must be exactly:

{full_name}

14. Never write:
Wrangler 4 XE 4XE

15. Never write:
{model} {trim}

if trim already exists in model.

16. Professional dealership style.
17. Natural paragraphs.
18. No bullet points.
19. Return ONLY corrected description.

Rewrite now.
"""

    response = llm.invoke(prompt)

    state["marketing_copy"] = response.content.strip()

    return state