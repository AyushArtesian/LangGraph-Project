# agents/generator.py

from llm import llm


def generator_node(state):
    vehicle = state["vehicle"]

    make = vehicle.get("make", "").strip()
    model = vehicle.get("model", "").strip()
    trim = vehicle.get("trim", "").strip()

    # Avoid duplicate trim in model name
    # Example:
    # model = "Wrangler 4 XE"
    # trim  = "4XE"
    # final should NOT become "Wrangler 4 XE 4XE"

    normalized_model = model.lower().replace(" ", "")
    normalized_trim = trim.lower().replace(" ", "")

    if trim and normalized_trim not in normalized_model:
        full_name = f"{make} {model} {trim}".strip()
    else:
        full_name = f"{make} {model}".strip()

    safety_features = []

    safety_map = {
        "abs": "Anti-lock Braking System (ABS)",
        "esc": "Electronic Stability Control (ESC)",
        "traction_control": "Traction Control",
        "tpms": "Tire Pressure Monitoring System (TPMS)",
        "backup_camera": "Backup Camera",
        "blind_spot_warning": "Blind Spot Warning",
        "fcw": "Forward Collision Warning",
        "cib": "Crash Imminent Braking",
        "dbs": "Dynamic Brake Support",
        "acc": "Adaptive Cruise Control",
        "ldw": "Lane Departure Warning",
        "lka": "Lane Keeping Assistance",
        "parking_assist": "Parking Assist",
        "rcta": "Rear Cross Traffic Alert",
        "keyless_ignition": "Keyless Ignition",
        "drl": "Daytime Running Light",
    }

    for key, label in safety_map.items():
        value = vehicle.get("safety", {}).get(key)

        if value and str(value).strip().lower() not in [
            "",
            "none",
            "null",
            "not applicable",
        ]:
            safety_features.append(label)

    safety_list = (
        "\n".join(f"- {feature}" for feature in safety_features)
        if safety_features
        else "- Standard safety systems"
    )

    prompt = f"""
You are a professional automotive marketing writer.

Create a dealership-quality vehicle description.

ABSOLUTE RULES:

1. Use ONLY the VPIC vehicle data provided below.
2. Never invent specifications.
3. Never invent features.
4. Never invent dimensions.
5. Never invent towing capacity.
6. Never invent seating capacity.
7. Never invent technology features.
8. Never invent interior features.
9. Never invent colors.
10. Never invent mileage.
11. Never mention model year.
12. Never mention any date.
13. Never mention pricing.
14. Never mention MSRP.
15. Never mention plant location.
16. Never mention factory location.
17. Never mention manufacturing location.
18. Never use information not explicitly present in VPIC data.
19. NEVER write the trim twice.
20. NEVER write "{trim} {trim}".
21. NEVER write duplicated model names.
22. The exact vehicle name "{full_name}" must appear naturally.

VEHICLE DATA:

Make: {make}
Model: {model}
Trim: {trim if trim else "N/A"}

Body Class: {vehicle.get("body_class", "N/A")}
Doors: {vehicle.get("doors", "N/A")}
Drive Type: {vehicle.get("drive_type", "N/A")}

Transmission: {vehicle.get("transmission", "N/A")}
Transmission Speeds: {vehicle.get("transmission_speeds", "N/A")}

Engine Cylinders: {vehicle.get("engine_cylinders", "N/A")}
Displacement: {vehicle.get("displacement_l", "N/A")} L
Horsepower: {vehicle.get("engine_power_hp", "N/A")}
Kilowatts: {vehicle.get("engine_power_kw", "N/A")}

Fuel Type Primary: {vehicle.get("fuel_primary", "N/A")}
Fuel Type Secondary: {vehicle.get("fuel_secondary", "N/A")}
Electrification: {vehicle.get("electrification", "N/A")}
Turbo: {vehicle.get("turbo", "N/A")}
Engine Configuration: {vehicle.get("engine_config", "N/A")}

CONFIRMED SAFETY FEATURES:
{safety_list}

WRITING REQUIREMENTS:

- 3 to 4 paragraphs
- Professional dealership tone
- Natural language
- No bullet points
- No markdown
- No headings
- No feature speculation
- Mention only confirmed VPIC data

IMPORTANT:

If a field is empty, missing, null, or N/A,
DO NOT mention it.

Return ONLY the marketing description.
"""

    response = llm.invoke(prompt)

    state["marketing_copy"] = response.content.strip()

    return state