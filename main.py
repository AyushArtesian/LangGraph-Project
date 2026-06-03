import sys
import requests
from tqdm import tqdm

from graph import graph


# ─────────────────────────────────────────────────────────
# VPIC API SETUP
# ─────────────────────────────────────────────────────────
VPIC_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/{vin}?format=json"

FIELDS_TO_EXTRACT = [
    "Make", "Model", "Model Year", "Trim",
    "Body Class", "Drive Type", "Transmission Style", "Transmission Speeds",
    "Engine Number of Cylinders", "Displacement (L)", "Displacement (CC)",
    "Engine Power (kW)", "Engine Brake (hp) From",
    "Fuel Type - Primary", "Fuel Type - Secondary",
    "Electrification Level", "Turbo", "Engine Configuration",
    "Anti-lock Braking System (ABS)", "Electronic Stability Control (ESC)",
    "Traction Control", "Tire Pressure Monitoring System (TPMS) Type",
    "Backup Camera", "Blind Spot Warning (BSW)",
    "Forward Collision Warning (FCW)", "Crash Imminent Braking (CIB)",
    "Dynamic Brake Support (DBS)", "Adaptive Cruise Control (ACC)",
    "Lane Departure Warning (LDW)", "Lane Keeping Assistance (LKA)",
    "Parking Assist", "Rear Cross Traffic Alert",
    "Keyless Ignition", "Daytime Running Light (DRL)",
    "Seat Belt Type", "Front Air Bag Locations", "Side Air Bag Locations",
    "Doors", "Steering Location"
]


def fetch_vpic(vin: str) -> dict:
    """Fetch and parse VPIC data for a VIN."""
    url = VPIC_URL.format(vin=vin.strip().upper())
    print(f"\n[vpic] Fetching: {url}")
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[vpic] ERROR: {e}")
        sys.exit(1)

    data = resp.json()
    parsed = {}

    for item in data.get("Results", []):
        var = item.get("Variable", "").strip()
        val = item.get("Value")
        if var in FIELDS_TO_EXTRACT and val and val.strip() not in ("", "Not Applicable", "null"):
            parsed[var] = val.strip()

    return parsed


def normalize_model_name(vpic: dict) -> str:
    import re

    model = vpic.get("Model", "").strip()
    trim = vpic.get("Trim", "").strip()

    full_model = model

    if trim:
        trim_spaced = re.sub(
            r'([A-Za-z])(\d)',
            r'\1 \2',
            trim
        )

        trim_spaced = re.sub(
            r'(\d)([A-Za-z])',
            r'\1 \2',
            trim_spaced
        )

        if trim_spaced.lower() not in model.lower():
            full_model = f"{model} {trim_spaced}"

    return full_model


def get_vin() -> str:
    """Get VIN from user."""
    vin = input("\nEnter VIN number: ").strip()
    if not vin:
        print("ERROR: VIN cannot be empty.")
        sys.exit(1)
    return vin


def build_vehicle_data(vpic: dict) -> dict:
    """Build vehicle data dict from VPIC response."""
    return {
        "vin": vpic.get("VIN", ""),
        "make": vpic.get("Make", ""),
        "model": normalize_model_name(vpic),
        "model_raw": vpic.get("Model", ""),
        "trim": vpic.get("Trim", ""),
        "year": vpic.get("Model Year", ""),
        "body_class": vpic.get("Body Class", ""),
        "drive_type": vpic.get("Drive Type", ""),
        "transmission": vpic.get("Transmission Style", ""),
        "transmission_speeds": vpic.get("Transmission Speeds", ""),
        "engine_cylinders": vpic.get("Engine Number of Cylinders", ""),
        "displacement_l": vpic.get("Displacement (L)", ""),
        "displacement_cc": vpic.get("Displacement (CC)", ""),
        "engine_power_kw": vpic.get("Engine Power (kW)", ""),
        "engine_power_hp": vpic.get("Engine Brake (hp) From", ""),
        "fuel_primary": vpic.get("Fuel Type - Primary", ""),
        "fuel_secondary": vpic.get("Fuel Type - Secondary", ""),
        "electrification": vpic.get("Electrification Level", ""),
        "turbo": vpic.get("Turbo", ""),
        "engine_config": vpic.get("Engine Configuration", ""),
        "doors": vpic.get("Doors", ""),
        "safety": {
            "abs": vpic.get("Anti-lock Braking System (ABS)", ""),
            "esc": vpic.get("Electronic Stability Control (ESC)", ""),
            "traction_control": vpic.get("Traction Control", ""),
            "tpms": vpic.get("Tire Pressure Monitoring System (TPMS) Type", ""),
            "backup_camera": vpic.get("Backup Camera", ""),
            "blind_spot_warning": vpic.get("Blind Spot Warning (BSW)", ""),
            "fcw": vpic.get("Forward Collision Warning (FCW)", ""),
            "cib": vpic.get("Crash Imminent Braking (CIB)", ""),
            "dbs": vpic.get("Dynamic Brake Support (DBS)", ""),
            "acc": vpic.get("Adaptive Cruise Control (ACC)", ""),
            "ldw": vpic.get("Lane Departure Warning (LDW)", ""),
            "lka": vpic.get("Lane Keeping Assistance (LKA)", ""),
            "parking_assist": vpic.get("Parking Assist", ""),
            "rcta": vpic.get("Rear Cross Traffic Alert", ""),
            "keyless_ignition": vpic.get("Keyless Ignition", ""),
            "drl": vpic.get("Daytime Running Light (DRL)", ""),
            "front_airbags": vpic.get("Front Air Bag Locations", ""),
            "side_airbags": vpic.get("Side Air Bag Locations", ""),
            "seat_belt": vpic.get("Seat Belt Type", ""),
        },
        "_vpic_raw": vpic,
    }


if __name__ == "__main__":
    vin = get_vin()
    vpic_data = fetch_vpic(vin)
    vehicle_data = build_vehicle_data(vpic_data)

    print(f"\n[main] Vehicle identified: {vehicle_data['make']} {vehicle_data['model']}")

    initial_state = {
        "vehicle": vehicle_data,
        "marketing_copy": "",
        "review_feedback": {},
        "quality_score": 0,
        "approved": False,
        "iteration": 0,
    }

    MAX_ITERATIONS = 5
    NODES_PER_ITERATION = 4
    TOTAL_STEPS = MAX_ITERATIONS * NODES_PER_ITERATION

    result = None

    print("\nStarting Marketing Copy Generation...\n")

    with tqdm(total=TOTAL_STEPS, desc="Pipeline", unit="node") as pbar:
        for event in graph.stream(initial_state):
            node_name = next(iter(event))
            pbar.set_description(f"Running: {node_name}")
            pbar.update(1)
            if node_name != "__end__":
                result = event[node_name]

    print("\n" + "=" * 80)
    print("FINAL MARKETING COPY")
    print("=" * 80)
    print(result["marketing_copy"])

    print("\n" + "=" * 80)
    print(f"QUALITY SCORE : {result['quality_score']}/100")
    print(f"APPROVED      : {result['approved']}")
    print(f"ITERATIONS    : {result['iteration']}")
    print("=" * 80)