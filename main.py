import sys
import json
import re
import uuid
import requests
from tqdm import tqdm

from graph import graph


VPIC_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/{vin}?format=json"

VPIC_FIELDS = {
    "Make":                                       "make",
    "Model":                                      "model_raw",
    "Model Year":                                 "year",
    "Trim":                                       "trim_raw",
    "Trim2":                                      "trim2",
    "Series":                                     "series",
    "Body Class":                                 "body_class",
    "Vehicle Type":                               "vehicle_type",
    "Doors":                                      "doors",
    "Drive Type":                                 "drive_type",
    "Transmission Style":                         "transmission",
    "Transmission Speeds":                        "transmission_speeds",
    "Engine Number of Cylinders":                 "engine_cylinders",
    "Displacement (L)":                           "displacement_l",
    "Displacement (CC)":                          "displacement_cc",
    "Engine Brake (hp) From":                     "engine_power_hp",
    "Engine Power (kW)":                          "engine_power_kw",
    "Engine Configuration":                       "engine_config",
    "Turbo":                                      "turbo",
    "Fuel Delivery / Fuel Injection Type":        "fuel_injection",
    "Other Engine Info":                          "other_engine_info",
    "Fuel Type - Primary":                        "fuel_primary",
    "Fuel Type - Secondary":                      "fuel_secondary",
    "Electrification Level":                      "electrification",
    "Anti-lock Braking System (ABS)":             "abs",
    "Electronic Stability Control (ESC)":         "esc",
    "Traction Control":                           "traction_control",
    "Tire Pressure Monitoring System (TPMS) Type": "tpms",
    "Backup Camera":                              "backup_camera",
    "Rear Visibility System":                     "rear_visibility_system",
    "Keyless Ignition":                           "keyless_ignition",
    "Daytime Running Light (DRL)":                "drl",
    "Brake System Type":                          "brake_system_type",
    "Blind Spot Warning (BSW)":                   "blind_spot_warning",
    "Forward Collision Warning (FCW)":            "fcw",
    "Crash Imminent Braking (CIB)":               "cib",
    "Dynamic Brake Support (DBS)":                "dbs",
    "Adaptive Cruise Control (ACC)":              "acc",
    "Lane Departure Warning (LDW)":               "ldw",
    "Lane Keeping Assistance (LKA)":              "lka",
    "Lane Centering Assistance":                  "lane_centering_assist",
    "Pedestrian Automatic Emergency Braking (PAEB)": "paeb",
    "Blind Spot Intervention (BSI)":              "bsi",
    "Parking Assist":                             "parking_assist",
    "Rear Cross Traffic Alert":                   "rcta",
    "Rear Automatic Emergency Braking":           "rear_aeb",
    "Headlamp Light Source":                      "headlamp_light_source",
    "Semiautomatic Headlamp Beam Switching":      "semi_auto_headlamps",
    "Adaptive Driving Beam (ADB)":                "adb",
    "Front Air Bag Locations":                    "front_airbags",
    "Side Air Bag Locations":                     "side_airbags",
    "Curtain Air Bag Locations":                  "curtain_airbags",
    "Knee Air Bag Locations":                     "knee_airbags",
    "Seat Belt Type":                             "seat_belt_type",
    "Other Restraint System Info":                "other_restraint_info",
    "Gross Vehicle Weight Rating From":           "gvwr",
}

NULL_VALUES = {"", "Not Applicable", "null", "None", "N/A"}


def fetch_vpic(vin: str) -> dict:
    url = VPIC_URL.format(vin=vin.strip().upper())
    print(f"\n[vpic] Fetching: {url}")
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[vpic] ERROR: {e}")
        sys.exit(1)

    parsed = {}
    error_code = None

    for item in resp.json().get("Results", []):
        var = item.get("Variable", "").strip()
        val = item.get("Value")
        if var == "Error Code":
            error_code = val
        mapped_key = VPIC_FIELDS.get(var)
        if mapped_key and val and str(val).strip() not in NULL_VALUES:
            parsed[mapped_key] = str(val).strip()

    if error_code and error_code != "0":
        print(f"[vpic] WARNING: Error code {error_code}. Data may be incomplete.")

    return parsed


def normalize_spacing(text: str) -> str:
    """Insert space between letter↔digit: GLC300→GLC 300, 4XE→4 XE"""
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    return text.strip()


def build_vehicle(vpic: dict, vin: str) -> dict:
    model_raw = vpic.get("model_raw", "")
    trim_raw  = vpic.get("trim_raw", "")

    model_norm = normalize_spacing(model_raw)
    trim_norm  = normalize_spacing(trim_raw)

    # Full display name — avoid duplicating trim if already in model
    model_no_space = model_norm.lower().replace(" ", "")
    trim_no_space  = trim_norm.lower().replace(" ", "")
    if trim_norm and trim_no_space not in model_no_space:
        full_name = f"{vpic.get('make', '')} {model_norm} {trim_norm}".strip()
    else:
        full_name = f"{vpic.get('make', '')} {model_norm}".strip()

    safety_keys = [
        "abs", "esc", "traction_control", "tpms", "backup_camera",
        "rear_visibility_system", "keyless_ignition", "drl", "brake_system_type",
        "blind_spot_warning", "bsi", "fcw", "cib", "dbs", "acc", "ldw", "lka",
        "lane_centering_assist",
        "paeb", "parking_assist", "rcta", "rear_aeb", "semi_auto_headlamps",
        "adb", "headlamp_light_source",
        "front_airbags", "side_airbags", "curtain_airbags", "knee_airbags",
        "seat_belt_type", "other_restraint_info",
    ]
    safety = {k: vpic.get(k, "") for k in safety_keys}

    return {
        "vin":               vin.strip().upper(),
        "year":              vpic.get("year", ""),
        "make":              vpic.get("make", ""),
        "model":             model_norm,
        "model_raw":         model_raw,
        "trim":              trim_norm,
        "trim_raw":          trim_raw,
        "full_name":         full_name,
        "series":            vpic.get("series", ""),
        "body_class":        vpic.get("body_class", ""),
        "vehicle_type":      vpic.get("vehicle_type", ""),
        "doors":             vpic.get("doors", ""),
        "drive_type":        vpic.get("drive_type", ""),
        "transmission":      vpic.get("transmission", ""),
        "transmission_speeds": vpic.get("transmission_speeds", ""),
        "engine_cylinders":  vpic.get("engine_cylinders", ""),
        "displacement_l":    vpic.get("displacement_l", ""),
        "displacement_cc":   vpic.get("displacement_cc", ""),
        "engine_power_hp":   vpic.get("engine_power_hp", ""),
        "engine_power_kw":   vpic.get("engine_power_kw", ""),
        "engine_config":     vpic.get("engine_config", ""),
        "turbo":             vpic.get("turbo", ""),
        "other_engine_info": vpic.get("other_engine_info", ""),
        "fuel_primary":      vpic.get("fuel_primary", ""),
        "fuel_secondary":    vpic.get("fuel_secondary", ""),
        "electrification":   vpic.get("electrification", ""),
        "safety":            safety,
        # No dealer — these are populated from VPIC + prompt only
        "dealer_name":       "",
        "dealer_blurb":      "",
    }


if __name__ == "__main__":
    vin = input("\nEnter VIN number: ").strip()
    if not vin:
        print("No VIN entered. Exiting.")
        sys.exit(1)

    vpic_data = fetch_vpic(vin)
    vehicle   = build_vehicle(vpic_data, vin)

    print(f"\n[main] Vehicle  : {vehicle['full_name']}")
    print(f"[main] Trim raw : '{vehicle['trim_raw']}'  →  normalized: '{vehicle['trim']}'")
    print(f"[main] Make     : {vehicle['make']}")

    initial_state = {
        "vehicle":         vehicle,
        "marketing_copy":  "",
        "review_feedback": {},
        "quality_score":   0,
        "approved":        False,
        "iteration":       0,
        "max_iterations":  5,
        "run_id":         str(uuid.uuid4()),
        "trace":          []
    }

    MAX_ITERATIONS      = 5
    NODES_PER_ITERATION = 4
    TOTAL_STEPS         = MAX_ITERATIONS * NODES_PER_ITERATION

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