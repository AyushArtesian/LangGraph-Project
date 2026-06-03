from tqdm import tqdm

from graph import graph


sample_car = {
    "brand": "Toyota",
    "model": "Fortuner",
    "variant": "GR-S 4x4 AT",
    "year": 2025,
    "body_type": "SUV",

    "pricing": {
        "ex_showroom_price_inr": 5200000,
        "on_road_price_inr": 5900000
    },

    "engine": {
        "type": "2.8L Turbocharged Diesel",
        "displacement_cc": 2755,
        "cylinders": 4,
        "max_power_hp": 201,
        "max_power_rpm": 3000,
        "max_torque_nm": 500,
        "max_torque_rpm": 2800,
        "fuel_type": "Diesel",
        "emission_standard": "BS6 Phase 2"
    },

    "transmission": {
        "type": "Automatic",
        "gearbox": "6-Speed Automatic",
        "drive_type": "4WD"
    },

    "performance": {
        "top_speed_kmph": 190,
        "acceleration_0_100_sec": 10.8,
        "fuel_tank_capacity_l": 80,
        "mileage_kmpl": 14.4
    },

    "dimensions": {
        "length_mm": 4795,
        "width_mm": 1855,
        "height_mm": 1835,
        "wheelbase_mm": 2745,
        "ground_clearance_mm": 225,
        "kerb_weight_kg": 2200,
        "boot_space_l": 296
    },

    "suspension": {
        "front": "Double Wishbone",
        "rear": "Multi-Link",
        "front_brakes": "Ventilated Disc",
        "rear_brakes": "Disc"
    },

    "wheels_and_tyres": {
        "wheel_size_inch": 18,
        "tyre_type": "All-Terrain",
        "alloy_wheels": True,
        "spare_wheel": "Full Size"
    },

    "exterior_features": [
        "LED Headlamps",
        "LED DRLs",
        "Sequential Turn Indicators",
        "Rear Spoiler",
        "Chrome Front Grille",
        "Roof Rails",
        "Powered Tailgate",
        "Electric ORVMs",
        "Auto Folding ORVMs",
        "Rear Wiper and Washer",
        "Fog Lamps"
    ],

    "interior_features": [
        "Leather Upholstery",
        "Ventilated Front Seats",
        "Power Driver Seat",
        "Dual Zone Climate Control",
        "Ambient Lighting",
        "Push Button Start",
        "Smart Key Access",
        "Auto Dimming IRVM",
        "Premium Soft Touch Dashboard",
        "Rear AC Vents"
    ],

    "infotainment": {
        "screen_size_inch": 10.1,
        "touchscreen": True,
        "android_auto": True,
        "apple_carplay": True,
        "wireless_charging": True,
        "bluetooth": True,
        "usb_ports": 4,
        "premium_sound_system": "JBL 11 Speaker Audio System",
        "connected_car_technology": True,
        "voice_assistant": True
    },

    "safety": {
        "airbags": 7,
        "abs": True,
        "ebd": True,
        "traction_control": True,
        "electronic_stability_control": True,
        "hill_start_assist": True,
        "hill_descent_control": True,
        "tpms": True,
        "360_camera": True,
        "front_parking_sensors": True,
        "rear_parking_sensors": True,
        "adaptive_cruise_control": True,
        "lane_departure_warning": True,
        "blind_spot_monitor": True,
        "rear_cross_traffic_alert": True
    },

    "comfort_and_convenience": {
        "cruise_control": True,
        "wireless_charger": True,
        "sunroof": False,
        "powered_tailgate": True,
        "rain_sensing_wipers": True,
        "auto_headlamps": True,
        "rear_window_sunshade": True
    },

    "off_road_capabilities": {
        "4x4": True,
        "terrain_modes": [
            "Mud",
            "Sand",
            "Rock",
            "Snow"
        ],
        "locking_differential": True,
        "water_wading_depth_mm": 700,
        "hill_descent_control": True
    },

    "colors_available": [
        "Attitude Black",
        "Platinum White Pearl",
        "Sparkling Black Crystal Shine",
        "Avant-Garde Bronze",
        "Super White"
    ],

    "warranty": {
        "standard_warranty_years": 3,
        "standard_warranty_km": 100000,
        "roadside_assistance_years": 3
    },

    "ratings": {
        "global_ncap": 5,
        "owner_rating": 4.7
    }
}

initial_state = {
    "car_json": sample_car,
    "marketing_copy": "",
    "review_feedback": "",
    "quality_score": 0,
    "approved": False,
    "iteration": 0,
}

MAX_ITERATIONS = 5
NODES_PER_ITERATION = 4
TOTAL_STEPS = MAX_ITERATIONS * NODES_PER_ITERATION

result = None

print("\nStarting Marketing Copy Workflow...\n")

with tqdm(
    total=TOTAL_STEPS,
    desc="Marketing Pipeline",
    unit="node"
) as pbar:

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
print("QUALITY SCORE")
print("=" * 80)

print(result["quality_score"])

print("\n" + "=" * 80)
print("APPROVED")
print("=" * 80)

print(result["approved"])

print("\n" + "=" * 80)
print("ITERATIONS")
print("=" * 80)

print(result["iteration"])