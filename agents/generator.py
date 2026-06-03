from llm import llm


def generator_node(state):
    car_json = state["car_json"]

    prompt = f"""
You are an expert automotive marketing copywriter producing dealership-grade copy.

Generate complete marketing copy from the vehicle JSON below.

MANDATORY STRUCTURE — include ALL of these sections with clear headings:

1. Headline (model name + top 3 highlights)
2. Opening paragraph (aspirational, premium tone — 3-4 sentences)
3. Pricing (ex-showroom + on-road in INR, clearly formatted)
4. Engine & Emissions (displacement, power, torque, fuel type, emission standard)
5. Transmission & Drivetrain (gearbox type, drive type)
6. Performance (top speed, 0-100, mileage, fuel tank)
7. Dimensions & Space (length, width, height, wheelbase, ground clearance, kerb weight, boot space)
8. Suspension & Brakes (front/rear suspension types, front/rear brakes)
9. Wheels & Tyres (size, tyre type, alloy wheels, spare wheel)
10. Exterior Features (bullet list — include every item from the JSON)
11. Interior Features & Comfort (bullet list — include every item from interior_features + comfort_and_convenience)
12. Infotainment & Connectivity (screen size, Android Auto, Apple CarPlay, JBL system, wireless charging, USB ports, voice assistant)
13. Safety & Driver Assistance (total airbags + every ADAS item from safety JSON)
14. Off-Road Capabilities (4x4 system, terrain modes, locking differential, water wading depth, hill descent control)
15. Warranty & Support (standard warranty years/km, roadside assistance)
16. Ratings (Global NCAP stars, owner rating)
17. Available Colors (list all)
18. Buying Recommendation (strong, confident closing call to action)

RULES:
- Use ONLY information present in the JSON. Never invent specs.
- Bold key numbers and spec values.
- Use bullet points for feature lists.
- Premium, persuasive, professional tone.

Vehicle JSON:
{car_json}
"""

    response = llm.invoke(prompt)
    state["marketing_copy"] = response.content

    return state