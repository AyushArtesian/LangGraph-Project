from llm import llm


def refiner_node(state):

    feedback = state["review_feedback"]
    score = feedback.get("score", 0)
    missing = feedback.get("missing_features", [])
    errors = feedback.get("factual_errors", [])
    improvements = feedback.get("improvements", [])
    quality = feedback.get("quality_breakdown", {})

    # ── Build targeted fix instructions based on what actually failed ──

    fix_blocks = []

    if missing:
        fix_blocks.append(
            "MISSING SECTIONS — these checklist items were not found in the copy. "
            "You MUST include them explicitly:\n"
            + "\n".join(f"  • {item.replace('_', ' ')}" for item in missing)
        )

    if errors:
        fix_blocks.append(
            "FACTUAL ERRORS — fix these spec mismatches against the vehicle JSON:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    # Identify weak quality dimensions (scored below 14/20)
    weak_dims = []
    dim_labels = {
        "headline":   "Headline — make it more specific, premium, and compelling",
        "opening":    "Opening paragraph — must be more aspirational and desire-creating",
        "language":   "Language quality — eliminate generic filler; use premium automotive vocabulary",
        "value_prop": "Value proposition — buyer must immediately understand why to choose this car",
        "cta":        "Closing CTA — must be confident, specific, and action-driving",
    }
    for dim, label in dim_labels.items():
        if quality.get(dim, 20) < 14:
            weak_dims.append(label)

    if weak_dims:
        fix_blocks.append(
            "WRITING QUALITY — improve these specific dimensions:\n"
            + "\n".join(f"  • {d}" for d in weak_dims)
        )

    if improvements:
        fix_blocks.append(
            "ADDITIONAL FEEDBACK FROM REVIEWER:\n"
            + "\n".join(f"  • {i}" for i in improvements)
        )

    targeted_section = (
        "\n\n".join(fix_blocks)
        if fix_blocks
        else "Improve overall writing quality — make every sentence more premium and persuasive."
    )

    prompt = f"""
You are an elite automotive copywriter producing dealership-grade marketing copy.

VEHICLE DATA (source of truth — never invent specs):
{state["car_json"]}

CURRENT COPY (score: {score}/100 — target: 85+):
{state["marketing_copy"]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECIFIC ISSUES TO FIX IN THIS REWRITE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{targeted_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY STRUCTURE — every section must be present:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.  Headline          — model name + 3 most compelling highlights
2.  Opening           — 3-4 aspirational sentences that create desire
3.  Pricing           — ex-showroom (₹52,00,000) + on-road (₹59,00,000)
4.  Engine            — displacement (2755 cc), power (201 hp @ 3000 rpm), torque (500 Nm @ 2800 rpm), fuel, BS6 Phase 2
5.  Transmission      — 6-Speed Automatic, 4WD
6.  Performance       — top speed (190 km/h), 0-100 (10.8 sec), mileage (14.4 kmpl), tank (80 L)
7.  Dimensions        — length (4795), width (1855), height (1835), wheelbase (2745), ground clearance (225), kerb weight (2200 kg), boot (296 L)
8.  Suspension/Brakes — Double Wishbone front, Multi-Link rear, Ventilated Disc front, Disc rear
9.  Wheels/Tyres      — 18-inch, All-Terrain, alloy wheels, full-size spare
10. Exterior Features — bullet list of all 11 items from JSON
11. Interior/Comfort  — bullet list of all interior_features + comfort_and_convenience items
12. Infotainment      — 10.1-inch, Android Auto, Apple CarPlay, wireless charging, 4 USB, JBL 11-speaker, connected car, voice assistant
13. Safety/ADAS       — 7 airbags + all 10 ADAS features from JSON
14. Off-Road          — 4x4, Mud/Sand/Rock/Snow modes, locking diff, 700 mm wading, hill descent
15. Warranty          — 3 years / 100,000 km + 3 years roadside assistance
16. Ratings           — 5-star Global NCAP, 4.7/5 owner rating
17. Colors            — all 5 colors
18. Buying CTA        — strong, confident, specific recommendation

WRITING RULES:
- Bold every key number and spec value.
- Bullet points for all feature lists.
- No repetition across sections.
- No generic phrases like "best-in-class" without spec backing.
- Every sentence must earn its place.

Output ONLY the final marketing copy. No preamble, no commentary.
"""

    response = llm.invoke(prompt)
    state["marketing_copy"] = response.content

    return state