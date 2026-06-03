# LangGraph Automotive Marketing Copy Pipeline

A production-quality, multi-agent LangGraph workflow that takes a raw vehicle VIN, decodes it via the NHTSA vPIC API, and iteratively generates, reviews, refines, and scores dealership-grade automotive marketing copy using Azure OpenAI — with no year, price, or hallucinated content.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Graph Flow](#graph-flow)
4. [Project Structure](#project-structure)
5. [Module Reference](#module-reference)
   - [main.py](#mainpy)
   - [state.py](#statepy)
   - [graph.py](#graphpy)
   - [llm.py](#llmpy)
   - [agents/generator.py](#agentsgeneratorpy)
   - [agents/reviewer.py](#agentsreviewerpy)
   - [agents/refiner.py](#agentsrefinerpy)
   - [agents/judge.py](#agentsjudgepy)
6. [VPIC Field Mapping](#vpic-field-mapping)
7. [Vehicle Object Schema](#vehicle-object-schema)
8. [State Schema](#state-schema)
9. [Safety & ADAS Feature Handling](#safety--adas-feature-handling)
10. [Scoring & Approval Model](#scoring--approval-model)
11. [Prerequisites](#prerequisites)
12. [Installation](#installation)
13. [Environment Variables](#environment-variables)
14. [Running the Pipeline](#running-the-pipeline)
15. [Sample Output](#sample-output)
16. [Customization](#customization)
17. [Dependencies](#dependencies)
18. [Ignored Files](#ignored-files)

---

## Overview

This project solves a common dealership problem: producing factually accurate, hallucination-free vehicle descriptions at scale from raw VIN data.

The workflow:
1. Accepts a VIN from the user at the command line.
2. Decodes the VIN using the NHTSA vPIC public API to retrieve ~60+ vehicle attributes.
3. Normalizes and structures the data into a vehicle object.
4. Runs a four-node LangGraph pipeline (generator → reviewer → judge, with a refiner loop) to produce and iteratively improve marketing copy.
5. Terminates when the copy is **approved** (score ≥ 90) or when **5 iterations** are exhausted.
6. Prints the final copy, quality score, approval status, and iteration count.

The system enforces strict no-hallucination rules: every feature mentioned in the copy must be traceable to a confirmed VPIC field.

---

## Architecture

```
VIN (user input)
      │
      ▼
  NHTSA vPIC API
      │
      ▼
  fetch_vpic()  ──►  build_vehicle()
                            │
                            ▼
                     MarketingState
                            │
                            ▼
                 ┌──────────────────────┐
                 │   LangGraph Graph    │
                 │                      │
                 │  START               │
                 │    │                 │
                 │    ▼                 │
                 │  generator           │
                 │    │                 │
                 │    ▼                 │
                 │  reviewer            │◄─────────┐
                 │    │                 │          │
                 │    ▼                 │          │
                 │  judge ─── END       │          │
                 │    │                 │          │
                 │    └──► refiner ─────┘          │
                 └──────────────────────┘
```

- **generator** uses VPIC data to write the initial 4-paragraph copy.
- **reviewer** validates the copy against VPIC data and returns structured JSON feedback.
- **judge** scores the copy deterministically from the reviewer's JSON flags and decides approve or retry.
- **refiner** rewrites the copy using targeted fix instructions built from the reviewer's flags.
- The loop is: `generator → reviewer → judge → refiner → reviewer → judge → ...` up to 5 iterations.

---

## Graph Flow

Defined in `graph.py`:

```
START → generator → reviewer → judge
                                 │
                      ┌──────────┴──────────┐
                      │                     │
                  approved or          score < 90 and
                  iteration >= 5        iteration < 5
                      │                     │
                     END               refiner → reviewer → judge
```

### Routing logic (`route_after_judge`)

| Condition | Next node |
|---|---|
| `state["approved"] == True` | `END` — prints ✅ approval message |
| `state["iteration"] >= MAX_ITERATIONS (5)` | `END` — prints ❌ rejection message |
| otherwise | `"refiner"` — prints retry message and loops |

The `refiner` node connects back to `reviewer` (not `generator`), so the LLM never starts over from scratch; it always builds on the current copy.

---

## Project Structure

```text
.
├── agents/
│   ├── generator.py   # Initial 4-paragraph copy generation from VPIC data
│   ├── reviewer.py    # Structured JSON compliance review against VPIC
│   ├── refiner.py     # Targeted rewrite driven by reviewer feedback flags
│   └── judge.py       # Deterministic scoring + approval/retry routing
├── graph.py           # LangGraph StateGraph definition and routing
├── llm.py             # AzureChatOpenAI client initialization
├── main.py            # CLI entry point: VIN input, VPIC fetch, graph execution
├── state.py           # MarketingState TypedDict definition
├── requirements.txt   # Python package dependencies
└── .gitignore         # Excludes .env, __pycache__, venv
```

---

## Module Reference

### `main.py`

The CLI entry point. Responsible for:

1. **VIN input** — prompts the user with `input("\nEnter VIN number: ")`.
2. **VPIC fetch** (`fetch_vpic`) — calls `https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/{vin}?format=json`, parses the `Results` array, and maps ~60 VPIC `Variable` names to internal keys. Null/inapplicable values (`""`, `"Not Applicable"`, `"null"`, `"None"`, `"N/A"`) are dropped. Exits with an error if the HTTP request fails.
3. **Vehicle build** (`build_vehicle`) — structures the flat VPIC dict into a nested vehicle object with a `safety` sub-dict and computed fields like `full_name`.
4. **Model name normalization** (`normalize_spacing`) — inserts spaces at every letter↔digit boundary (e.g. `GLC300` → `GLC 300`, `4XE` → `4 XE`) to prevent the LLM from inheriting concatenated VPIC trim strings.
5. **Graph execution** — initializes `MarketingState`, sets `MAX_ITERATIONS = 5`, `NODES_PER_ITERATION = 4`, `TOTAL_STEPS = 20`, and streams graph events with a `tqdm` progress bar. Each event's node name is displayed in the bar description.
6. **Output** — prints the final `marketing_copy`, `quality_score`, `approved`, and `iteration` values inside a formatted 80-char separator block.

#### Constants in `main.py`

| Constant | Value | Purpose |
|---|---|---|
| `VPIC_URL` | `https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/{vin}?format=json` | NHTSA decode endpoint |
| `NULL_VALUES` | `{"", "Not Applicable", "null", "None", "N/A"}` | Values treated as absent |
| `MAX_ITERATIONS` | `5` | Maximum retry loops |
| `NODES_PER_ITERATION` | `4` | Progress bar denominator |
| `TOTAL_STEPS` | `20` | Total tqdm steps |

---

### `state.py`

Defines the single shared state object passed through every node.

```python
class MarketingState(TypedDict):
    vehicle: dict          # All parsed VPIC + normalized fields + safety sub-dict
    marketing_copy: str    # Generated or refined 4-paragraph description
    review_feedback: Any   # Structured dict returned by reviewer_node
    quality_score: int     # 0–100, computed deterministically by judge_node
    approved: bool         # True when score >= APPROVAL_THRESHOLD and no critical issues
    iteration: int         # Incremented by judge_node on every pass
```

All nodes receive this dict and return an updated copy of it.

---

### `graph.py`

Builds the `StateGraph` and exports the compiled `graph` object.

```python
builder = StateGraph(MarketingState)

builder.add_node("generator", generator_node)
builder.add_node("reviewer",  reviewer_node)
builder.add_node("refiner",   refiner_node)
builder.add_node("judge",     judge_node)

builder.add_edge(START,       "generator")
builder.add_edge("generator", "reviewer")
builder.add_edge("reviewer",  "judge")
builder.add_edge("refiner",   "reviewer")   # refiner always feeds back into reviewer

builder.add_conditional_edges("judge", route_after_judge)

graph = builder.compile()
```

`MAX_ITERATIONS = 5` is defined here and used by `route_after_judge`. The `refiner → reviewer` edge (not `refiner → judge`) ensures the reviewer always re-validates after every refinement.

---

### `llm.py`

Initializes the shared Azure OpenAI client used by all agents.

```python
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    temperature=0.3,
)
```

- `temperature=0.3` keeps output mostly deterministic while allowing natural language variation.
- Credentials are loaded from a `.env` file via `python-dotenv`.
- All four agent nodes import and share this single `llm` instance.

---

### `agents/generator.py`

Generates the initial 4-paragraph vehicle description from VPIC data.

#### Helper functions

| Function | Purpose |
|---|---|
| `_val(v)` | Strips whitespace; returns `""` for null-like values (`none`, `null`, `not applicable`, `n/a`, `no`) |
| `_build_status_features(safety)` | Returns `(standard_features, optional_features)` lists by reading VPIC status values (`"Standard"` / `"Optional"`) from the safety dict |
| `_build_standard_features(safety)` | Thin wrapper returning only the standard list from `_build_status_features` |
| `_build_optional_features(safety)` | Thin wrapper returning only the optional list |
| `_list_to_prose(items)` | Converts a list to natural English: `["a", "b", "c"]` → `"a, b, and c"` |

#### Feature classification logic

`_build_status_features` processes the `safety` dict with these rules:

- **Skipped keys** (handled specially outside the loop): `tpms`, `backup_camera`, `rear_visibility_system`, `front_airbags`, `side_airbags`, `curtain_airbags`, `knee_airbags`, `seat_belt_type`, `other_restraint_info`, `brake_system_type`, `headlamp_light_source`.
- For remaining keys, the value must be exactly `"standard"` or `"optional"` (case-insensitive) to be included.
- Human-readable labels are applied via `label_overrides` (e.g. `"esc"` → `"electronic stability control"`).
- **TPMS**: Added to standard features if any non-null value is present (VPIC returns type strings like `"Direct"` rather than Standard/Optional).
- **Rear visibility**: Added as `"a rear visibility system"` to standard or optional based on the value of `backup_camera` or `rear_visibility_system`; whichever is non-null is used.
- **Headlamps**: Added as `"{source} headlamps"` (e.g. `"LED headlamps"`) if `headlamp_light_source` is present.
- **Airbags**: Combined into a single prose phrase: `"front and side and curtain airbags for added peace of mind"` from whichever airbag location keys are present.

#### Engine description assembly

```python
turbo_prefix = "turbocharged " if turbo.lower() == "yes" else ""
inline_cfg   = "inline " if "in-line" in engine_cfg.lower() else (engine_cfg.lower() + " " if engine_cfg else "")
engine_desc  = f"{turbo_prefix}{disp_l}L {inline_cfg}{cylinders}-cylinder".strip()
```

Example outputs:
- `"turbocharged 2.0L inline 4-cylinder"`
- `"3.5L V 6-cylinder"`
- `""` (if displacement or cylinders are absent)

#### Prompt rules enforced

The generator prompt includes 8 explicit rules:
1. Model name must always have spaces between letters and numbers (e.g. `4 XE` not `4XE`).
2. Do NOT include any year, date, plant location, list price, or base price.
3. Do NOT add `-Class` suffix to the model name.
4. Do NOT invent features — use only confirmed VPIC data.
5. Do NOT mention kW — use only HP.
6. Paragraph structure:
   - Para 1: Opening sentence introducing the vehicle + powertrain
   - Para 2: Body style description + standard safety/tech features
   - Para 3: Brake system + optional/available ADAS features
   - Para 4: Closing value statement (no color or mileage)
7. Include transmission details in the powertrain description if present.
8. Output only the 4-paragraph description — no preamble, no notes.

---

### `agents/reviewer.py`

Validates the marketing copy against VPIC data and returns structured JSON feedback.

#### Helper functions

| Function | Purpose |
|---|---|
| `extract_json(text)` | Strips markdown code fences, attempts `json.loads`, then falls back to regex `\{.*\}` extraction |

#### Validation checks

The reviewer prompt asks the LLM to evaluate 8 binary (yes/no) flags:

| Flag | What it checks |
|---|---|
| `model_name_present` | Does the exact model name appear in the copy? |
| `model_name_correct_spacing` | Is there a space at every letter↔digit boundary (e.g. `"4 XE"` not `"4XE"`)? |
| `duplicate_trim_detected` | Does the trim string appear twice in a row (e.g. `"4 XE 4 XE"`)? |
| `duplicate_model_detected` | Is the model name written twice unnecessarily? |
| `contains_year_date` | Does the copy mention any calendar year (e.g. 2023) or date? |
| `contains_price` | Does the copy mention any price, MSRP, or dollar amount? |
| `contains_plant_location` | Does the copy mention a plant city, factory, or manufacturing location? |
| `contains_hallucinated_data` | Does the copy claim anything not present in the VPIC data? |

The JSON response also includes:
- `hallucination_examples: []` — list of specific hallucinated claims
- `summary: "..."` — one-sentence summary of the main issues or "No issues found"

#### Fallback on parse failure

If `extract_json` returns `None`, reviewer sets all flags to worst-case values (all issues present) so the judge will always retry rather than silently approving bad output:

```python
result = {
    "model_name_present":         "no",
    "model_name_correct_spacing": "no",
    "duplicate_trim_detected":    "yes",
    "duplicate_model_detected":   "yes",
    "contains_year_date":         "yes",
    "contains_price":             "yes",
    "contains_plant_location":    "yes",
    "contains_hallucinated_data": "yes",
    "hallucination_examples":     ["review parse failure"],
    "summary":                    "review parse failure",
}
```

#### Console output

After each review, the node prints a single summary line:
```
[reviewer] Model=yes | Spacing=yes | DupTrim=no | Hallucinations=no | No issues found
```

---

### `agents/refiner.py`

Rewrites the current copy to fix all issues flagged by the reviewer.

#### Helper functions

Same utility functions as `generator.py`, plus:

| Function | Purpose |
|---|---|
| `_is_yes(value)` | Returns `True` for boolean `True` or string `"yes"` (case-insensitive) |
| `_hallucination_text(value)` | Flattens a list or string of hallucination examples to a semicolon-separated string |
| `_build_status_features(safety)` | Identical implementation to generator — rebuilds the standard/optional feature lists |

#### Fix instruction generation

The refiner reads the reviewer's feedback dict and dynamically builds a numbered list of targeted fix instructions. One fix block is added per failed flag:

| Failed flag | Fix instruction |
|---|---|
| `model_name_present == "no"` | ADD the model name — must appear as `{full_name}` |
| `model_name_correct_spacing == "no"` | FIX spacing — show wrong (`{trim_raw}`) vs correct (`{trim}`) forms |
| `duplicate_trim_detected == "yes"` | REMOVE the duplicate trim with wrong/correct examples |
| `duplicate_model_detected == "yes"` | REMOVE the duplicated model name |
| `contains_year_date == "yes"` | REMOVE all year and date references |
| `contains_price == "yes"` | REMOVE all pricing — MSRP, list, base |
| `contains_plant_location == "yes"` | REMOVE all plant/factory/manufacturing location references |
| `contains_hallucinated_data == "yes"` | REMOVE hallucinated content, lists specific examples |

If there are no failing flags, the fixes block defaults to `"Improve overall quality and flow."`.

#### Absolute rules

The refiner prompt repeats the same 5 non-negotiable rules as the generator:
1. Model name must appear as `{full_name}` — never the raw un-spaced form.
2. No years, no dates, no pricing, no plant locations.
3. No `-Class` suffix.
4. Only VPIC-confirmed features.
5. Exactly 4 paragraphs, flowing prose only, no bullet points, no bold, no headers.

The full vehicle VPIC data (including standard and optional feature lists) is passed into every refinement prompt so the LLM never loses context between iterations.

---

### `agents/judge.py`

Scores the copy deterministically from the reviewer's JSON flags and sets `approved`.

#### Scoring algorithm

The judge starts from a perfect score of 100 and applies deductions:

| Check | Effect |
|---|---|
| `model_name_present == "no"` | **Score = 0** (critical disqualification) |
| `model_name_correct_spacing == "no"` | **Score = 0** (critical disqualification) |
| `duplicate_trim_detected == "yes"` | −40 points |
| `duplicate_model_detected == "yes"` | −30 points |
| `contains_year_date == "yes"` | −40 points |
| `contains_price == "yes"` | −40 points |
| `contains_plant_location == "yes"` | −30 points |
| `contains_hallucinated_data == "yes"` | −30 points |

Score is floored at 0 (`max(score, 0)`).

#### Approval logic

```python
approved = (score >= APPROVAL_THRESHOLD) and (critical_count == 0)
```

- `APPROVAL_THRESHOLD = 90`
- A copy passes only if **score ≥ 90 AND no critical flags were raised**.
- This means a copy with correct model name and spacing but with a single year reference scores 60 (100 − 40) and is rejected.

#### State updates

At the end of every judge pass:
```python
state["quality_score"] = score
state["approved"]      = approved
state["iteration"]     += 1
```

#### Console output

```
[judge] Iteration 2 | Score: 100/100 | Approved: True
[judge] No issues found.
```
or:
```
[judge] Iteration 1 | Score: 60/100 | Approved: False
[judge] Issues:
   - copy contains a year or date reference
   - hallucinated content: adaptive cruise control at 150 mph
```

---

## VPIC Field Mapping

`main.py` maps NHTSA vPIC `Variable` names to internal keys. The full mapping:

| VPIC Variable | Internal Key |
|---|---|
| Make | `make` |
| Model | `model_raw` |
| Model Year | `year` |
| Trim | `trim_raw` |
| Trim2 | `trim2` |
| Series | `series` |
| Body Class | `body_class` |
| Vehicle Type | `vehicle_type` |
| Doors | `doors` |
| Drive Type | `drive_type` |
| Transmission Style | `transmission` |
| Transmission Speeds | `transmission_speeds` |
| Engine Number of Cylinders | `engine_cylinders` |
| Displacement (L) | `displacement_l` |
| Displacement (CC) | `displacement_cc` |
| Engine Brake (hp) From | `engine_power_hp` |
| Engine Power (kW) | `engine_power_kw` |
| Engine Configuration | `engine_config` |
| Turbo | `turbo` |
| Fuel Delivery / Fuel Injection Type | `fuel_injection` |
| Other Engine Info | `other_engine_info` |
| Fuel Type - Primary | `fuel_primary` |
| Fuel Type - Secondary | `fuel_secondary` |
| Electrification Level | `electrification` |
| Anti-lock Braking System (ABS) | `abs` |
| Electronic Stability Control (ESC) | `esc` |
| Traction Control | `traction_control` |
| Tire Pressure Monitoring System (TPMS) Type | `tpms` |
| Backup Camera | `backup_camera` |
| Rear Visibility System | `rear_visibility_system` |
| Keyless Ignition | `keyless_ignition` |
| Daytime Running Light (DRL) | `drl` |
| Brake System Type | `brake_system_type` |
| Blind Spot Warning (BSW) | `blind_spot_warning` |
| Forward Collision Warning (FCW) | `fcw` |
| Crash Imminent Braking (CIB) | `cib` |
| Dynamic Brake Support (DBS) | `dbs` |
| Adaptive Cruise Control (ACC) | `acc` |
| Lane Departure Warning (LDW) | `ldw` |
| Lane Keeping Assistance (LKA) | `lka` |
| Lane Centering Assistance | `lane_centering_assist` |
| Pedestrian Automatic Emergency Braking (PAEB) | `paeb` |
| Blind Spot Intervention (BSI) | `bsi` |
| Parking Assist | `parking_assist` |
| Rear Cross Traffic Alert | `rcta` |
| Rear Automatic Emergency Braking | `rear_aeb` |
| Headlamp Light Source | `headlamp_light_source` |
| Semiautomatic Headlamp Beam Switching | `semi_auto_headlamps` |
| Adaptive Driving Beam (ADB) | `adb` |
| Front Air Bag Locations | `front_airbags` |
| Side Air Bag Locations | `side_airbags` |
| Curtain Air Bag Locations | `curtain_airbags` |
| Knee Air Bag Locations | `knee_airbags` |
| Seat Belt Type | `seat_belt_type` |
| Other Restraint System Info | `other_restraint_info` |
| Gross Vehicle Weight Rating From | `gvwr` |

---

## Vehicle Object Schema

`build_vehicle()` in `main.py` produces this object (passed as `state["vehicle"]`):

```python
{
    "vin":                 str,   # Uppercased input VIN
    "year":                str,   # Model year
    "make":                str,   # e.g. "Jeep"
    "model":               str,   # Normalized model with spaces (e.g. "Wrangler 4 XE")
    "model_raw":           str,   # Raw VPIC model string (e.g. "Wrangler 4XE")
    "trim":                str,   # Normalized trim with spaces
    "trim_raw":            str,   # Raw VPIC trim string
    "full_name":           str,   # "Make Model Trim" — deduped if trim is in model
    "series":              str,
    "body_class":          str,   # e.g. "Sport Utility Vehicle (SUV)"
    "vehicle_type":        str,   # e.g. "MULTIPURPOSE PASSENGER VEHICLE (MPV)"
    "doors":               str,   # e.g. "4"
    "drive_type":          str,   # e.g. "4WD/4-Wheel Drive/4x4"
    "transmission":        str,   # e.g. "Automatic"
    "transmission_speeds": str,   # e.g. "8"
    "engine_cylinders":    str,   # e.g. "4"
    "displacement_l":      str,   # e.g. "2.0"
    "displacement_cc":     str,   # e.g. "1995.0"
    "engine_power_hp":     str,   # e.g. "375"
    "engine_power_kw":     str,   # e.g. "280"
    "engine_config":       str,   # e.g. "In-Line"
    "turbo":               str,   # "Yes" or ""
    "other_engine_info":   str,
    "fuel_primary":        str,   # e.g. "Gasoline"
    "fuel_secondary":      str,   # e.g. "Electric"
    "electrification":     str,   # e.g. "PHEV (Plug-in Hybrid Electric Vehicle)"
    "safety": {
        "abs":                    str,
        "esc":                    str,
        "traction_control":       str,
        "tpms":                   str,
        "backup_camera":          str,
        "rear_visibility_system": str,
        "keyless_ignition":       str,
        "drl":                    str,
        "brake_system_type":      str,
        "blind_spot_warning":     str,
        "bsi":                    str,
        "fcw":                    str,
        "cib":                    str,
        "dbs":                    str,
        "acc":                    str,
        "ldw":                    str,
        "lka":                    str,
        "lane_centering_assist":  str,
        "paeb":                   str,
        "parking_assist":         str,
        "rcta":                   str,
        "rear_aeb":               str,
        "semi_auto_headlamps":    str,
        "adb":                    str,
        "headlamp_light_source":  str,
        "front_airbags":          str,
        "side_airbags":           str,
        "curtain_airbags":        str,
        "knee_airbags":           str,
        "seat_belt_type":         str,
        "other_restraint_info":   str,
    },
    "dealer_name":   "",   # Placeholder — not populated by VPIC
    "dealer_blurb":  "",   # Placeholder — not populated by VPIC
}
```

#### `full_name` deduplication logic

```python
model_no_space = model_norm.lower().replace(" ", "")
trim_no_space  = trim_norm.lower().replace(" ", "")
if trim_norm and trim_no_space not in model_no_space:
    full_name = f"{make} {model_norm} {trim_norm}".strip()
else:
    full_name = f"{make} {model_norm}".strip()
```

This prevents redundancy when VPIC's `Model` field already contains the trim string (common for some manufacturers).

---

## State Schema

`state.py` defines `MarketingState` as a `TypedDict`:

| Field | Type | Initial Value | Description |
|---|---|---|---|
| `vehicle` | `dict` | Result of `build_vehicle()` | Complete VPIC-decoded vehicle data |
| `marketing_copy` | `str` | `""` | The 4-paragraph marketing description — updated by generator and refiner |
| `review_feedback` | `Any` | `{}` | Structured dict with 8 boolean flags + hallucination examples + summary — set by reviewer |
| `quality_score` | `int` | `0` | 0–100 score computed deterministically by judge |
| `approved` | `bool` | `False` | `True` when score ≥ 90 and no critical flags |
| `iteration` | `int` | `0` | Incremented by judge on each pass; caps at `MAX_ITERATIONS` |

---

## Safety & ADAS Feature Handling

Safety and ADAS features are handled differently depending on the VPIC field type:

### Standard/Optional status fields
These keys hold values like `"Standard"`, `"Optional"`, or `"Not Applicable"`:
- `abs`, `esc`, `traction_control`, `keyless_ignition`, `drl`
- `blind_spot_warning`, `bsi`, `fcw`, `cib`, `dbs`, `acc`, `ldw`, `lka`
- `lane_centering_assist`, `paeb`, `parking_assist`, `rcta`, `rear_aeb`
- `semi_auto_headlamps`, `adb`

### Type/value fields (handled specially)
| Field | VPIC value type | Handling |
|---|---|---|
| `tpms` | Type string e.g. `"Direct"` | Added to standard if any non-null value |
| `backup_camera` / `rear_visibility_system` | Status or type | Combined: if either is non-null, adds `"a rear visibility system"` |
| `headlamp_light_source` | Source string e.g. `"LED"` | Appended as `"LED headlamps"` |
| `front_airbags`, `side_airbags`, `curtain_airbags`, `knee_airbags` | Location string | Combined into `"front and side and curtain airbags for added peace of mind"` |
| `brake_system_type` | Type string | Passed directly to the prompt (not to feature lists) |
| `seat_belt_type`, `other_restraint_info` | Strings | Passed directly to the prompt |

---

## Scoring & Approval Model

The judge in `agents/judge.py` uses a **deduction-based** model:

```
Starting score: 100

Critical failures (immediately set score to 0):
  - model name missing from copy
  - model name spacing incorrect

Major deductions (applied cumulatively):
  - duplicate trim name:        −40
  - duplicate model name:       −30
  - year/date reference:        −40
  - pricing information:        −40
  - plant/factory location:     −30
  - hallucinated content:       −30

Floor: max(score, 0)

Approved if: score >= 90 AND critical_count == 0
```

**Maximum achievable deduction without critical failure:** −210, floored to 0.

**Why 90?** The threshold was set high (changed from 85 to 90 in `agents/judge.py`) to require the copy to pass nearly all checks. A single year reference (−40) produces a score of 60, well below the threshold.

---

## Prerequisites

- Python **3.10+** (uses `tuple[list, list]` union type hints, `dict | None`, and match/walrus patterns in internal logic)
- An **Azure OpenAI** resource with a deployed chat model (e.g. `gpt-4o`, `gpt-4`, `gpt-35-turbo`)
- Active internet access to reach `vpic.nhtsa.dot.gov`

---

## Installation

```bash
git clone https://github.com/AyushArtesian/LangGraph-Project.git
cd LangGraph-Project
pip install -r requirements.txt
```

Or install in a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root (it is `.gitignore`d):

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
AZURE_OPENAI_API_KEY=<your-api-key>
```

| Variable | Description |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | The model deployment name (e.g. `gpt-4o`) |
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI API key |

The `llm.py` client uses `api_version="2024-12-01-preview"` and `temperature=0.3`.

---

## Running the Pipeline

```bash
python main.py
```

You will be prompted:
```
Enter VIN number:
```

Enter any valid 17-character VIN (e.g. `1C4HJXDG2PW502696`). The pipeline will:

1. Fetch and display the VPIC URL being called.
2. Print the resolved vehicle name, trim, and make.
3. Start the LangGraph pipeline with a `tqdm` progress bar.
4. Print inline status from reviewer and judge nodes.
5. Print the final output block.

Example session:
```
Enter VIN number: 1C4HJXDG2PW502696

[vpic] Fetching: https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/1C4HJXDG2PW502696?format=json

[main] Vehicle  : Jeep Wrangler 4 XE
[main] Trim raw : '4XE'  →  normalized: '4 XE'
[main] Make     : Jeep

Starting Marketing Copy Generation...

Pipeline:   5%|█         | 1/20 [00:03<01:02,  3.11s/node] Running: generator
[reviewer] Model=yes | Spacing=yes | DupTrim=no | Hallucinations=no | No issues found

[judge] Iteration 1 | Score: 100/100 | Approved: True
[judge] No issues found.
✅ Approved at iteration 1 with score 100/100

================================================================================
FINAL MARKETING COPY
================================================================================
<4-paragraph vehicle description>

================================================================================
QUALITY SCORE : 100/100
APPROVED      : True
ITERATIONS    : 1
================================================================================
```

---

## Sample Output

For a Jeep Wrangler 4XE, the pipeline produces a 4-paragraph description following this structure:

**Para 1** — Vehicle introduction + powertrain:
> The Jeep Wrangler 4 XE combines legendary Jeep capability with advanced plug-in hybrid technology, delivering an exciting blend of efficiency, power, and off-road confidence. Powered by a turbocharged 2.0L inline 4-cylinder engine paired with an electric drive system, the Wrangler 4 XE produces an impressive 375 horsepower and features a 4WD drivetrain for outstanding traction and control.

**Para 2** — Body style + standard safety features:
> Designed with Jeep's iconic SUV styling, this four-door Wrangler 4 XE offers the versatility and rugged character drivers expect. Key features include keyless ignition, electronic stability control, traction control, anti-lock brakes, daytime running lights, a rear visibility system, tire pressure monitoring, and front and side airbags for added peace of mind.

**Para 3** — Brake system + available ADAS:
> This vehicle is equipped with a hydraulic braking system and features standard safety technologies engineered to enhance driver confidence. Available driver-assistance technologies include blind spot warning, dynamic brake support, forward collision warning, crash imminent braking, and semi-automatic headlamp beam switching.

**Para 4** — Closing value statement:
> The Wrangler 4 XE delivers open-air freedom, adventure-ready capability, and innovative hybrid technology that make it a standout choice for drivers seeking both efficiency and authentic Jeep performance.

---

## Customization

### Change the vehicle
Run the script and enter a different VIN. Any VIN decodable by the NHTSA vPIC API is supported.

### Adjust iteration limit
In `graph.py`:
```python
MAX_ITERATIONS = 5   # increase or decrease
```
Also update `NODES_PER_ITERATION` and `TOTAL_STEPS` in `main.py` to keep the progress bar accurate.

### Change the approval threshold
In `agents/judge.py`:
```python
APPROVAL_THRESHOLD = 90   # lower to 85 for a more lenient gate
```

### Change the scoring weights
Edit the deduction values at the top of `judge_node()` in `agents/judge.py`.

### Add new review checks
1. Add a new key to the reviewer prompt JSON schema.
2. Add a corresponding deduction in `judge_node()`.
3. Add a fix instruction in `refiner_node()`.

### Switch to a different LLM provider
Replace `AzureChatOpenAI` in `llm.py` with any LangChain-compatible chat model (e.g. `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`) and update the corresponding environment variables.

### Add dealer information
The `vehicle` object has placeholder fields `dealer_name` and `dealer_blurb`. Populate them in `build_vehicle()` or after calling it in `main.py`, then reference them in the generator and refiner prompts.

### Tune prompts for different industries
The generator, refiner, and reviewer prompts are plain Python f-strings inside each agent file. Edit them directly to change tone, paragraph structure, or adapt the pipeline to a different product category.

---

## Dependencies

From `requirements.txt`:

| Package | Purpose |
|---|---|
| `langgraph` | StateGraph execution engine, node/edge wiring, streaming |
| `langchain` | Base LangChain abstractions |
| `langchain-openai` | `AzureChatOpenAI` integration |
| `python-dotenv` | `.env` file loading |
| `pydantic` | Data validation (used internally by LangChain) |

`tqdm` and `requests` are used in `main.py` but not listed in `requirements.txt` — install them separately if needed:

```bash
pip install tqdm requests
```

---

## Ignored Files

From `.gitignore`:

| Pattern | Reason |
|---|---|
| `.env` / `.env.*` | API keys and credentials — never committed |
| `__pycache__/` | Python bytecode cache |
| `venv/` | Virtual environment directory |
