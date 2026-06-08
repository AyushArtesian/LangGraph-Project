# LangGraph Automotive Marketing Copy Pipeline

> **Turn any vehicle VIN into professional dealership-quality marketing copy — automatically.**

A production-ready, multi-agent AI pipeline that decodes a Vehicle Identification Number (VIN) using the NHTSA vPIC public API, extracts every confirmed vehicle specification and safety feature, and feeds that structured data through a four-node LangGraph agentic workflow powered by Azure OpenAI. The result is factually grounded, dealership-style marketing copy that passes an automated quality review before being returned to the caller.

---


## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [How It Works — End-to-End Flow](#how-it-works--end-to-end-flow)
4. [Architecture](#architecture)
5. [Project Structure](#project-structure)
6. [Core Components](#core-components)
   - [main.py — CLI & VIN Decoder](#mainpy--cli--vin-decoder)
   - [api.py — FastAPI Service](#apipy--fastapi-service)
   - [graph.py — LangGraph Workflow](#graphpy--langgraph-workflow)
   - [state.py — Shared State Schema](#statepy--shared-state-schema)
   - [llm.py — Azure OpenAI Client](#llmpy--azure-openai-client)
7. [Agent Nodes](#agent-nodes)
   - [Generator Agent](#generator-agent)
   - [Reviewer Agent](#reviewer-agent)
   - [Judge Agent](#judge-agent)
   - [Refiner Agent](#refiner-agent)
8. [VPIC Data Extraction](#vpic-data-extraction)
9. [Quality Scoring System](#quality-scoring-system)
10. [API Reference](#api-reference)
11. [Requirements](#requirements)
12. [Installation](#installation)
13. [Environment Variables](#environment-variables)
14. [Running Locally](#running-locally)
    - [CLI Mode](#cli-mode)
    - [API Mode](#api-mode)
15. [Example Output](#example-output)
16. [Workflow Rules & Constraints](#workflow-rules--constraints)
17. [Deployment](#deployment)
    - [Azure Web App](#azure-web-app)
    - [CI/CD Pipeline](#cicd-pipeline)
18. [Dependencies](#dependencies)
19. [Troubleshooting](#troubleshooting)
20. [Contributing](#contributing)
21. [License](#license)

---

## Overview

This project solves a real-world automotive industry problem: writing accurate, fact-consistent vehicle descriptions at scale. Traditional marketing copy writing is time-consuming, error-prone, and requires deep product knowledge. This pipeline automates the entire process by:

1. Pulling every known specification for a vehicle directly from the U.S. government's NHTSA vPIC database (the same source automakers use to register VINs).
2. Structuring that data into a rich `MarketingState` object.
3. Running a LangGraph multi-agent loop where an AI **Generator**, **Reviewer**, **Judge**, and **Refiner** collaborate iteratively until the copy meets a strict quality threshold.
4. Returning the final approved copy via CLI output or a REST API response.

The pipeline is deployed as an **Azure Web App** with automated CI/CD through GitHub Actions.

---

## Key Features

| Feature | Detail |
|---|---|
| **VIN → Copy in seconds** | Submit any 17-character VIN and receive dealership-quality marketing text |
| **NHTSA vPIC integration** | Decodes 50+ vehicle attributes including engine, drivetrain, safety systems, and airbags |
| **Multi-agent LangGraph loop** | Four specialized AI nodes with deterministic routing logic |
| **Hallucination prevention** | Reviewer agent cross-checks every claim against raw VPIC data |
| **Deterministic quality scoring** | Judge applies a 100-point scoring rubric; copy must score ≥ 90 to be approved |
| **Automatic refinement** | Failed copy is rewritten by the Refiner agent with precise fix instructions |
| **Iteration control** | Configurable max iterations (1–10) prevent infinite loops |
| **FastAPI REST service** | Swagger UI included at `/docs` |
| **CLI interface** | Run interactively from the terminal |
| **Azure deployment** | `startup.sh` + GitHub Actions workflow included |
| **LangSmith tracing** | All agent nodes are instrumented with `@traceable` decorators |
| **Model name normalization** | Automatically inserts spaces between letters and digits (e.g., `4XE` → `4 XE`) |

---

## How It Works — End-to-End Flow

```
User submits VIN
       │
       ▼
① fetch_vpic(vin)
  ├── Calls NHTSA vPIC API
  ├── Extracts 50+ raw vehicle attributes
  └── Returns parsed dict

       │
       ▼
② build_vehicle(vpic, vin)
  ├── Normalizes model and trim names (adds spaces between letter↔digit)
  ├── Builds full display name (avoids duplicating trim in model)
  ├── Separates safety features into a dedicated sub-dict
  └── Returns structured vehicle dict

       │
       ▼
③ LangGraph Workflow (MarketingState)
  │
  ├── [generator] Writes initial 4-paragraph marketing copy
  │       Uses confirmed VPIC specs + standard/optional feature lists
  │
  ├── [reviewer] Checks copy against VPIC data
  │       Returns structured JSON with 8 compliance flags
  │
  ├── [judge] Scores the copy (0–100) and decides: APPROVE or RETRY
  │       ┌── Score ≥ 90 and no critical issues → APPROVE → END
  │       └── Score < 90 or iterations < max   → RETRY
  │                                                   │
  │                                                   ▼
  │                                            [refiner] Rewrites copy
  │                                            with targeted fix instructions
  │                                                   │
  │                                                   └──► [reviewer] (loop)
  │
  └── END: Return final MarketingState

       │
       ▼
④ Response
  ├── CLI: Prints copy + score + approval status + iteration count
  └── API: Returns JSON response
```

---

## Architecture

```
                      ┌─────────────────────────────────────────────────┐
                      │              LangGraph Workflow                  │
                      │                                                  │
  VIN                 │  START                                           │
   │                  │    │                                             │
   ▼                  │    ▼                                             │
NHTSA vPIC API        │  [generator]  ──────────────────►  [reviewer]   │
   │                  │                                        │         │
   ▼                  │                                        ▼         │
fetch_vpic()          │                                     [judge]      │
   │                  │                                    /       \     │
   ▼                  │                              approved?    retry? │
build_vehicle()       │                                  │           │   │
   │                  │                                END        [refiner]
   ▼                  │                                               │   │
MarketingState ───────┤                                               │   │
                      │                                    [reviewer] ◄───┘
                      └─────────────────────────────────────────────────┘
```

### Conditional Routing (judge node)

```python
def route_after_judge(state) -> str:
    if state["approved"]:
        return END                          # ✅ Copy passed quality check
    if state["iteration"] >= state["max_iterations"]:
        return END                          # ❌ Iteration limit reached
    return "refiner"                        # 🔄 Retry with targeted fixes
```

---

## Project Structure

```
LangGraph-Project/
│
├── agents/                          # Individual LangGraph agent nodes
│   ├── generator.py                 # Initial copy generation
│   ├── reviewer.py                  # Factual compliance review
│   ├── judge.py                     # Quality scoring and approval
│   └── refiner.py                   # Copy correction and rewriting
│
├── .github/
│   └── workflows/
│       └── main_langgraph-project.yml   # Azure Web App CI/CD pipeline
│
├── api.py                           # FastAPI application and endpoints
├── graph.py                         # LangGraph node registration and routing
├── llm.py                           # Azure OpenAI client configuration
├── main.py                          # CLI entry point + VIN decoding logic
├── requirements.txt                 # Python package dependencies
├── startup.sh                       # Production startup script (Gunicorn)
├── state.py                         # Shared MarketingState TypedDict schema
└── .gitignore                       # Excludes .env, __pycache__, venv
```

---

## Core Components

### `main.py` — CLI & VIN Decoder

The CLI entry point and the heart of the VIN-to-vehicle-object pipeline.

**Responsibilities:**
- Accepts a VIN from the user via `input()`
- Calls `fetch_vpic(vin)` to retrieve raw data from the NHTSA vPIC REST API
- Calls `build_vehicle(vpic, vin)` to normalize and structure the data
- Initializes `MarketingState` and streams it through the LangGraph workflow
- Displays a `tqdm` progress bar tracking each node execution
- Prints the final marketing copy, quality score, approval status, and iteration count

**Key functions:**

| Function | Purpose |
|---|---|
| `fetch_vpic(vin)` | Calls `https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/{vin}?format=json`, maps 50+ VPIC fields to normalized keys, warns on non-zero error codes |
| `normalize_spacing(text)` | Regex to insert a space between any letter↔digit boundary (`GLC300` → `GLC 300`, `4XE` → `4 XE`) |
| `build_vehicle(vpic, vin)` | Assembles the complete vehicle dict including full display name and a nested `safety` sub-dict |

**VPIC fields extracted** (50+ attributes):

```
make, model, year, trim, body_class, vehicle_type, doors, drive_type,
transmission, transmission_speeds, engine_cylinders, displacement_l,
displacement_cc, engine_power_hp, engine_power_kw, engine_config,
turbo, fuel_injection, fuel_primary, fuel_secondary, electrification,
abs, esc, traction_control, tpms, backup_camera, rear_visibility_system,
keyless_ignition, drl, brake_system_type, blind_spot_warning, bsi,
fcw, cib, dbs, acc, ldw, lka, lane_centering_assist, paeb,
parking_assist, rcta, rear_aeb, headlamp_light_source, semi_auto_headlamps,
adb, front_airbags, side_airbags, curtain_airbags, knee_airbags,
seat_belt_type, other_restraint_info, gvwr
```

---

### `api.py` — FastAPI Service

A production-ready REST API exposing the marketing copy pipeline over HTTP.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check — returns service name and `"status": "healthy"` |
| `POST` | `/generate-marketing-copy` | Accepts a VIN + optional iteration limit, runs the full pipeline, returns structured JSON |
| `GET` | `/docs` | Auto-generated interactive Swagger UI |
| `GET` | `/redoc` | Auto-generated ReDoc documentation |

**Request model (`VINRequest`):**

```python
class VINRequest(BaseModel):
    vin: str                         # 17-character vehicle VIN
    max_iterations: int = Field(     # 1–10, default 5
        default=5, ge=1, le=10
    )
```

**Response model (`MarketingCopyResponse`):**

```python
class MarketingCopyResponse(BaseModel):
    vin: str                  # Uppercase VIN as submitted
    vehicle_name: str         # Full normalized display name
    marketing_copy: str       # Approved marketing text
    quality_score: int        # Final score (0–100)
    approved: bool            # Whether the copy passed quality threshold
    iterations: int           # Number of generation/refinement cycles used
```

**Error handling:**

| HTTP Status | Condition |
|---|---|
| `400 Bad Request` | Empty or missing VIN |
| `404 Not Found` | VPIC returned no data for the VIN |
| `500 Internal Server Error` | Graph execution failure or unexpected exception |

---

### `graph.py` — LangGraph Workflow

Defines and compiles the stateful LangGraph workflow graph.

```python
builder = StateGraph(MarketingState)

# Register nodes
builder.add_node("generator", generator_node)
builder.add_node("reviewer",  reviewer_node)
builder.add_node("refiner",   refiner_node)
builder.add_node("judge",     judge_node)

# Define edges
builder.add_edge(START,       "generator")   # Entry point
builder.add_edge("generator", "reviewer")    # Always review after generation
builder.add_edge("reviewer",  "judge")       # Always judge after review
builder.add_edge("refiner",   "reviewer")    # Always re-review after refinement

# Conditional routing from judge
builder.add_conditional_edges("judge", route_after_judge)

graph = builder.compile()
```

The `route_after_judge` function checks `state["approved"]` and `state["iteration"]` to decide between `END` and `"refiner"`.

---

### `state.py` — Shared State Schema

The single shared data structure passed between all LangGraph nodes.

```python
class MarketingState(TypedDict):
    vehicle: dict          # Full parsed VPIC data + dealer info
    marketing_copy: str    # The generated/refined marketing description
    review_feedback: Any   # Structured JSON dict returned by the reviewer
    quality_score: int     # 0–100 score assigned by the judge
    approved: bool         # Whether quality threshold was met
    iteration: int         # Current iteration count (incremented by judge)
    max_iterations: int    # Maximum allowed iterations before forced END
```

---

### `llm.py` — Azure OpenAI Client

Configures and exposes the shared `AzureChatOpenAI` instance used by all agent nodes.

```python
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    temperature=0.3,          # Low temperature for factual consistency
)
```

The `temperature=0.3` setting is intentionally low to minimize creative hallucination while still producing natural marketing language.

---

## Agent Nodes

### Generator Agent

**File:** `agents/generator.py`  
**LangSmith trace name:** `Generator Agent`

The generator creates the first draft of a four-paragraph vehicle description using structured VPIC data.

**Internal helpers:**

| Helper | Purpose |
|---|---|
| `_val(v)` | Strips whitespace, returns empty string for null/N/A/None values |
| `_build_status_features(safety)` | Partitions VPIC safety features into `Standard` and `Optional` lists |
| `_build_standard_features(safety)` | Returns only confirmed Standard features |
| `_build_optional_features(safety)` | Returns only confirmed Optional features |
| `_list_to_prose(items)` | Converts a Python list to natural comma-separated English prose with Oxford comma |

**Feature processing logic:**
- Safety features with VPIC status `"Standard"` are written as **included** in the vehicle
- Safety features with VPIC status `"Optional"` are written as **available**
- TPMS is always included if any value is present (VPIC uses type strings like `"Direct"` rather than Standard/Optional)
- Rear visibility/backup camera is auto-detected from either the `backup_camera` or `rear_visibility_system` field
- Airbag types (front, side, curtain, knee) are combined into a single prose sentence

**Paragraph structure enforced by the prompt:**

| Paragraph | Content |
|---|---|
| 1 | Opening sentence introducing the vehicle + powertrain description (engine, HP, drivetrain, transmission) |
| 2 | Body style description + complete standard safety/tech features list |
| 3 | Brake system type + available (optional) ADAS features |
| 4 | Closing value statement (no year, no color, no mileage — those are not in VPIC) |

**Strict prompt rules:**
- Model name must always include a space between letters and numbers
- No years, no dates, no plant location, no pricing
- No `-Class` suffix on model names
- No invented features — only confirmed VPIC data
- HP only (never kW)

---

### Reviewer Agent

**File:** `agents/reviewer.py`  
**LangSmith trace name:** `Reviewer Agent`

The reviewer performs automated compliance checking by sending the marketing copy and the full VPIC dataset to the LLM and asking it to return a structured JSON verdict.

**JSON response schema:**

```json
{
  "model_name_present": "yes|no",
  "model_name_correct_spacing": "yes|no",
  "duplicate_trim_detected": "yes|no",
  "duplicate_model_detected": "yes|no",
  "contains_year_date": "yes|no",
  "contains_price": "yes|no",
  "contains_plant_location": "yes|no",
  "contains_hallucinated_data": "yes|no",
  "hallucination_examples": [],
  "summary": "brief summary"
}
```

**Checks performed:**

| Check | Description |
|---|---|
| `model_name_present` | The vehicle's model name must appear in the copy |
| `model_name_correct_spacing` | Spaces must separate letters from digits in the model/trim name |
| `duplicate_trim_detected` | Trim name must not appear twice in a row (e.g., `Wrangler 4XE 4XE`) |
| `duplicate_model_detected` | Model name must not be unnecessarily repeated |
| `contains_year_date` | No model year or calendar year allowed |
| `contains_price` | No pricing, MSRP, or dollar amounts allowed |
| `contains_plant_location` | No manufacturing plant or factory location allowed |
| `contains_hallucinated_data` | No features, specs, or claims absent from VPIC |

**Fallback behavior:** If the LLM response cannot be parsed as valid JSON, all checks default to the most restrictive values (all failures), forcing the refiner to rewrite the copy.

**Feature name equivalents (reviewer is instructed never to flag these):**

```
FCW  = Forward Collision Warning
CIB  = Crash Imminent Braking
DBS  = Dynamic Brake Support
PAEB = Pedestrian Automatic Emergency Braking
ACC  = Adaptive Cruise Control
LDW  = Lane Departure Warning
LKA  = Lane Keeping Assistance
BSW  = Blind Spot Warning
BSI  = Blind Spot Intervention
ADB  = Adaptive Driving Beam
Backup Camera = Rear Visibility System
LED Headlamps = Headlamp Light Source LED
```

---

### Judge Agent

**File:** `agents/judge.py`  
**LangSmith trace name:** `Judge Agent`

The judge applies a deterministic, rule-based scoring algorithm to the reviewer's JSON output and decides whether the copy is approved.

**Scoring rubric:**

| Condition | Score Impact |
|---|---|
| Start score | 100 |
| Model name missing | Score → 0 (CRITICAL) |
| Model name spacing incorrect | Score → 0 (CRITICAL) |
| Duplicate trim detected | −40 |
| Duplicate model detected | −30 |
| Year or date present | −40 |
| Price present | −40 |
| Plant location present | −30 |
| Hallucinated data present | −30 |
| Minimum floor | 0 (score cannot go negative) |

**Approval condition:**

```python
APPROVAL_THRESHOLD = 90

approved = (score >= APPROVAL_THRESHOLD) and (critical_count == 0)
```

The copy must **both** reach the 90-point threshold **and** have zero critical violations to be approved.

The judge increments `state["iteration"]` before returning, which the routing function checks against `max_iterations`.

---

### Refiner Agent

**File:** `agents/refiner.py`  
**LangSmith trace name:** `Refiner Agent`

The refiner receives the current marketing copy, the judge's score, and the reviewer's feedback flags, and rewrites the copy with precise, targeted fix instructions embedded in the prompt.

**Fix instructions generated per flag:**

| Reviewer Flag | Fix Instruction Injected |
|---|---|
| `model_name_present: no` | ADD the model name explicitly as `{full_name}` |
| `model_name_correct_spacing: no` | FIX spacing — show WRONG vs CORRECT examples |
| `duplicate_trim_detected: yes` | REMOVE duplicate trim — show WRONG vs CORRECT |
| `duplicate_model_detected: yes` | REMOVE the extra model name occurrence |
| `contains_year_date: yes` | REMOVE all year/date references |
| `contains_price: yes` | REMOVE all pricing (MSRP, dollar amounts, base price) |
| `contains_plant_location: yes` | REMOVE factory/plant/manufacturing references |
| `contains_hallucinated_data: yes` | REMOVE specific hallucinations listed in examples |

If no flags are set, the refiner is instructed to `"Improve overall quality and flow."` — this handles edge cases where the judge score is below threshold but the reviewer found no specific violations.

The refiner receives the same VPIC vehicle data as the generator to ensure it uses only confirmed features when rewriting.

---

## VPIC Data Extraction

The pipeline uses the **NHTSA vPIC (Vehicle Product Information Catalog) API** — a free, public REST API that returns official vehicle data for any valid VIN.

**API endpoint:**
```
GET https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/{VIN}?format=json
```

**Field mapping** (`VPIC_FIELDS` dict in `main.py`):

The raw API returns a flat list of `{Variable, Value}` pairs. `fetch_vpic()` maps 50+ official VPIC variable names to clean snake_case keys:

```python
"Make"                                   → "make"
"Model"                                  → "model_raw"
"Model Year"                             → "year"
"Trim"                                   → "trim_raw"
"Body Class"                             → "body_class"
"Drive Type"                             → "drive_type"
"Engine Number of Cylinders"             → "engine_cylinders"
"Displacement (L)"                       → "displacement_l"
"Engine Brake (hp) From"                 → "engine_power_hp"
"Electrification Level"                  → "electrification"
"Anti-lock Braking System (ABS)"         → "abs"
"Electronic Stability Control (ESC)"     → "esc"
"Forward Collision Warning (FCW)"        → "fcw"
"Adaptive Cruise Control (ACC)"          → "acc"
"Lane Departure Warning (LDW)"           → "ldw"
"Pedestrian Automatic Emergency Braking" → "paeb"
"Front Air Bag Locations"                → "front_airbags"
# ... and 30+ more
```

Null values (`""`, `"Not Applicable"`, `"null"`, `"None"`, `"N/A"`) are filtered out during extraction.

---

## Quality Scoring System

The pipeline uses a deterministic, non-LLM scoring system to evaluate copy quality. This ensures consistent, reproducible scoring that cannot be manipulated by the LLM.

```
Score starts at:  100
                   │
    ┌──────────────┼──────────────────────────────────┐
    │              │                                  │
CRITICAL        MAJOR                              MINOR
(Score → 0)   (Large deductions)              (No deductions)
    │              │
Model missing  Year/date:   −40
Bad spacing    Price:        −40
               Dup trim:     −40
               Hallucination: −30
               Dup model:    −30
               Plant loc:    −30
                              │
                        Floor: 0
                              │
                    Threshold: 90 to APPROVE
```

**Score examples:**

| Violations | Score | Approved? |
|---|---|---|
| None | 100 | ✅ Yes |
| Hallucination only | 70 | ❌ No |
| Year + Price | 20 | ❌ No |
| Model name missing | 0 | ❌ No (CRITICAL) |
| Duplicate trim | 60 | ❌ No |

---

## API Reference

### `GET /`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "LangGraph Automotive Marketing Copy API"
}
```

---

### `POST /generate-marketing-copy`

Generate marketing copy for a vehicle VIN.

**Request body:**
```json
{
  "vin": "1C4JJXP66PW500000",
  "max_iterations": 5
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `vin` | `string` | ✅ Yes | — | 17-character Vehicle Identification Number |
| `max_iterations` | `integer` | No | `5` | Maximum refinement iterations (1–10) |

**Success response (200):**
```json
{
  "vin": "1C4JJXP66PW500000",
  "vehicle_name": "Jeep Wrangler 4 XE",
  "marketing_copy": "The Jeep Wrangler 4 XE combines legendary Jeep capability with advanced plug-in hybrid technology...",
  "quality_score": 100,
  "approved": true,
  "iterations": 1
}
```

| Field | Type | Description |
|---|---|---|
| `vin` | `string` | Uppercase VIN as submitted |
| `vehicle_name` | `string` | Normalized full vehicle display name |
| `marketing_copy` | `string` | The approved 4-paragraph marketing description |
| `quality_score` | `integer` | Final quality score (0–100) |
| `approved` | `boolean` | `true` if score ≥ 90 and no critical violations |
| `iterations` | `integer` | Number of generation + refinement cycles used |

**Error responses:**

| Status | Body | Cause |
|---|---|---|
| `400` | `{"detail": "VIN is required"}` | Empty VIN submitted |
| `404` | `{"detail": "Unable to decode VIN"}` | VPIC returned no data |
| `500` | `{"detail": "..."}` | Graph execution or internal failure |

---

## Requirements

- **Python:** 3.12 or higher (CI runs on 3.14)
- **Azure OpenAI:** An active Azure subscription with a deployed chat model
- **Internet access:** Required to reach `vpic.nhtsa.dot.gov` at runtime

---

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/AyushArtesian/LangGraph-Project.git
cd LangGraph-Project
```

**2. Create and activate a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_KEY=your-api-key-here

# Optional: LangSmith tracing (set to enable agent observability)
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=LangGraph-Project
LANGCHAIN_TRACING_V2=true
```

| Variable | Required | Description |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | ✅ Yes | Full Azure OpenAI resource endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | ✅ Yes | Your deployed model name (e.g., `gpt-4o`) |
| `AZURE_OPENAI_API_KEY` | ✅ Yes | Azure OpenAI API key |
| `LANGSMITH_API_KEY` | No | Enables LangSmith agent tracing |
| `LANGSMITH_PROJECT` | No | LangSmith project name for trace grouping |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to activate LangSmith tracing |

> ⚠️ **Security:** The `.env` file is listed in `.gitignore` and will never be committed to the repository.

---

## Running Locally

### CLI Mode

Run the interactive command-line pipeline:

```bash
python main.py
```

You will be prompted to enter a VIN:

```
Enter VIN number: 1C4JJXP66PW500000

[vpic] Fetching: https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/1C4JJXP66PW500000?format=json

[main] Vehicle  : Jeep Wrangler 4 XE
[main] Trim raw : '4XE'  →  normalized: '4 XE'
[main] Make     : Jeep

Starting Marketing Copy Generation...

Pipeline: 100%|████████████████████| 4/20 [00:08<00:00, node]

================================================================================
FINAL MARKETING COPY
================================================================================
The Jeep Wrangler 4 XE combines legendary Jeep capability...

================================================================================
QUALITY SCORE : 100/100
APPROVED      : True
ITERATIONS    : 1
================================================================================
```

---

### API Mode

**Start the development server:**
```bash
uvicorn api:app --reload
```

The server starts at `http://127.0.0.1:8000`.

**Interactive docs:** Open `http://127.0.0.1:8000/docs` in your browser for Swagger UI.

**Example `curl` request:**
```bash
curl -X POST http://127.0.0.1:8000/generate-marketing-copy \
  -H "Content-Type: application/json" \
  -d '{
    "vin": "1C4JJXP66PW500000",
    "max_iterations": 5
  }'
```

**Example Python request:**
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/generate-marketing-copy",
    json={"vin": "1C4JJXP66PW500000", "max_iterations": 3}
)
data = response.json()
print(data["marketing_copy"])
print(f"Score: {data['quality_score']}/100 | Approved: {data['approved']}")
```

---

## Example Output

**Input VIN:** `1C4JJXP66PW500000` (Jeep Wrangler 4XE)

**Generated marketing copy:**

> The Jeep Wrangler 4 XE combines legendary Jeep capability with advanced plug-in hybrid technology, delivering an exciting blend of efficiency, power, and off-road confidence. Powered by a turbocharged 2.0L inline 4-cylinder engine paired with an electric drive system, the Wrangler 4 XE produces an impressive 375 horsepower and features a 4WD drivetrain for outstanding traction and control across a variety of driving conditions.
>
> Designed with Jeep's iconic SUV styling, this four-door Wrangler 4 XE offers the versatility and rugged character drivers expect while adding the benefits of plug-in hybrid performance. Key features include keyless ignition, electronic stability control, traction control, anti-lock brakes, daytime running lights, a rear visibility system, tire pressure monitoring, and front and side airbags for added peace of mind.
>
> This vehicle is equipped with a hydraulic braking system and features standard safety technologies engineered to enhance driver confidence. Available driver-assistance technologies include blind spot monitoring, dynamic brake support, forward collision warning, collision intervention braking, and semi-automatic headlamp beam switching.
>
> The Wrangler 4 XE delivers the open-air freedom, adventure-ready capability, and innovative hybrid technology that make it a standout choice for drivers seeking both efficiency and authentic Jeep performance.

**Quality score:** 100/100 | **Approved:** true | **Iterations:** 1

---

## Workflow Rules & Constraints

The pipeline enforces strict content rules to ensure factual, professional, and legally safe output:

### Always Required
- ✅ Vehicle model name must appear in the copy
- ✅ Model name must have a space between any letter and digit (`4 XE`, not `4XE`)
- ✅ All claims must be backed by VPIC data
- ✅ Safety features must accurately reflect their VPIC status (Standard vs Optional)
- ✅ Exactly 4 paragraphs, flowing prose only (no bullet points, no bold, no headers)

### Always Prohibited
- ❌ Model year or any calendar year (`2023`, `2024`, etc.)
- ❌ Pricing of any kind (MSRP, base price, dealer price, dollar amounts)
- ❌ Plant location, manufacturing city, or country of assembly
- ❌ Features not present in the VPIC decode response (hallucinations)
- ❌ kW power values (HP only)
- ❌ `-Class` suffix added to model names

### Model Name Normalization Rules
| Raw VPIC | Normalized | In Copy |
|---|---|---|
| `4XE` | `4 XE` | Wrangler 4 XE |
| `GLC300` | `GLC 300` | GLC 300 |
| `M3` | `M 3` | M 3 |
| `X5` | `X 5` | X 5 |

### Feature Mention Rules
| VPIC Status | How to write in copy |
|---|---|
| `Standard` | "equipped with", "features", "includes" |
| `Optional` | "available", "optional" |
| Not present | Must not be mentioned |

---

## Deployment

### Azure Web App

The application is deployed as an **Azure App Service (Web App)** using Python on Linux.

**Production startup (`startup.sh`):**
```bash
gunicorn \
    -w 1 \
    -k uvicorn.workers.UvicornWorker \
    api:app
```

This runs the FastAPI app through Gunicorn with a single Uvicorn async worker, which is the recommended configuration for Azure App Service.

**Azure App Service configuration required:**
- Runtime: Python 3.12+
- Startup Command: `bash startup.sh` (or set via portal)
- Environment variables: Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, and `AZURE_OPENAI_API_KEY` in **Configuration → Application settings**

---

### CI/CD Pipeline

**File:** `.github/workflows/main_langgraph-project.yml`

The pipeline triggers on every push to the `main` branch (or manual dispatch via `workflow_dispatch`).

**Pipeline steps:**

```
push to main
     │
     ▼
[build job]
  ├── actions/checkout@v4
  ├── actions/setup-python@v5  (Python 3.14)
  ├── python -m venv antenv
  ├── pip install -r requirements.txt
  └── actions/upload-artifact@v4
        └── Uploads entire repo (excluding antenv/)

     │
     ▼
[deploy job]
  ├── actions/download-artifact@v4
  └── azure/webapps-deploy@v3
        ├── app-name: LangGraph-Project
        ├── slot-name: Production
        └── publish-profile: ${{ secrets.AZUREAPPSERVICE_PUBLISHPROFILE_... }}
```

**Required GitHub secret:**
- `AZUREAPPSERVICE_PUBLISHPROFILE_0834DD5A84804A95A58A827D02C6BB90` — the Azure Web App publish profile (download from the Azure portal under **Get publish profile**)

> **Note:** The Azure build engine (Oryx) is enabled via `SCM_DO_BUILD_DURING_DEPLOYMENT=true`, so `pip install` runs automatically during deployment. The `antenv/` directory is excluded from the artifact to reduce payload size.

---

## Dependencies

Full list from `requirements.txt`:

| Package | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server for local development |
| `gunicorn` | Production WSGI/ASGI process manager |
| `langgraph` | Multi-agent graph orchestration framework |
| `langchain` | LLM abstraction and chain utilities |
| `langchain-openai` | `AzureChatOpenAI` integration for LangChain |
| `langsmith` | LLM observability, tracing, and evaluation |
| `python-dotenv` | Loads `.env` files into environment variables |
| `requests` | HTTP client for NHTSA vPIC API calls |
| `tqdm` | Progress bar for CLI pipeline visualization |
| `pydantic` | Data validation and request/response models |

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### `AZURE_OPENAI_API_KEY` not found / authentication error
- Verify your `.env` file exists at the project root
- Confirm all three `AZURE_OPENAI_*` variables are set correctly
- Check that your Azure OpenAI deployment name matches exactly (case-sensitive)

### VPIC returns empty or incomplete data
- Verify the VIN is exactly 17 characters and contains no spaces
- Some VINs return partial data with a non-zero error code — the pipeline will warn but continue
- The NHTSA vPIC API may occasionally be slow; the HTTP timeout is set to 15 seconds

### Graph execution fails / `Graph execution failed` 500 error
- Check that `AZURE_OPENAI_ENDPOINT` ends with a `/` (e.g., `https://your-resource.openai.azure.com/`)
- Ensure your Azure OpenAI deployment is active and the model supports chat completions
- Review Azure OpenAI quota and rate limits for your subscription

### Copy is never approved (always hits `max_iterations`)
- Check the reviewer logs — if JSON parsing always fails, the copy will always fail all checks
- Review the judge output to identify which specific flag is repeatedly failing
- Consider increasing `max_iterations` for complex vehicles with many features

### `ModuleNotFoundError` on import
- Ensure you have activated the virtual environment before running
- Run `pip install -r requirements.txt` again
- Confirm you are using Python 3.12 or higher: `python --version`

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and ensure all existing behavior still works
4. Commit with a descriptive message: `git commit -m "Add: description of change"`
5. Push to your fork: `git push origin feature/your-feature-name`
6. Open a pull request against the `main` branch

**Areas open for contribution:**
- Additional VPIC field mappings (new safety technologies as NHTSA adds them)
- Support for DMS (Dealer Management System) data integration
- Alternative LLM backends (OpenAI direct, Anthropic, local models)
- Expanded reviewer checks (grammar, sentence length, readability scores)
- Unit tests for VIN parsing, vehicle building, and agent prompts
- Docker / containerization support

---

## License

This project is provided as-is for demonstration and educational purposes. See the repository owner for licensing details.
