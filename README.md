# LangGraph Marketing Copy Pipeline

A LangGraph-based multi-agent workflow that generates, critiques, refines, and scores automotive marketing copy from structured vehicle JSON data using Azure OpenAI.

## Overview

This project runs a four-node iterative pipeline:

1. **generator**: Creates dealership-style marketing copy from car specs.
2. **reviewer**: Detects missing info, factual issues, writing issues, and improvement points.
3. **refiner**: Rewrites the copy using targeted feedback.
4. **judge**: Runs a 3-stage evaluation (coverage checklist, factual check, quality score) and decides approve/retry.

The graph loops until:
- quality score reaches threshold (`>= 85`), or
- max iterations are reached (`5`).

## Project Structure

```text
.
├── agents/
│   ├── generator.py   # Initial copy generation
│   ├── reviewer.py    # Structured review feedback
│   ├── refiner.py     # Feedback-driven rewrite
│   └── judge.py       # Final scoring + routing decision
├── graph.py           # LangGraph state machine definition
├── llm.py             # AzureChatOpenAI client setup
├── main.py            # Entry point with sample vehicle JSON + progress bar
├── state.py           # TypedDict state contract
└── requirements.txt   # Python dependencies
```

## How It Works

### State (`MarketingState`)

Defined in `state.py`:

- `car_json: dict` — source vehicle data
- `marketing_copy: str` — generated/refined copy
- `review_feedback: str` — structured evaluator feedback
- `quality_score: int` — computed score out of 100
- `approved: bool` — approval signal
- `iteration: int` — loop counter

### Graph Flow

Defined in `graph.py`:

`START -> generator -> reviewer -> refiner -> judge -> (END or reviewer)`

Routing logic:
- End if approved
- End if `iteration >= 5`
- Otherwise retry through `reviewer`

## Prerequisites

- Python 3.10+
- Azure OpenAI deployment

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
AZURE_OPENAI_API_KEY=<your-api-key>
```

The model client is initialized in `llm.py` with:
- `api_version="2024-12-01-preview"`
- `temperature=0.3`

## Running the Pipeline

```bash
python main.py
```

`main.py` currently includes a built-in sample Toyota Fortuner JSON payload and streams graph execution with a tqdm progress bar.

## Output

At completion, the script prints:
- Final marketing copy
- Final quality score
- Approval status
- Iteration count

## Scoring Model (Judge)

`agents/judge.py` uses a weighted 100-point score:

- **Coverage**: 35 points (presence checklist of required details)
- **Factual Accuracy**: 30 points (penalties for spec mismatches)
- **Marketing Quality**: 35 points (headline, opening, language, value proposition, CTA)

Approval threshold: `85`.

## Notes

- Reviewer and judge both parse LLM output as JSON and include fallback handling for malformed responses.
- Prompts are strict about using only input JSON specs to reduce hallucinations.
- The sample run is deterministic only to the extent allowed by model behavior and temperature.

## Customization

- Replace `sample_car` in `main.py` with your own vehicle JSON.
- Adjust loop settings in `main.py`:
  - `MAX_ITERATIONS`
  - Progress bar step calculation
- Tune prompts in `agents/` for different tone, structure, or industries.
- Change `APPROVAL_THRESHOLD` in `agents/judge.py` as needed.

## Dependencies

From `requirements.txt`:

- `langgraph`
- `langchain`
- `langchain-openai`
- `python-dotenv`
- `pydantic`
