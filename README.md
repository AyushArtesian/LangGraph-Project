# LangGraph Automotive Marketing Copy Pipeline

A LangGraph-based application that turns a vehicle VIN into dealership-style marketing copy by combining NHTSA vPIC vehicle data with a multi-agent review loop powered by Azure OpenAI.

## Highlights

- Decodes VINs with the NHTSA vPIC API
- Normalizes vehicle and safety data into a structured state object
- Uses a four-node LangGraph workflow for generation, review, refinement, and approval
- Exposes a FastAPI service for programmatic use
- Includes Azure Web App deployment assets for production hosting

## How It Works

1. A VIN is submitted through the CLI or API.
2. The app calls the NHTSA vPIC decoder to fetch vehicle attributes.
3. `build_vehicle()` shapes the decoded data into a reusable vehicle object.
4. LangGraph runs the marketing workflow:
   - `generator` creates the first draft
   - `reviewer` checks factual consistency against VPIC data
   - `judge` scores the result and decides whether it passes
   - `refiner` rewrites the copy when another iteration is needed
5. The workflow stops when the copy is approved or the iteration limit is reached.

## Architecture

```text
VIN input
   |
   v
NHTSA vPIC API
   |
   v
fetch_vpic() -> build_vehicle() -> MarketingState
                                |
                                v
                       START -> generator -> reviewer -> judge
                                                    |         |
                                                    |         v
                                                    |        END
                                                    v
                                                  refiner
                                                    |
                                                    +-------> reviewer
```

## Project Structure

```text
.
├── agents/
│   ├── generator.py
│   ├── judge.py
│   ├── refiner.py
│   └── reviewer.py
├── .github/workflows/
│   └── main_langgraph-project.yml
├── api.py
├── graph.py
├── llm.py
├── main.py
├── requirements.txt
├── startup.sh
└── state.py
```

## Key Components

### Core application files

- `main.py` - CLI entry point, VIN decoding, vehicle normalization, and graph execution
- `api.py` - FastAPI app with health and marketing-copy generation endpoints
- `graph.py` - LangGraph node registration and routing logic
- `state.py` - shared `MarketingState` schema
- `llm.py` - Azure OpenAI client configuration

### Agent nodes

- `agents/generator.py` - produces the initial four-paragraph vehicle description
- `agents/reviewer.py` - checks spacing, duplication, pricing, dates, and hallucinated claims
- `agents/refiner.py` - rewrites copy from reviewer feedback
- `agents/judge.py` - applies deterministic scoring and approval rules

### Deployment files

- `startup.sh` - starts the FastAPI app with Gunicorn and the Uvicorn worker
- `.github/workflows/main_langgraph-project.yml` - builds the app and deploys it to Azure Web App on pushes to `main`

## Requirements

- Python 3.12+
- Azure OpenAI access
- Internet access for the NHTSA vPIC API

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root with:

```env
AZURE_OPENAI_ENDPOINT=your-azure-openai-endpoint
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_KEY=your-api-key
```

## Running Locally

### API

Start the FastAPI app locally:

```bash
uvicorn api:app --reload
```

Available endpoints:

- `GET /` - health check
- `POST /generate-marketing-copy` - generate marketing copy from a VIN
- `GET /docs` - interactive Swagger UI

Example request:

```bash
curl -X POST http://127.0.0.1:8000/generate-marketing-copy \
  -H "Content-Type: application/json" \
  -d '{
    "vin": "1C4JJXP66PW500000",
    "max_iterations": 5
  }'
```

### CLI

Run the command-line entry point:

```bash
python main.py
```

You will be prompted to enter a VIN, and the pipeline will print the generated copy, quality score, approval status, and iteration count.

## API Response

A successful generation request returns:

```json
{
  "vin": "1C4JJXP66PW500000",
  "vehicle_name": "Jeep Wrangler 4 XE",
  "marketing_copy": "...",
  "quality_score": 100,
  "approved": true,
  "iterations": 1
}
```

## Workflow Rules

The system is designed to keep output fact-based and consistent:

- model names are normalized to add spaces between letters and numbers
- pricing, date, and plant-location references are disallowed
- only VPIC-confirmed features should be mentioned
- approval requires the score threshold to be met without critical violations

## Deployment

The repository includes an Azure deployment workflow that:

1. installs Python dependencies
2. uploads the application artifact
3. deploys the app to the `LangGraph-Project` Azure Web App

For production startup, `startup.sh` runs:

```bash
gunicorn -w 1 -k uvicorn.workers.UvicornWorker api:app
```

## Dependencies

Main packages used in this project:

- FastAPI
- Uvicorn
- Gunicorn
- LangGraph
- LangChain
- LangChain OpenAI
- LangSmith
- python-dotenv
- requests
- tqdm
- pydantic

## Notes

- The quality of the output depends on the completeness of the vPIC decode response.
- Azure OpenAI credentials are required before running the workflow successfully.
- The FastAPI service is the best option for integrating this pipeline into other systems.
