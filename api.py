from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph import graph
from main import fetch_vpic, build_vehicle

app = FastAPI(
    title="Vehicle Marketing Copy API",
    version="1.0.0"
)


class VINRequest(BaseModel):
    vin: str


class MarketingCopyResponse(BaseModel):
    vin: str
    vehicle_name: str
    marketing_copy: str
    quality_score: int
    approved: bool
    iterations: int


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "LangGraph Automotive Marketing Copy API"
    }


@app.post(
    "/generate-marketing-copy",
    response_model=MarketingCopyResponse,
    tags=["generate"],
    operation_id="marketing_copy",
)
def generate_marketing_copy(request: VINRequest):

    vin = request.vin.strip()

    if not vin:
        raise HTTPException(
            status_code=400,
            detail="VIN is required"
        )

    try:
        # Fetch VPIC data
        vpic_data = fetch_vpic(vin)

        if not vpic_data:
            raise HTTPException(
                status_code=404,
                detail="Unable to decode VIN"
            )

        # Build vehicle object
        vehicle = build_vehicle(vpic_data, vin)

        # Initial LangGraph state
        initial_state = {
            "vehicle": vehicle,
            "marketing_copy": "",
            "review_feedback": {},
            "quality_score": 0,
            "approved": False,
            "iteration": 0,
        }

        result = None

        # Execute graph
        for event in graph.stream(initial_state):
            node_name = next(iter(event))

            if node_name != "__end__":
                result = event[node_name]

        if not result:
            raise HTTPException(
                status_code=500,
                detail="Graph execution failed"
            )

        return {
            "vin": vin.upper(),
            "vehicle_name": vehicle["full_name"],
            "marketing_copy": result["marketing_copy"],
            "quality_score": result["quality_score"],
            "approved": result["approved"],
            "iterations": result["iteration"]
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )