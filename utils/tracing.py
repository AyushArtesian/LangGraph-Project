from datetime import datetime

def add_trace(
        state,
        agent, 
        status = "success",
        details = None,
):
    state["trace"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "agent": agent,
        "iteration": state["iteration"],
        "run_id": state["run_id"],
        "status": status,
        "details": details or {},
    })