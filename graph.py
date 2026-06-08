# graph.py

from langgraph.graph import StateGraph, START, END

from state import MarketingState
from agents.generator import generator_node
from agents.reviewer  import reviewer_node
from agents.refiner   import refiner_node
from agents.judge     import judge_node



builder = StateGraph(MarketingState)

builder.add_node("generator", generator_node)
builder.add_node("reviewer",  reviewer_node)
builder.add_node("refiner",   refiner_node)
builder.add_node("judge",     judge_node)

# Initial flow
builder.add_edge(START,       "generator")
builder.add_edge("generator", "reviewer")
builder.add_edge("reviewer",  "judge")

# Retry flow: judge → refiner → reviewer → judge
builder.add_edge("refiner",   "reviewer")


def route_after_judge(state) -> str:
    if state["approved"]:
        print(
            f"\n✅ Approved at iteration {state['iteration']} "
            f"with score {state['quality_score']}/100"
        )
        return END

    if state["iteration"] >= state["max_iterations"]:
        print(
            f"\n❌ Rejected after {state['iteration']} iterations. "
            f"Final score: {state['quality_score']}/100"
        )

        
        return END

    print(f"\nRetrying... iteration {state['iteration']}")
    return "refiner"


builder.add_conditional_edges("judge", route_after_judge)

graph = builder.compile()
