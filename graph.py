from langgraph.graph import StateGraph
from langgraph.graph import START
from langgraph.graph import END

from state import MarketingState

from agents.generator import generator_node
from agents.reviewer import reviewer_node
from agents.refiner import refiner_node
from agents.judge import judge_node

builder = StateGraph(MarketingState)

builder.add_node("generator", generator_node)
builder.add_node("reviewer", reviewer_node)
builder.add_node("refiner", refiner_node)
builder.add_node("judge", judge_node)

# Initial flow
builder.add_edge(START, "generator")
builder.add_edge("generator", "reviewer")
builder.add_edge("reviewer", "judge")

# Retry flow
builder.add_edge("refiner", "reviewer")


def route_after_judge(state):

    if state["approved"]:
        print(
            f"\nApproved with score {state['quality_score']}"
        )
        return END

    if state["iteration"] >= 5:
        print(
            f"\nRejected after {state['iteration']} iterations"
        )
        return END

    print(
        f"\nRetrying... iteration {state['iteration']}"
    )

    return "refiner"


builder.add_conditional_edges(
    "judge",
    route_after_judge
)

graph = builder.compile()