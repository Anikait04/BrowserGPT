# graph.py — pure LangGraph wiring (extracted from agent.py).
#
# build_graph() has no I/O: no env, no DB, no PNG. Compilation + checkpointer
# stay in agent.get_app() / runner. Unit-testable without aiosqlite.

from __future__ import annotations

from langgraph.graph import StateGraph

from src.workflow.state import AgentState
from src.workflow.constants import (
    NODE_DELEGATION,
    NODE_EXTRACT,
    NODE_NAVIGATION,
    NODE_PLANNER,
    NODE_VERIFY,
    NODE_WAIT_FOR_USER,
)
from src.workflow.nodes.delegation import delegation_node
from src.workflow.nodes.extraction import extract_information_node
from src.workflow.nodes.human_gate import wait_for_user_node
from src.workflow.nodes.navigation import navigation_node
from src.workflow.nodes.planner import planner_node
from src.workflow.nodes.verify import verify_node
from src.workflow.routing import (
    route_from_delegation,
    route_from_verification,
    route_from_wait_for_user,
)


def build_graph() -> StateGraph:
    """Assemble the persistent-loop delegation graph (uncompiled)."""
    from langgraph.graph import END

    graph = StateGraph(AgentState)

    graph.add_node(NODE_PLANNER, planner_node)
    graph.add_node(NODE_DELEGATION, delegation_node)
    graph.add_node(NODE_NAVIGATION, navigation_node)
    graph.add_node(NODE_VERIFY, verify_node)
    graph.add_node(NODE_EXTRACT, extract_information_node)
    graph.add_node(NODE_WAIT_FOR_USER, wait_for_user_node)

    graph.set_entry_point(NODE_DELEGATION)
    graph.add_edge(NODE_PLANNER, NODE_DELEGATION)
    # Delegation is the single router: every run starts here, and based on
    # context it sends work to planner / navigation / extract / wait_for_user.
    # Persistent loop: delegation NEVER routes to END directly.
    graph.add_conditional_edges(
        NODE_DELEGATION,
        route_from_delegation,
        {
            NODE_PLANNER: NODE_PLANNER,
            NODE_NAVIGATION: NODE_NAVIGATION,
            NODE_EXTRACT: NODE_EXTRACT,
            NODE_WAIT_FOR_USER: NODE_WAIT_FOR_USER,
        },
    )

    graph.add_edge(NODE_NAVIGATION, NODE_VERIFY)
    graph.add_conditional_edges(
        NODE_VERIFY,
        route_from_verification,
        {
            NODE_NAVIGATION: NODE_NAVIGATION,
            NODE_DELEGATION: NODE_DELEGATION,
        },
    )
    graph.add_edge(NODE_EXTRACT, NODE_DELEGATION)
    # wait_for_user is the ONLY gateway to END (explicit user exit).
    graph.add_conditional_edges(
        NODE_WAIT_FOR_USER,
        route_from_wait_for_user,
        {
            NODE_DELEGATION: NODE_DELEGATION,
            END: END,
        },
    )
    return graph
