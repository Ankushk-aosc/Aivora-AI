"""AI Agent Framework (continuous-engineering priority #17).

Each agent is a declared, permission-scoped configuration over the
capabilities that already exist and are tested - not a new execution
engine. An agent restricts which capability routes it may call and
which RBAC permission is required to invoke it; it does not grant a
capability that isn't already real (an agent listing 'vision' as a tool
would just fail the same way calling VISION_AI directly does - agents
don't fabricate capability).
"""

from dataclasses import dataclass, field

from ai_platform.orchestrator import AIOrchestrator
from ai_platform.registry import REGISTRY


@dataclass
class AgentDefinition:
    name: str
    instructions: str
    allowed_capabilities: list      # subset of REGISTRY routes this agent may use
    required_permission: str = "read"  # RBAC permission needed to invoke this agent
    memory: bool = False             # whether this agent keeps conversation history
    limits: dict = field(default_factory=lambda: {"max_calls_per_request": 1})


AGENTS = {
    "financial_analyst": AgentDefinition(
        name="Financial Analyst",
        instructions="Answer financial questions using the calculator for any "
        "arithmetic and the LLM only for explanation, never for numbers.",
        allowed_capabilities=["CALCULATOR", "FINANCIAL_LLM", "FORECASTING_AI"],
        required_permission="read",
    ),
    "research_analyst": AgentDefinition(
        name="Research Analyst",
        instructions="Search for and cite real sources; never present a claim "
        "without a retrievable citation.",
        allowed_capabilities=["RESEARCH_AI", "GENERAL_LLM"],
        required_permission="read",
    ),
    "document_analyst": AgentDefinition(
        name="Document Analyst",
        instructions="Answer questions about uploaded documents via RAG only; "
        "say 'Insufficient information available' rather than guessing.",
        allowed_capabilities=["RAG", "GENERAL_LLM"],
        required_permission="read",
    ),
    "data_analyst": AgentDefinition(
        name="Data Analyst",
        instructions="Answer questions about structured data via the database "
        "query templates only; never fabricate a query result.",
        allowed_capabilities=["DATABASE_AI", "GENERAL_LLM"],
        required_permission="read",
    ),
    "fraud_analyst": AgentDefinition(
        name="Fraud Analyst",
        instructions="Score transactions for fraud risk and flag high-risk "
        "cases for human approval; never auto-reject a transaction.",
        allowed_capabilities=["FRAUD_AI", "ANOMALY_AI"],
        required_permission="write",
    ),
    "enterprise_assistant": AgentDefinition(
        name="Enterprise Assistant",
        instructions="General-purpose assistant with access to the full "
        "capability set the orchestrator already exposes.",
        allowed_capabilities=list(REGISTRY.keys()),
        required_permission="read",
    ),
}


class AgentExecutionError(Exception):
    pass


class Agent:
    def __init__(self, agent_id: str, orchestrator: AIOrchestrator):
        if agent_id not in AGENTS:
            raise KeyError(f"Unknown agent '{agent_id}'. Known: {sorted(AGENTS)}")
        self.definition = AGENTS[agent_id]
        self.orchestrator = orchestrator

    def handle(self, query: str):
        """Runs the query through the shared orchestrator, then enforces
        this agent's capability allowlist on the OUTCOME - if the
        orchestrator routed to a capability this agent isn't allowed to
        use, the agent refuses to return that answer rather than
        silently allowing scope creep."""
        response = self.orchestrator.handle(query)
        if response.capability not in self.definition.allowed_capabilities:
            raise AgentExecutionError(
                f"'{self.definition.name}' is not permitted to use capability "
                f"'{response.capability}' (allowed: {self.definition.allowed_capabilities}). "
                f"The orchestrator routed here based on the query content; ask a "
                f"question within this agent's scope."
            )
        return response


def list_agents():
    return [
        {"id": agent_id, "name": d.name, "instructions": d.instructions,
         "allowed_capabilities": d.allowed_capabilities,
         "required_permission": d.required_permission}
        for agent_id, d in AGENTS.items()
    ]
