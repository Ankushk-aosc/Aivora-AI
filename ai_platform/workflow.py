"""Workflow Engine (continuous-engineering priority #21).

A minimal, real step-chain executor over capabilities that already
exist and are tested - not a new execution substrate. Each step's
input is the previous step's output (or the workflow's initial input
for the first step); every step is logged via ai_platform.observability
so a workflow run has the same audit trail as any single capability
call. A step failing stops the workflow rather than silently
continuing with a partial/fabricated result.
"""

import datetime
from dataclasses import dataclass, field

from ai_platform.observability import track_request


@dataclass
class WorkflowStep:
    name: str
    fn: callable       # (context: dict) -> dict; reads/writes named keys in context
    requires_approval: bool = False


@dataclass
class WorkflowResult:
    workflow: str
    steps_completed: list
    context: dict
    status: str  # "completed" | "failed" | "awaiting_approval"
    error: str = None

    def to_dict(self):
        return {"workflow": self.workflow, "steps_completed": self.steps_completed,
                "status": self.status, "error": self.error,
                "context": {k: v for k, v in self.context.items() if k != "_internal"}}


class Workflow:
    def __init__(self, name: str, steps: list):
        self.name = name
        self.steps = steps

    def run(self, initial_context: dict) -> WorkflowResult:
        context = dict(initial_context)
        completed = []
        for step in self.steps:
            with track_request(f"WORKFLOW:{self.name}:{step.name}") as record:
                try:
                    output = step.fn(context)
                except Exception as e:
                    record.error = f"{type(e).__name__}: {e}"
                    return WorkflowResult(
                        workflow=self.name, steps_completed=completed, context=context,
                        status="failed", error=f"Step '{step.name}' failed: {e}",
                    )
                context.update(output)
                record.output_preview = str(output)[:120]

            completed.append(step.name)

            if step.requires_approval and context.get("_approval_pending"):
                return WorkflowResult(
                    workflow=self.name, steps_completed=completed, context=context,
                    status="awaiting_approval",
                )

        return WorkflowResult(
            workflow=self.name, steps_completed=completed, context=context,
            status="completed",
        )


# ----------------------------------------------------------------------
# Invoice review workflow (spec Part 29's example):
#   Document -> Extraction -> Fraud check -> Approval -> Audit
# ----------------------------------------------------------------------

def _step_extract_document(context: dict) -> dict:
    from rag.document_loader import load_document
    document = load_document(context["document_path"])
    text = document.text
    return {"document_text": text, "document_format": document.format}


def _step_parse_invoice_fields(context: dict) -> dict:
    """Real, deterministic field extraction (regex over the extracted
    text) - not an LLM guess, same 'deterministic core' pattern as the
    calculator. Only fields it can actually find are returned."""
    import re
    text = context["document_text"]
    amount_match = re.search(r"(?:amount|total)[:\s]*[₹$€]?\s*([\d,]+\.?\d*)", text, re.I)
    vendor_match = re.search(r"vendor[:\s]*([A-Za-z0-9 &.,'-]{2,60})", text, re.I)
    return {
        "invoice_amount": float(amount_match.group(1).replace(",", "")) if amount_match else None,
        "invoice_vendor": vendor_match.group(1).strip() if vendor_match else None,
    }


def _step_fraud_check(context: dict) -> dict:
    from ai_platform.fraud import load_model
    if context.get("invoice_amount") is None:
        return {"fraud_risk_score": None, "fraud_check_note": "No amount extracted - cannot score"}
    model = load_model()
    txn = {
        "amt": context["invoice_amount"], "category": context.get("invoice_category", "misc_net"),
        "trans_date_trans_time": context.get("invoice_timestamp")
        or datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "city_pop": context.get("invoice_city_pop", 100000),
    }
    score = model.score([txn])[0]
    return {"fraud_risk_score": score}


def _step_gate_approval(context: dict) -> dict:
    from ai_platform.approval import request_approval, requires_approval
    score = context.get("fraud_risk_score")
    if score is not None and requires_approval("FRAUD_AI", risk_score=score):
        req = request_approval(
            "FRAUD_AI", f"Invoice from '{context.get('invoice_vendor')}' for "
            f"{context.get('invoice_amount')} (risk={score})",
            evidence={"vendor": context.get("invoice_vendor"),
                      "amount": context.get("invoice_amount"), "risk_score": score},
            confidence="model_score", requested_by="workflow:invoice_review",
        )
        return {"_approval_pending": True, "approval_request_id": req["id"]}
    return {"_approval_pending": False}


INVOICE_REVIEW_WORKFLOW = Workflow("invoice_review", [
    WorkflowStep("extract_document", _step_extract_document),
    WorkflowStep("parse_invoice_fields", _step_parse_invoice_fields),
    WorkflowStep("fraud_check", _step_fraud_check),
    WorkflowStep("gate_approval", _step_gate_approval, requires_approval=True),
])


WORKFLOWS = {"invoice_review": INVOICE_REVIEW_WORKFLOW}


def run_workflow(name: str, initial_context: dict) -> WorkflowResult:
    if name not in WORKFLOWS:
        raise KeyError(f"Unknown workflow '{name}'. Known: {sorted(WORKFLOWS)}")
    return WORKFLOWS[name].run(initial_context)
