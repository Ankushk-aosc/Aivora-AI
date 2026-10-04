"""Structured outputs for the tool pipeline, and the validation that enforces them.

EXPERIMENT_RULES_v2 section 1a, owner-set and binding: **100% of extracted values
must carry a span verified present in the context.** A value whose span cannot be
located is rejected and becomes an abstention. That rule lives here, in one
function, so no caller can route around it.

The model is asked for JSON. Models produce almost-JSON, so parsing is tolerant
of fenced code blocks, trailing commas and leading prose - but tolerant parsing
never invents a field, and a response that cannot be parsed is an abstention, not
a guess.
"""

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Optional

# What the model is asked to return for an extraction.
EXTRACTION_SCHEMA = {
    "field": "the name of the field you were asked for, copied from the context",
    "value": "the value exactly as it appears in the context, or null",
    "source_span": "the complete line or phrase from the context containing the "
                   "value, copied character for character",
    "found": "true if the value is present in the context, false otherwise",
}

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


@dataclass
class ExtractionResult:
    found: bool
    value: Optional[str] = None
    field_name: Optional[str] = None
    source_span: Optional[str] = None
    span_validated: bool = False
    value_in_span: bool = False
    abstained: bool = False
    reason: Optional[str] = None
    raw_model_output: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class PipelineAnswer:
    answer: str
    component: str
    abstained: bool = False
    value: Optional[float] = None
    operation: Optional[str] = None
    operands: dict = field(default_factory=dict)
    source_spans: list = field(default_factory=list)
    reason: Optional[str] = None
    detail: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def normalise(text):
    """Whitespace- and unicode-insensitive comparison form.

    Models reflow whitespace and swap hyphens and quotes when copying a span.
    Treating that as a validation failure would reject correct extractions, so
    comparison is done on a normalised form while the ORIGINAL span is what gets
    reported.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = (text.replace("–", "-").replace("—", "-")
            .replace("‘", "'").replace("’", "'")
            .replace("“", '"').replace("”", '"'))
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_json_object(text):
    """The first JSON object in a model response, or None.

    Never repairs by inventing: it strips fences, takes the first balanced
    object, and removes trailing commas. Anything else is a parse failure.
    """
    if not text:
        return None
    candidates = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        match = _OBJECT.search(candidate)
        if match:
            for attempt in (match.group(0),
                            _TRAILING_COMMA.sub(r"\1", match.group(0))):
                try:
                    parsed = json.loads(attempt)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    pass
    return None


def validate_extraction(payload, context, raw_output=None):
    """Turn a parsed model response into a VALIDATED extraction, or an abstention.

    Three things must hold before a value is allowed through:
      1. the model said found = true
      2. source_span appears in the context (normalised comparison)
      3. value appears inside that span

    Failing any of them produces an abstention with the reason recorded. This is
    the binding 100% span rule; there is no path to an answer that skips it.
    """
    if payload is None:
        return ExtractionResult(found=False, abstained=True,
                                reason="model output was not parseable JSON",
                                raw_model_output=raw_output)

    found = payload.get("found")
    if isinstance(found, str):
        found = found.strip().lower() in ("true", "yes", "1")
    value = payload.get("value")
    span = payload.get("source_span")
    field_name = payload.get("field")

    if not found or value in (None, "", "null"):
        return ExtractionResult(found=False, abstained=True,
                                field_name=field_name,
                                reason="model reported the value is not present",
                                raw_model_output=raw_output)

    if not span:
        return ExtractionResult(found=True, value=str(value), field_name=field_name,
                                abstained=True,
                                reason="no source span supplied, so the value "
                                       "cannot be traced to the context",
                                raw_model_output=raw_output)

    normalised_context = normalise(context)
    span_ok = normalise(span) in normalised_context
    value_ok = normalise(value) in normalise(span)

    if not span_ok:
        return ExtractionResult(found=True, value=str(value), field_name=field_name,
                                source_span=str(span), span_validated=False,
                                abstained=True,
                                reason="source span does not appear in the context",
                                raw_model_output=raw_output)
    if not value_ok:
        return ExtractionResult(found=True, value=str(value), field_name=field_name,
                                source_span=str(span), span_validated=True,
                                value_in_span=False, abstained=True,
                                reason="value does not appear within its own "
                                       "source span",
                                raw_model_output=raw_output)

    return ExtractionResult(found=True, value=str(value), field_name=field_name,
                            source_span=str(span), span_validated=True,
                            value_in_span=True, abstained=False,
                            raw_model_output=raw_output)


INSUFFICIENT = ("Insufficient information: the requested value is not present in "
                "the context provided.")


def abstention(reason, component="validator"):
    return PipelineAnswer(answer=INSUFFICIENT, component=component, abstained=True,
                          reason=reason)
