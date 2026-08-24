"""Knowledge Graph AI (spec Part 19).

Real graph storage via networkx (in-memory, no server). Entity/relation
extraction is rule-based (regex over a handful of common financial
relation verbs), NOT LLM-based - the proprietary model is not yet
reliable enough to extract structured relations, and a wrong extraction
silently corrupting a graph is worse than a smaller but honest one.
"""

import re

import networkx as nx

# A "proper noun phrase" is 1-6 consecutive tokens each starting with a
# capital letter (allowing an internal &/' like "P&G" or "O'Brien", but
# NOT a period - see below). This is what actually fixes entity-boundary
# extraction: it naturally stops at the first lowercase connector word
# ("a", "for", "with", "that"), which a permissive [\w -]+ character
# class does not - an earlier version used the permissive form and
# silently captured "Beta Logistics, a subsidiary that handles" as one
# "entity" instead of stopping at "Beta Logistics". Caught by testing
# path_between() against real multi-sentence text, not assumed correct
# from a single clean example.
#
# A second bug, also found by live multi-sentence testing: an earlier
# version allowed "." inside a token (for abbreviations like "Inc."),
# which let the match run straight through a sentence-ending period into
# the next sentence's capitalized word ("Beta Logistics. Beta Logistics"
# became one "entity"). Dropping "." from the token class stops capture
# at the period - the tradeoff is "Inc." becomes "Inc" (harmless), which
# is worth it to stop cross-sentence bleed (not harmless).
_ENTITY = r"((?:[A-Z][\w&'-]*\s?){1,6})"

_RELATION_PATTERNS = [
    ("OWNS", re.compile(_ENTITY + r"\s+(?:owns|acquired|holds a stake in)\s+" + _ENTITY)),
    ("SUBSIDIARY_OF", re.compile(_ENTITY + r"\s+is a subsidiary of\s+" + _ENTITY)),
    ("VENDOR_OF", re.compile(_ENTITY + r"\s+(?:paid|purchased from|is a vendor (?:of|for))\s+" + _ENTITY)),
    ("EMPLOYS", re.compile(_ENTITY + r"\s+employs\s+" + _ENTITY)),
    ("SUPPLIES", re.compile(_ENTITY + r"\s+supplies\s+" + _ENTITY)),
]


def _clean_entity(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().rstrip(".,")


def extract_relations(text: str) -> list:
    """Rule-based extraction: returns [(subject, relation, object), ...].
    Conservative by design - misses relations phrased unusually rather
    than guessing, since a graph edge implies a factual claim."""
    triples = []
    for relation, pattern in _RELATION_PATTERNS:
        for m in pattern.finditer(text):
            subj, obj = _clean_entity(m.group(1)), _clean_entity(m.group(2))
            if subj and obj and subj.lower() != obj.lower():
                triples.append((subj, relation, obj))
    return triples


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_text(self, text: str, source: str = None) -> list:
        triples = extract_relations(text)
        for subj, relation, obj in triples:
            self.graph.add_edge(subj, obj, relation=relation, source=source)
        return triples

    def add_triple(self, subject: str, relation: str, obj: str, source: str = None):
        self.graph.add_edge(subject, obj, relation=relation, source=source)

    def relationships(self, entity: str) -> list:
        """All edges touching `entity`, in either direction."""
        results = []
        if entity in self.graph:
            for _, target, data in self.graph.out_edges(entity, data=True):
                results.append({"subject": entity, "relation": data["relation"],
                                 "object": target, "source": data.get("source")})
            for source, _, data in self.graph.in_edges(entity, data=True):
                results.append({"subject": source, "relation": data["relation"],
                                 "object": entity, "source": data.get("source")})
        return results

    def path_between(self, a: str, b: str):
        """Shortest relationship path between two entities, undirected."""
        undirected = self.graph.to_undirected()
        if a not in undirected or b not in undirected:
            return None
        try:
            return nx.shortest_path(undirected, a, b)
        except nx.NetworkXNoPath:
            return None

    def stats(self) -> dict:
        return {"entities": self.graph.number_of_nodes(),
                "relationships": self.graph.number_of_edges()}

    def to_dict(self) -> dict:
        return {
            "entities": list(self.graph.nodes()),
            "relationships": [
                {"subject": u, "relation": d["relation"], "object": v, "source": d.get("source")}
                for u, v, d in self.graph.edges(data=True)
            ],
        }
