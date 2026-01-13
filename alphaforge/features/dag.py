from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class LineageGraph:
    nodes: Dict[str, dict] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)

    def add(self, node_id: str, meta: dict) -> None:
        self.nodes[node_id] = meta

    def link(self, parent_id: str, child_id: str) -> None:
        self.edges.append((parent_id, child_id))
