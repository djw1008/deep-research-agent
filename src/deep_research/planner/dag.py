"""DAG (有向无环图) 数据结构与拓扑排序。"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterator


class DAGCycleError(Exception):
    """DAG 中存在环时抛出。"""


class DAG:
    """有向无环图：节点为 task_id，边表示依赖关系 (u -> v 表示 v 依赖 u)。"""

    def __init__(self) -> None:
        self._nodes: set[str] = set()
        self._edges: dict[str, list[str]] = defaultdict(list)
        self._in_degree: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # 增删查
    # ------------------------------------------------------------------
    def add_node(self, node_id: str) -> None:
        self._nodes.add(node_id)

    def add_edge(self, from_node: str, to_node: str) -> None:
        if from_node == to_node:
            raise DAGCycleError(f"自环不允许: {from_node}")
        self._nodes.add(from_node)
        self._nodes.add(to_node)
        self._edges[from_node].append(to_node)
        self._in_degree[to_node] += 1
        self._in_degree.setdefault(from_node, 0)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_dependencies(self, node_id: str) -> list[str]:
        deps: list[str] = []
        for src, dsts in self._edges.items():
            if node_id in dsts:
                deps.append(src)
        return deps

    def get_successors(self, node_id: str) -> list[str]:
        return list(self._edges.get(node_id, []))

    def __iter__(self) -> Iterator[str]:
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    # ------------------------------------------------------------------
    # 拓扑排序
    # ------------------------------------------------------------------
    def topological_sort(self) -> list[str]:
        """Kahn 算法拓扑排序，返回节点全序列表。"""
        in_deg = dict(self._in_degree)
        for n in self._nodes:
            in_deg.setdefault(n, 0)

        queue: deque[str] = deque([n for n in self._nodes if in_deg.get(n, 0) == 0])
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for succ in self._edges.get(node, []):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)

        if len(result) != len(self._nodes):
            remaining = self._nodes - set(result)
            raise DAGCycleError(
                f"DAG 中存在环，剩余节点: {sorted(remaining)}"
            )
        return result

    def get_parallel_groups(self) -> list[list[str]]:
        """按执行层分组，同层节点可并发执行。"""
        in_deg = dict(self._in_degree)
        for n in self._nodes:
            in_deg.setdefault(n, 0)

        groups: list[list[str]] = []
        current: list[str] = [n for n in self._nodes if in_deg.get(n, 0) == 0]
        current.sort()
        visited: set[str] = set()

        while current:
            groups.append(current)
            visited.update(current)
            next_layer: list[str] = []
            for node in current:
                for succ in self._edges.get(node, []):
                    in_deg[succ] -= 1
                    if in_deg[succ] == 0 and succ not in visited:
                        next_layer.append(succ)
            next_layer.sort()
            current = next_layer

        if sum(len(g) for g in groups) != len(self._nodes):
            raise DAGCycleError("DAG 中存在环，无法计算并行分组")
        return groups

    def to_dict(self) -> dict:
        return {
            "nodes": sorted(self._nodes),
            "edges": {k: sorted(v) for k, v in sorted(self._edges.items())},
        }

    def to_ascii(self) -> str:
        """将 DAG 打印为 ASCII 层级图。"""
        groups = self.get_parallel_groups()
        lines: list[str] = []
        lines.append("=" * 50)
        lines.append(f"DAG: {len(self._nodes)} nodes, {sum(len(v) for v in self._edges.values())} edges")
        lines.append("=" * 50)
        for i, group in enumerate(groups, 1):
            lines.append(f"  Layer {i}: {', '.join(group)}")
            # 打印每个节点的出边
            for node in group:
                succs = self._edges.get(node, [])
                if succs:
                    lines.append(f"    {node} -> {', '.join(sorted(succs))}")
        lines.append("=" * 50)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<DAG nodes={len(self._nodes)} edges={sum(len(v) for v in self._edges.values())}>"
