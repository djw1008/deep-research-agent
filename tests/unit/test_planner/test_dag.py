"""Tests for DAG."""

import pytest

from deep_research.planner import DAG, DAGCycleError


def test_add_node_and_edge():
    dag = DAG()
    dag.add_node("A")
    dag.add_node("B")
    dag.add_edge("A", "B")

    assert "A" in dag
    assert "B" in dag
    assert len(dag) == 2
    assert dag.get_successors("A") == ["B"]
    assert dag.get_dependencies("B") == ["A"]


def test_topological_sort():
    dag = DAG()
    dag.add_edge("A", "C")
    dag.add_edge("B", "C")
    dag.add_edge("C", "D")

    order = dag.topological_sort()
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("C")
    assert order.index("C") < order.index("D")


def test_topological_sort_cycle_raises():
    dag = DAG()
    dag.add_edge("A", "B")
    dag.add_edge("B", "A")

    with pytest.raises(DAGCycleError):
        dag.topological_sort()


def test_self_loop_raises():
    dag = DAG()
    with pytest.raises(DAGCycleError):
        dag.add_edge("A", "A")


def test_parallel_groups():
    dag = DAG()
    dag.add_edge("A", "C")
    dag.add_edge("B", "C")
    dag.add_edge("C", "D")

    groups = dag.get_parallel_groups()
    assert groups[0] == ["A", "B"]
    assert groups[1] == ["C"]
    assert groups[2] == ["D"]


def test_to_dict():
    dag = DAG()
    dag.add_edge("A", "B")
    d = dag.to_dict()
    assert d["nodes"] == ["A", "B"]
    assert d["edges"] == {"A": ["B"]}
