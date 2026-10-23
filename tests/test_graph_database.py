from storage.graph_database import GraphDatabase


def test_graph_database_stores_nodes_and_edges():
    graph = GraphDatabase()
    graph.upsert_node("A", {})
    graph.upsert_node("B", {})
    graph.add_edge("A", "B", "causes")
    assert len(graph.edges) == 1
