import pytest

from .basic import Node, Edge


class TestNode(object):
    @pytest.mark.parametrize("state,edges", [
        ("STATE A", {0: "EDGE 1", 1: "EDGE 2"}),
        ("STATE B", {0: "EDGE 1", 1: "EDGE 2", 2: "EDGE 3"}),
        (("STATE A", "AND B"), {'0': "EDGE 1", '1': "EDGE 2"})
    ])
    def test_getters(self, state, edges):
        node = Node(state)
        node.expand(edges)

        assert node.state == state
        assert node.edges == edges

    @pytest.mark.parametrize("edges,c,popular_key,result_key", [
        ({0: Edge(3), 1: Edge(1)}, 1., 0, 0),
        ({0: Edge(3), 1: Edge(1)}, 2., 0, 1),
        ({0: Edge(3), 1: Edge(1), 2: Edge(2)}, 1., 0, 2),
        ({0: Edge(3), 1: Edge(1), 2: Edge(2)}, 1., 1, 0),
        ({}, 0, 0, 0)
    ])
    def test_select_edge(self, edges, c, popular_key, result_key):
        node = Node("S0")
        node.expand(edges)

        if len(edges) == 0:
            assert node.select_edge(c) == None
        else:
            node.edges[popular_key]._num_visits += 10
            assert node.select_edge(c) == (result_key, node.edges[result_key])


class TestEdge(object):
    @pytest.mark.parametrize("next_node,edges,qvalue", [
        (Node("STATE A"), {0: Edge(3)}, 1),
        (Node("STATE B"), {0: Edge(3), 1: Edge(1)}, 2),
        (Node(("STATE A", "AND B")), {}, .7)
    ])
    def test_gettes(self, next_node, edges, qvalue):
        next_node.expand(edges)
        edge = Edge(qvalue, next_node)

        assert edge.next_node == next_node
        assert edge.qvalue == qvalue
        assert edge.num_visits == 0

    @pytest.mark.parametrize("current_qvalue,init_visits,return_t", [
        (1, 1, 1),
        (1, 1, 2),
        (2, 2, 3),
        (2, 3, 7),
        (2, 2, 2)
    ])
    def test_update(self, current_qvalue, init_visits, return_t):
        edge = Edge(current_qvalue)
        edge._num_visits = init_visits
        edge.update(return_t)

        # NOTE: Current Q-value have to be weighted by visits number because
        # it's average over Q-value after this number of visits
        assert edge.qvalue == (current_qvalue * init_visits + return_t) / (init_visits + 1)
