import math


def trace(root):
    """
    Trace all the nodes and edges in the graph starting at root

    :param root: root node of the compute graph
    :return:
    """
    nodes, edges = set(), set()

    def build(node):
        if node not in nodes:
            nodes.add(node)
            for child in node.children:
                edges.add((child, node))
                build(child)

    build(root)

    return nodes, edges


class Value:

    def __init__(self, data, children=(), op='', label=''):
        self.data = data
        self.grad = 0.0
        self.children = set(children)
        self.op = op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def tanh(self):
        x = self.data
        tanh_x = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(tanh_x, (self, ), 'tanh')
        return out

    def plot(self):
        """
        Plot the compute graph.

        :return: graph object
        """
        from graphviz import Digraph

        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
        nodes, edges = trace(self)

        for node in nodes:
            uid = str(id(node))

            # for values in the compute graph, draw a rectangular ('record') node
            dot.node(name=uid,
                     label=f"{{ {node.label} | data: {node.data:.4f} | grad: {node.grad:.4f} }}", shape='record')

            if node.op:
                # for operations in the compute graph, additionally draw an elliptical node
                dot.node(name=uid + node.op, label=node.op)
                # draw an edge from the operation (elliptical node) to the value (rectangular node)
                dot.edge(uid + node.op, uid)

        for node_from, node_to in edges:
            # draw edges in the compute graph
            dot.edge(str(id(node_from)), str(id(node_to)) + node_to.op)

        return dot
