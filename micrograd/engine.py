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
            for child in node._children:
                edges.add((child, node))
                build(child)
    build(root)

    return nodes, edges


class Value:

    def __init__(self, data, children=(), op='', label=''):
        self.data = data
        self.grad = 0.0
        self.label = label

        # internal states for compute graph construction
        self._children = set(children)
        self._op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):  # called when the left operand (other) does not support addition e.g. 3 + Value(2)
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):  # called when the left operand (other) does not support subtraction e.g. 3 - Value(2)
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):  # called when the left operand (other) does not support multiplication e.g. 3 * Value(2)
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):  # called when the left operand (other) does not support division e.g. 3 / Value(2)
        return other * self ** -1

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += math.exp(self.data) * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), f'**{other.data}')

        def _backward():
            self.grad += other.data * self.data ** (other.data - 1) * out.grad
            other.grad += math.log(self.data) * self.data ** other.data * out.grad
        out._backward = _backward

        return out

    def __rpow__(self, other):  # called when the left operand (other) does not support power e.g. 3 ** Value(2)
        other = other if isinstance(other, Value) else Value(other)
        return other ** self

    def tanh(self):
        x = self.data
        tanh_x = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(tanh_x, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - tanh_x**2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        x = self.data
        relu_x = max(0, x)
        out = Value(relu_x, (self, ), 'relu')

        def _backward():
            self.grad += (x > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.data
        sigmoid_x = 1 / (1 + math.exp(-x))
        out = Value(sigmoid_x, (self, ), 'sigmoid')

        def _backward():
            self.grad += sigmoid_x * (1 - sigmoid_x) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # topological order of all nodes in the graph
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:  # process parent only after all its children have been processed
                    build_topo(child)
                topo.append(node)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0  # initialize gradient to 1.0 for the last node
        for node in reversed(topo):
            node._backward()

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

            if node._op:
                # for operations in the compute graph, additionally draw an elliptical node
                dot.node(name=uid + node._op, label=node._op)
                # draw an edge from the operation (elliptical node) to the value (rectangular node)
                dot.edge(uid + node._op, uid)

        for node_from, node_to in edges:
            # draw edges in the compute graph
            dot.edge(str(id(node_from)), str(id(node_to)) + node_to._op)

        return dot
