"""
Scalar automatic-differentiation engine (reverse mode).

Every arithmetic operation on Value objects creates a new Value that remembers
its parents and a closure with the local derivative rule. backward() sorts the
graph topologically and applies the chain rule from the output back to the leaves.
"""
import math


class Value:
    """One node of the computation graph.

    data      -- the scalar result of the forward pass
    grad      -- d(output)/d(this node), filled in by backward()
    _prev     -- parent nodes this value was computed from
    _op       -- name of the operation that produced it (for printing the graph)
    _backward -- local derivative rule: pushes self.grad into the parents' grads
    """

    __slots__ = ("data", "grad", "_prev", "_op", "_backward", "label")

    def __init__(self, data, _prev=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # ------------------------------------------------------------ core ops
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents are supported"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            # d(a^k)/da = k * a^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # subgradient 0 at x = 0, same convention as PyTorch
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------- extra ops for the classifier
    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    # ---------------------------------------- derived ops (reuse the core ones)
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self ** -1

    # ------------------------------------------------------------ backward
    def topo_order(self):
        """Nodes in topological order (every node after all of its parents).

        Iterative DFS, so deep graphs do not hit Python's recursion limit.
        """
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if id(p) not in visited:
                    stack.append((p, False))
        return order

    def backward(self):
        # Like PyTorch, leaf gradients accumulate across backward() calls;
        # call zero_grad on the parameters before every training step.
        order = self.topo_order()
        self.grad = 1.0
        for node in reversed(order):
            node._backward()


def vsum(values):
    """Sum of a list of Values (the .sum() of the scalar engine)."""
    total = Value(0.0)
    for v in values:
        total = total + v
    return total
