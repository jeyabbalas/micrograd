import random

from micrograd.engine import Value


class Module:

    def zero_grad(self):
        for w in self.parameters():
            w.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, in_features, bias=True, label_prefix=''):
        self.label_prefix = label_prefix if len(label_prefix) == 0 else f",{label_prefix}"
        self.w = [Value(random.uniform(-1, 1), label=f"w{i + 1}{self.label_prefix}")
                  for i, _ in enumerate(range(in_features))]
        self.b = Value(random.uniform(-1, 1), label=f"w0{self.label_prefix}") if bias else None

    def __call__(self, x):
        a = self.w[0] * x[0] if self.b is None else self.b + self.w[0] * x[0]
        for i in range(1, len(self.w)):
            a += self.w[i] * x[i]
        a.label = f"a{self.label_prefix[1:] if self.label_prefix.startswith(',') else self.label_prefix}"
        return a

    def parameters(self):
        return self.w + ([self.b] if self.b is not None else [])


class Linear(Module):

    def __init__(self, in_features, out_features, bias=True, label_prefix=''):
        self.neurons = [Neuron(in_features, bias=bias, label_prefix=f"{i + 1}_{label_prefix}")
                        for i, _ in enumerate(range(out_features))]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):

    def __init__(self, in_features, out_features_list, bias=True):
        num_neurons = [in_features] + out_features_list
        self.layers = [Linear(num_neurons[i], num_neurons[i + 1], bias=bias, label_prefix=f"{i + 1}")
                       for i, _ in enumerate(range(len(num_neurons) - 1))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
