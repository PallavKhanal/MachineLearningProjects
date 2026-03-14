import math
import random


class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        # the actual number this value holds.
        self.data = data

        # how much does changing this value change the final loss?
        # we start with 0 because we dont know yet - we'll calculate it later.
        self.grad = 0.0

        # which values were used to create this one?
        # if c = a + b, then c's _prev = {a, b}
        # this is how we build the family tree of computations
        self._prev = set(_children)

        # what operation made this value( * + tanh , etc)
        # used for debugging and visualization
        self._op = _op

        # label for the node, optional. Just for our sanity
        self.label = label

        # this is the function that, when called, computes the gradients
        # for this Value's children. We'll fill this in for each operation
        # by default, it does nothing (lambda : None is a no-op function)
        self._backward = lambda: None

    def __repr__(self):
        # this controls what python prints when you do print(some_value).
        # e.g Value(data = 3.0)
        return f"Value(data = {self.data})"

    # ---- ADDITION ----
    def __add__(self, other):
        """
        self + other

        like combining two piles of blocks:
        if a=3 and b=4 then a+b = Value(7)
        """

        # If other is just a plain number (like 2.0), wrap it in a Value
        # so everything stays consistent.
        other = other if isinstance(other, Value) else Value(other)

        # creating an output value. its data is the sum.
        # we tell it: "your parents are self and other, and you were born from '+'"
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            """
            the chain rule for addition:
            if out = a + b and we know d(loss)/d(out), then
            d(loss)/d(a) = d(loss)/d(out) * d(out)/d(a)
            d(loss)/d(a) = d(loss)/d(out) * 1 (because out = a + b and the derivative
            of a+b w.r.t a is 1)
            similarly d(loss)/d(b) = d(loss)/d(out) * 1
            """
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward   # ← assigned here, AFTER the def block, INSIDE __add__
        return out

    def __radd__(self, other):
        # Handles the case where Python does: 2 + Value(3)
        # python tries 2.__add__(Value) first, fails, then tries Value.__radd__(2)
        return self + other

    # ---- MULTIPLICATION ----
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            """
            the chain rule for multiplication:
            if out = a * b
            d(loss)/d(a) = d(loss)/d(out) * d(out)/d(a)
            d(loss)/d(a) = d(loss)/d(out) * b (because out = a*b and derivative of
            a*b w.r.t a is b)

            similarly
            d(loss)/d(b) = d(loss)/d(out) * a
            """
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward   # ← assigned here, AFTER the def block, INSIDE __mul__
        return out

    def __rmul__(self, other):
        # Handles the case where Python does: 2 * Value(3)
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        # a - b is the same as a + (-b)
        return self + (-other)

    def __truediv__(self, other):
        # a/b is the same as a * b^(-1)
        return self * other**-1

    # ---- POWER ----
    def __pow__(self, other):
        """
        self ** other (self to the power of other)

        Like: Value(3) ** 2 = Value(9)
        Note: 'other' must be a plain number here (int, float), not a Value.
        """
        assert isinstance(other, (int, float)), "only int/float powers supported"

        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            """
            Power rule from Calc1:
            d/dx (x^n) = n * x^(n-1)

            So: self.grad += (other * self.data**(other-1)) * out.grad
            """
            self.grad += (other * self.data**(other - 1)) * out.grad

        out._backward = _backward   # ← assigned here, AFTER the def block, INSIDE __pow__
        return out

    # ---- TANH (activation function) ----
    def tanh(self):
        """
        tanh squishes any number into the range (-1, 1).

        Why do we need this?
        Without an activation function, stacking layers is pointless -
        it collapses into one big linear equation.

        tanh adds "non-linearity" - the network can learn curves, not just straight lines.

        tanh(0) = 0
        tanh(1) = 0.76
        tanh(-1) = -0.76

        Very Large Numbers -> close to 1
        Very Negative Numbers -> close to -1
        """
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)  # tanh formula

        out = Value(t, (self,), 'tanh')

        def _backward():
            """
            the derivative of tanh(x) is 1 - tanh(x)^2

            notice when t is near 1 or -1, the gradient is near 0
            this is called "vanishing gradient" - a known problem with tanh

            So: self.grad += (1 - t**2) * out.grad
            """
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward   # ← assigned here, AFTER the def block, INSIDE tanh
        return out

    # ---- BACKPROPAGATION ----
    def backward(self):
        """
        This is the big function!! We will be calling this on our loss value,
        and it figures out the gradient of EVERY value in the computation graph.

        It works in two steps:
        1. Build a "topological order" of all Values (parents before children)
        2. Walk backwards through that order, calling _backward() on each
        """

        # Step 1: topological sort
        # think of it like: you can't do the dishes until you have eaten, and
        # you can't eat until you have cooked.
        # we need to process values in the right order
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)  # process children first
                topo.append(v)         # then append self

        build_topo(self)

        # Step 2: The gradient of the loss w.r.t. itself is always 1.
        # d(loss)/d(loss) = 1. This seeds the whole backward pass.
        self.grad = 1.0

        # Step 3: Walk backwards, calling each node's _backward()
        for v in reversed(topo):
            v._backward()


class Neuron:
  """
  One single neuron. Like one tiny decison - maker.

  It takes a list of inputs, multiplies each of them by a weight,
  adds a bias, then squshes through tanh.

  Example: a neuron with 3 inputs has 3 weights and 1 bias. = 4 parameters.
  """

  def __init__(self, nin):
    """

    nin : number of inputs this neuron recieves

    initialize weights randomly between -1 and 1.
    why random: if all weights start at 0, every neuron learns the same thing
    each input gets its own weight: "How much do I care for this input?"

    """
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]

    #one bias. This is like a "default lean" - even if all inputs are ,
    # the bias lets the neuron still fire a non-zero output.
    self.b = Value(random.uniform(-1, 1))


  def __call__(self, x):
    """
    Run the neuron on inputs x.

    step 1: Weighted sum = w1*x1 + w2*x2 + ... + wn*xn + b
    step 2: apply tanh to squish the result

    """

    #zip (self.w, x) pairs each weight with its corresponding input
    # wi * xi computes the weighted contribution of each input

    #sum(..., self.b) adds them all up, starting from the bias
    act = self.b
    for wi, xi in zip(self.w, x):
      act = act + wi * xi

    out = act.tanh()
    return out

  def parameters(self):
    return self.w + [self.b]


class Layer:
  """
  A layer is just a bunch of neuron sitting side by side.

  All neurons in a layer recieve the SAME inputs,
  but each has its own weights and produce its own output.

  If you have 4 neurons in a layer, you get 4 outputs.

  """

  def __init__(self, nin, nout):
    """
    nin : number of inputs
    nout : number of neurons in this layer
    """
    self.neurons = [Neuron(nin) for _ in range(nout)]


  def __call__(self, x):
    #Run every neuron on the input x, collect all their outputs
    outs = [n(x) for n in self.neurons]

    #If only one neuron , return the value directly (not a list)
    #this makes the output layer cleaner for single output problems

    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    #collect parameters from ALL neurons in this layer

    return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
  """
  MLP = Multi Layer Perceptron. This is a feedforward neural network.

  You give it:
  - nin : number of inputs
  -nouts : a list of number of neurons per layer

  Example: MLP(3, [4, 4, 1])
  - 3 inputs
  - 4 neurons in the first layer
  - 4 neurons in the second layer
  - 1 output
  """

  def __init__(self, nin, nouts):
    #Build the size of each connecntion:
    # [nin, nouts[0], nouts[1], ..., nouts[-1]]
    # e.g for MLP (3, [4,4,1]) -> [3, 4, 4 ,1]
    sz = [nin] + nouts

    #create layers. layer i connects sz[i] inputs to sz[i+1] outputs.
    #zip(sz, sz[1:]) pairs consecutive sizes: (3,4 ), (4,4), (4,1)
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
        # Forward pass: feed x through each layer in sequence
        # The output of one layer becomes the input to the next
        for layer in self.layers:
            x = layer(x)
        return x

  def parameters(self):
        # Collect ALL parameters from ALL layers
        return [p for layer in self.layers for p in layer.parameters()]

  @property
  def num_params(self):
        return len(self.parameters())

  def __repr__(self):
        layer_sizes = [len(l.neurons) for l in self.layers]
        return f"MLP({self.layers[0].neurons[0].w.__len__()} → {' → '.join(map(str, layer_sizes))})"
