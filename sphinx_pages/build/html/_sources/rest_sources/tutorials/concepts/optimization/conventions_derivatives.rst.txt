Shark Conventions for Derivatives
=================================

Background
----------

Gradient-based optimization is the most common reason for computing
derivatives in Shark. Gradient-based optimizers operate on objective
functions, however, these will often delegate the job further on to
models, kernels, and loss or more general cost functions. Thus, all
these types of objects have the capability to compute derivatives
foreseen in their respective interfaces. However, getting the best
performance when evaluating derivatives is not always straightforward.
Since Shark aims for maximal speed, the library enforces a very
specific evaluation scheme for derivative computations. The design
rationale will be explained in the following.

An Example: The Derivative of the Error
---------------------------------------

Let's consider a simple example, namely the derivative of a squared
error term

.. math::
  E = \frac 1 N \sum_{i=1}^N L(f_w(x_i), t_i)

w.r.t. a parameter :math:`w_i` of a parametric family :math:`f_w(x)`
of models, evaluated with the squared loss :math:`L(y, t) = (y-t)^2`:

.. math::
  \frac{\partial E}{\partial w_k} &= \frac 1 N \sum_{i=1}^N L(f_w(x_i), t_i)

                                  &= \frac 1 N \sum_{i=1}^N \frac{\partial L}{\partial y}(f_w(x_i), t_i) \frac{\partial f_w}{\partial w_k}(x_i)

                                  &= \sum_{i=1}^N \frac 2 N (f_w(x_i) - t_i) \frac{\partial f_w}{\partial w}(x_i)

The derivative involves the chain rule for the combination of model
and loss. The term can be understood as a weighted sum of the partial
model derivatives, where the weights are the loss derivatives. Note
how these weights do not require the model derivatives, but they do
depend on the model's output values. This is the general situation
when chaining computations:

.. math::
  \frac{\partial f \circ g(x)}{\partial w_k} = f'(g(x)) g'(x)

The value :math:`g(x)` is needed as the point in which the derivative
:math:`f'` is to be evaluated, and it is used rather independent of
the derivative :math:`g'(x)`.

In a typical error function the overall derivative is a weighted sum of
model derivatives, evaluated in different points. The weights require
only the model evaluations in these points, not their derivatives. This
hints at the following order of evaluation:

  * evaluate the model values :math:`y_i = f_w(x_i)`,
  * evaluate the loss derivatives :math:`\frac{\partial L}{\partial y}(y_i, t_i)`,
  * evaluate the model derivatives :math:`\frac{\partial f}{\partial w_i}(x_i)`
    and compute their weighted sum.



Two-stage Derivative Computation
--------------------------------


The order of computation as laid out above is a necessity
for the efficient evaluation of derivatives of chains. We lift this necessity
to a principle: `first evaluate, then differentiate`. In other words,
always call `eval` on an object before calling `evalDerivative`
or similar functions. Otherwise the results of the derivative are
undefined. This holds for models and kernels. Objective functions and losses
can compute both at once since they can be interpreted as the ends of the chain
of computation - the loss is required to evaluate and return the full derivative
while the objective function returns the final summed result.

In simple situations the order of evaluation is not crucial. However,
in general the requirement to evaluate before computing derivatives is
not restrictive at all. So even if there is no natural order of calls,
the order is dictated by the Shark interface design.

The rationale behind this design is that there are often strong
synergies between the computation of the value and its derivative.
More often than not, the derivative is a cheap byproduct once the
value has been computed. Thus, efficient evaluation of both the value
and the derivative requires either the computation of both at the same
time, or the storage of intermediate results. The first way is not
viable in case of chained computations. Therefore Shark has decided to
take the second route, and to store intermediate values for derivative
computations in the object. This state is written by the evaluation
method and read by the derivative computations.




Another Example: The derivative of a concatenation of models
------------------------------------------------------------


Let's assume the function :math:`f_w` of the previous example would in fact
not be a single model, but two models where the output of the first model is
the input of the second, such that with :math:`w=(u,v)` being the combined parameter
vector of both models, we get :math:`f_w(x)=g_u(h_v(x))`.Thus the derivatives are:

.. math::
  \frac{\partial E}{\partial u_k}f_w(x) =\frac{\partial g_u}{\partial u_k}(h_v(x))

and

.. math::
  \frac{\partial E}{\partial v_k}f_w(x)
  = \frac{\partial g_u}{\partial h_v(x)}(h_v(x)) \frac{\partial h_v}{\partial v_k} h_v(x)

Please remember that the partial derivatives with respect to the arguments of :math:`g` are
full jacobi matrices and not single values or vectors. Thus the computation of :math:`v_k`
as stated here requires a matrix-vector product for every parameter :math:`v_k`, or a
matrix-matrix product if the derivative is computed for all :math:`v_k` at once.
But putting this into the the equation of the derivative of the error function of the
previous example, we  get for the derivative with respect to :math:`v_k`:

.. math::
  \frac{\partial E}{\partial v_k}
  &= \sum_{i=1}^N \frac 2 N (f_w(x_i) - t_i) \frac{\partial f_w}{\partial v_k}(x_i)\\
  &= \sum_{i=1}^N \frac 2 N (f_w(x_i) - t_i) \frac{\partial g_u}{\partial h_v(x_i)}(h_v(x_i)) \frac{\partial h_v}{\partial v_k} h_v(x_i)

now adding braces around the derivative of the loss and the partial derivative of :math:`g_u`
we see that this term can be computed as matrix-vector product. Thus the whole Term can be
computed using 2N matrix-vector products instead of N matrix-matrix and matrix-vector products!
This makes in practice a huge difference.




Weighted Sums of Derivatives
----------------------------


In the first example, the derivative of the squared error w.r.t. a model
parameter is a weighted sum of derivatives for single data points. The
computation of the weights requires the model's output values for the
same data points. Again, this situation is completely general, and thus
Shark makes it a principle: `derivatives are returned as weighted sums`.

A single call to a derivative function may evaluate the derivative in a
whole batch or even in a whole data set of different points. However, in
the next processing stage these values will typically all enter the same
cost function. Thus, the derivative is a weighted sum, with the cost
derivatives being the weights.

Now for chaining of the derivatives as in the second example, we can first
evaluate the weighted derivative with respects to the inputs of :math:`g`,
which amounts to computing the aforementioned bracing. After that the resulting
vector can be used to calculate the weighted derivative of :math:`h` with
respect to it's parameters. We can further optimize this scheme by
computing both derivatives of :math:`g` at the same time using that again
in many cases input and parameter derivative can share a lot of computations.

A well known example which uses both optimizations weighted sum calculation
and shared computation of derivatives is the back-propagation of error algorithm,
which not only allows for a more efficient computation in terms of the complexity
of the algorithm, but also allow for a more efficient optimization
for RAM throughput, etc. To check that this is in fact the same algorithm, define
:math:`g` and :math:`h` as neuron layers of a three layer neural network.

Thus Shark's derivative interfaces can be understood as a generalization of
the same computational trick. The exact weighting scheme applied slightly
varies across the different interfaces, e.g., models versus kernels.


Batching Derivatives and how to derive them
-------------------------------------------


As previously mentioned in short, batching is also applied to derivatives. The net
effect of batch computing is not as dramatic for the computation time as the application
of weighted derivatives, but still quite significant. However, deriving efficient batched
computations of weighted derivatives is not straight forward and we are constantly trying
to improve the results.
In the case of the parameter derivative for example the input is a matrix of values:
every row consists of one weight for every output and each row represents one sample.
Computing this derivative naiively, the result would be a three tensor which
we need to reduce to a single vector by summing over two dimensions. Thus choosing the
order in which this reduction is performed - preferently without actually calculating
the big tensor itself - can make a huge difference. The key to success in any case is
to use matrix notation wherever possible instead of using elementwise derivations
as is often done on a sheet of paper. While Vector and Matrix calculus sems unfamiliar at
first glance, it immediately answers the questions about which computations can be grouped
together and which efficient linear algebra operations can be used.


.. todo::

   TG: Present one simple and one involved use case? Or is this the wrong place?

   OK: We need one tutorial for Kernels and Models which explain how these
   derivatives can actually b calculated. Maybe introduce your nice scalar
   product syntax which makes the calculation a breese.
