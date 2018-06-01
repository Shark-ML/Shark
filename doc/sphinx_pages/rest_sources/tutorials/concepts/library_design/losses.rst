

Loss and Cost Functions
=======================


Shark uses the notion of loss and cost functions to define machine
learning tasks.

Loss functions
--------------

Consider a model (a hypothesis) :math:`f` mapping inputs :math:`x`
to predictions :math:`y=f(x)\in Y`.  Let :math:`t\in Y` be the true
label of input pattern :math:`x`.  Then a *loss function*
:math:`L:Y\times Y\to\mathbb{R}^+_0` measures the quality of the
prediction. If the prediction is perfectly accurate, the loss function
is zero (:math:`t=y\Rightarrow L(t, y)=0`). If not, the loss
function measures "how bad" the mistake is. The loss can be
interpreted as a penalty or error measure.

For a classification tast, a fundamental loss function 
is the 0-1-loss:

.. math::
  L(y,t)=\begin{cases} 0 & \text{if $y=t$}\\1 & \text{otherwise}\end{cases}

For regression, the squared loss is most popular:

.. math::
  L(y,t)= (y-t)^2

Using the concept of a loss function, the goal of supervised learning
can be described as finding a model :math:`f` minimizing the *risk*:

.. math::
  \mathcal{R}(f) = \mathbb{E}\{   L(t, f(x)) \}

Here the expectation :math:`\mathbb{E}` is over the joint distribution 
underlying the observations of inputs and corresponding labels.

Cost functions
--------------

Now let us consider a collection of observations
:math:`S=\{(x_1,t_1),(x_2,t_2),\dots,(x_N,t_N)\}\in(X\times Y)^N` and
corresponding predictions :math:`y_1.y_2,\dots,y_N`  by a model :math:`f`.
A *cost function* :math:`C` is a mapping assigning 
an overall cost value, which can be interpreted as an overall error,
to :math:`\{(y_1,t_1),(y_2,t_2),\dots,(y_N,t_N)\}\in(Y\times Y)^N`.
Every loss function induces a cost function, namely the *empirical
risk*:

.. math::
  \mathcal{R}_S(f) = C(\{(y_1,t_1),(y_2,t_2),\dots,(y_N,t_N)\})  = \frac 1 N \sum_{i=1}^N L(y_i,t_i)

The cost function induced by the 0-1-loss is the average
misclassification error and the cost function induced by the squared
loss is the mean squared error (MSE).

However, there are cost functions which cannot be decomposed using a loss
function. For example, the *area under the curve* (AUC).
In other words, all loss functions generate a cost function, but not all cost
functions must be based on a loss function.

List of Classes
---------------------------------
See the documentation for :doxy:`Loss functions<lossfunctions>` and :doxy:`Cost functions <costfunctions>`.


Derivatives
&&&&&&&&&&&


When both the loss function and the model are differentiable, it is possible
to calculate the derivative of the empirical risk with respect to the model
parameters :math:`w`:

.. math::
  \frac {\partial}{\partial w}\mathcal{R}_S(f)  = \frac 1 N \sum_{i=1}^N \frac {\partial}{\partial f(x_i)}L(f(x_i),t_i)\frac {\partial}{\partial w}f(x_i)

This allows embarrassingly parallelizable gradient descent on the cost
function. Please see the tutorial
:doc:`../optimization/conventions_derivatives` for learning more about the
handling of derivatives in Shark.




The base class 'AbstractCost<LabelTypeT,OutputTypeT>'
-----------------------------------------------------


The base class :doxy:`AbstractCost` is templatized with respect to
both the label and output type.  Using batches, that is, collections
of input elements, is an important concept in Shark, see the tutorial
:doc:`batches`. The proper batch types are inferred from the
label and output types:


========================   ==================================================
Types                      Description
========================   ==================================================
LabelType                  Type of a label :math:`t_i`
OutputType                 Type of a model output :math:`z_i`
BatchLabelType             Batch of Labels; same as Batch<LabelType>::type
BatchOutputType            Batch of Outputs; same as Batch<OutputType>::type
========================   ==================================================



Like all other interfaces in Shark, cost functions have flags indicating their
internal capabilities:



=========================================  ==================================================================
Flag, Accessor function                    Description
=========================================  ==================================================================
HAS_FIRST_DERIVATIVE, hasFirstDerivative   Can the cost function calculate its first derivative?
IS_LOSS_FUNCTION, isLossFunction           Is the cost function a loss
                                           in the above terms (i.e., separable)?
=========================================  ==================================================================



The interface of AbstractCost reflects the fact that costs can only be evaluated
on a complete set of data. The following functions can be used for evaluation of
``AbstractCost``. For brevity let ``L`` be the ``LabelType`` and ``O`` the
``OutputType``:


==============================================================================================   ===============================================================================
Method                                                                                           Description
==============================================================================================   ===============================================================================
``double eval(Data<L> const& label, Data<O> const& predictions)``                                Returns the mean cost of the predictions :math:`z_i` given the label 
                                                                                                 :math:`t_i`.
``double operator()(Data<L> const& label, Data<O> const& predictions)``                          Convenience function Returning eval(label,predictions)
==============================================================================================   ===============================================================================




The base class 'AbstractLoss<LabelTypeT,OutputTypeT>'
-----------------------------------------------------


The base class :doxy:`AbstractLoss` is derived from AbstractCost. It implements
all methods of its base class and offers several additional methods. Shark code is
allowed to read the flag ``IS_LOSS_FUNCTION`` via the public method ``isLossFunction()``
and to downcast an AbstractCost object to an AbstractLoss. This enables the use of the
following much more efficient interface:


===========================================================================================================   =========================================================================================
Method                                                                                                        Description
===========================================================================================================   =========================================================================================
``double eval(LabelType const& t, InputType const& z)``                                                       Returns the error of the prediction :math:`z` given the label :math:`t`.
``double eval(BatchLabelType const& T, BatchInputType const& Z)``                                             Returns the sum of errors of the predictions :math:`z_i \in Z` given the label :math:`t_i \in T`.
``double operator()(LabelType const& t, InputType const& z)``                                                 Calls eval(t,z)
``double operator()(BatchLabelType const& T, BatchInputType const& Z)``                                       Calls eval(T,Z
``double evalDerivative(BatchLabelType const& T, BatchInputType const& Z, BatchInputType const& gradient)``   Returns the error of the predictions :math:`z_i` given the label :math:`t_i`
                                                                                                              and computes :math:`\frac {\partial}{\partial z_i}L(z_i,t_i)`
===========================================================================================================   =========================================================================================
