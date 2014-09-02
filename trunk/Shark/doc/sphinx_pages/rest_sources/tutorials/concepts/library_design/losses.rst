

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

.. todo::

    i think what is missing here is a comment on the normalization
    conventions. i can't even quite figure them out on my own from the code.
    it seems that for losses, when the type passed to eval is data-of-something,
    then the result is normalized by the number of samples, and if i use the
    batch interface of eval, it is not normalized, correct? so, using the
    data interface is a statement that one wants normalization and otherwise
    on always does it oneself? i am aware that this could also go down to where
    the eval-interfaces are explained, but i think it should be at a more
    prominent place actually.


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
``double eval(Data<L> const& label, Data<O> const& predictions)``                                Returns the cumulated cost of the predictions :math:`z_i` given the label 
                                                                                                 :math:`t_i`. The loss is normalized by the number of points in the datasets, 
												 thus the mean loss is returned.
``double operator()(Data<L> const& label, Data<O> const& predictions)``                          Convenience function Returning eval(label,predictions)
``double evalDerivative(Data<L> const&label, Data<O> const& predictions, Data<O>& gradient)``    Returns the error of the predictions :math:`z_i` given the label :math:`t_i`
                                                                                                 and computes :math:`\frac 1 N \frac {\partial}{\partial z_i}L(z_i,t_i)`, where
												 :math:`N` is the number of data points. This function also returns the mean of
												 the loss as its return value.
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
``double eval(BatchLabelType const& T, BatchInputType const& Z)``                                             Returns the error of the predictions :math:`z_i \in Z` given the label :math:`t_i \in T`.
``double operator()(LabelType const& t, InputType const& z)``                                                 Calls eval(t,z)
``double operator()(BatchLabelType const& T, BatchInputType const& Z)``                                       Calls eval(T,Z)
``double evalDerivative(BatchLabelType const& T, BatchInputType const& Z, BatchInputType const& gradient)``   Returns the error of the predictions :math:`z_i` given the label :math:`t_i`
                                                                                                              and computes :math:`\frac {\partial}{\partial z_i}L(z_i,t_i)`
===========================================================================================================   =========================================================================================


List of Cost and Loss functions
-------------------------------


Currently only one instance of AbstractCost is implemented:


====================  ======================================================
Model                 Description
====================  ======================================================
:doxy:`NegativeAUC`   Area under the ROC (receiver operating characteristic)
                      curve. Value is negated so that it plays well with
                      optimizers (which perform minimization by convention)
====================  ======================================================



Loss Functions:


============================================  ==============================================================================
Model                                         Description
============================================  ==============================================================================
:doxy:`AbsoluteLoss`                          Returns the :math:`L_2`-norm of the distance, :math:`|t-z|_2`
:doxy:`SquaredLoss`                           Returns the squared distance in two-norm
                                              :math:`|t-z|_2^2`; standard regression loss
:doxy:`ZeroOneLoss`                           Returns 0 if :math:`t_i=z_i` otherwise standard classification loss
:doxy:`DiscreteLoss`                          Uses a cost matrix to calculate losses in a discrete output and label
                                              space (general classification loss)
:doxy:`CrossEntropy`                          Logarithmic likelihood function if the model outputs are 
                                              interpreted as exponents of a softmax classifier;
                                              useful, e.g., for training of neural networks with linear outputs
:doxy:`CrossEntropyIndependent`               Logarithmic likelihood function with
                                              additional independence assumptions
:doxy:`HingeLoss`			      Loss used in Maximum margin classification. Binary and multiclass implemented.
:doxy:`SquaredHingeLoss`		      Loss used in Maximum margin classification. It is the pointwise
					      Square of the HingeLoss. It is differentiable everywhere.
:doxy:`EpsilonHingeLoss`		      Loss for regression. It can be underestood as the 1-norm loss which is
					      cut off to 0 in a box of size epsilon around the label.
:doxy:`SquaredEpsilonHingeLoss`		      Maximum margin regression. It is zero in a ball of size epsilon around the 
					      label and outside the squared two-norm of the distance of prediction and label.
					      Thus very close points are not punished.
:doxy:`HuberLoss`			      Robust loss for rgression. It is quadratic close to 0 and becomes
					      a linear function for big discrepancies between model prediction and target.
:doxy:`TukeyBiweightLoss`		      Robust loss for regression. It is similar to the Huber loss, but instead
					      of becoming linear, it becomes constant.
============================================  ==============================================================================



.. todo::

    i think the descriptions in the right table need some update.
    for example, the one for CrossEntropyIndependent does not make sense;
