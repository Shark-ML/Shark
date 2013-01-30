

Loss and Cost Functions
=======================


Background
----------

Loss and cost functions are an integral part of machine learning.
Assume a model :math:`f` and a data set formed by pairs
of inputs and labels :math:`x_i` and :math:`t_i`, respectively.
For every input :math:`x_i` the model produces the label predictions
:math:`y_i=f(x_i)`. Then one important question is how far away the
model predictions are from the actual labels. Different loss and cost
functions propose different ways of calculating the "level of
disagreement" between the :math:`t_i` and :math:`y_i`.
Losses and costs can on the one hand be used for evaluating a trained
model. On the other hand, they can also be an criterion under which
learning and model optimization takes place. Further, these two roles
need not be taken by the same loss or cost function.


Costs and Losses in Shark
-------------------------

The major difference between a loss and a cost in Shark is an
assumption on how they can be calculated. In detail, a loss function
:math:`L` is by convention understood to support sample-wise evaluation,
for example to calculate the average error

.. todo::

    i think it is a bit confusing how the tutorial is about costs and
    losses, and suddenly all important explanations rely on the term
    "error". would it be possible to get completely rid of the word
    "error" in this tutorial? if not, how close can we get?


:math:`E` in the following way:

.. math::
  E = \frac 1 N \sum_{i=1}^N L(z_i,t_i)

In other words, a loss only needs to know the prediction of the model
for the i-th input and its target label at a time. One typical example
may be the zero-one loss commonly used with classification tasks.

In contrast, cost functions :math:`C` rely on all prediction-label pairs
being available at once. An error measure directly based on a cost function
will hence not be separable and directly have to compute:

.. math::
  E = C(z_1\dots z_N,t_1,\dots,t_N)

While such a structure or restriction makes any calculations less convenient,
such cost functions arise naturally for example in clustering, since the
quality of the clustering depends on all points at the same time.

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

.. todo::

    the above paragraph does not convince me at all. first, can we have
    a clearer example than clustering? second, the overall quality of a
    trained classification algorithm also depends on all points. i have
    zero experience in evaluation of clustering algorithms, but i do
    not see any reason why i could not propose a quality measure that
    can evaluate one single point at a time and sum over all of them.
    if this is a good/commonly-used criterion is another question, but
    the above paragraph is extremely general in this regard.



Derivatives
&&&&&&&&&&&


When both the loss function and the model are differentiable, it is possible
to calculate the derivative of the above error with respect to the model
parameters :math:`w`:

.. math::
  \frac {\partial}{\partial w} E = \frac 1 N \sum_{i=1}^N \frac {\partial}{\partial f(x_i)}L(f(x_i),t_i)\frac {\partial}{\partial w}f(x_i)

Thus allowing efficient embarassingly parallelizable gradient descent on the
error function.




The base class 'AbstractCost<LabelTypeT,OutputTypeT>'
-----------------------------------------------------


The base class :doxy:`AbstractCost` is templatized with respect to both the
label and output type, and the corresponding batch types are inferred:


========================   ==================================================
Types                      Description
========================   ==================================================
LabelType                  Type of a label :math:`t_i`
OutputType                 Type of a model output :math:`z_i`
BatchLabelType             Batch of Labels. Same as Batch<LabelType>::type.
BatchOutputType            Batch of Outputs. Same as Batch<OutputType>::type.
========================   ==================================================



Like all other interfaces in Shark, cost functions have flags indicating their
internal capabilities:



=========================================  ==================================================================
Flag, Accessor function                    Description
=========================================  ==================================================================
HAS_FIRST_DERIVATIVE, hasFirstDerivative   Can the cost function calculate its first derivative?
IS_LOSS_FUNCTION, isLossFunction           The cost function is in fact separable (a loss in the above terms)
=========================================  ==================================================================



The interface of AbstractCost reflects the fact that costs can only be evaluated
on a complete set of data. The following functions can be used for evaluation of
``AbstractCost``. For brevity let ``L`` be the ``LabelType`` and ``O`` the
``OutputType``:


==============================================================================================   ===============================================================================
Method                                                                                           Description
==============================================================================================   ===============================================================================
``double eval(Data<L> const& label, Data<O> const& predictions)``                                Returns the cost of the predictions :math:`z_i` given the label :math:`t_i`.
``double operator()(Data<L> const& label, Data<O> const& predictions)``                          Returns eval(label,predictions). Convenience function.
``double evalDerivative(Data<L> const&label, Data<O> const& predictions, Data<O>& gradient)``    Returns the error of the predictions :math:`z_i` given the label :math:`t_i`
                                                                                                 and computes :math:`\frac {\partial}{\partial z_i}L(z_i,t_i)`
==============================================================================================   ===============================================================================




The base class 'AbstractLoss<LabelTypeT,OutputTypeT>'
-----------------------------------------------------


The base class :doxy:`AbstractLoss` is derived from AbstractCost. It implements
all methods of its base class and offers several additional methods. Shark code is
allowed to read the flag ``IS_LOSS_FUNCTION`` via the public method ``isLossFunction()``,
and downcast an AbstractCost object to an AbstractLoss. This enables the use of the
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

.. todo::

    then this should also be the example used in the text above, no?


====================  ======================================================
Model                 Description
====================  ======================================================
:doxy:`NegativeAUC`   Area under the ROC (receiver operating characteristic)
                      curve. Value is negated so that it plays well with
                      optimizers (which perform minimization by convention).
====================  ======================================================



Loss Functions:


============================================  ==============================================================================
Model                                         Description
============================================  ==============================================================================
:doxy:`AbsoluteLoss`                          returns the :math:`L_1`-norm of the distance, :math:`|t-z|_1`
:doxy:`SquaredLoss`                           returns the squared distance in two-norm
                                              :math:`|t-z|_2^2`. Standard regression loss.
:doxy:`ZeroOneLoss`                           returns 0 if :math:`t_i=z_i` otherwise 1. Standard classification loss.
:doxy:`DiscreteLoss`                          uses a cost matrix to calculate errors in a discrete output and label
                                              space (general classification loss).
:doxy:`CrossEntropy`                          for maximization of class membership under some model assumptions.
                                              Useful for training of neural networks with linear outputs.
:doxy:`CrossEntropyIndependent`               maximization optimizes a model for simultaneously finding a set of attributes.
:doxy:`NegativeClassificationLogLikelihood`   interprets a network with n output neurons with outputs summing to one as
                                              conditional propability :math:`p(z_i|x_i)`. Used for classifier training.
============================================  ==============================================================================



.. todo::

    i think the descriptions in the right table need some update.
    for example, the one for CrossEntropyIndependent does not make sense;
    and NegativeClassificationLogLikelihood is independent of
    NNs, and this may hold for crossentropy as well (the description
    of which also does not say what it actually does, only what it's
    for..)? Also, if I don't misinterpret the AbsoluteLoss code, then
    it is not the 1-norm that is used to calculate the distance?? This
    needs to be checked!!! If there is a misunderstanding about the 1-norm,
    then the other tutorials should be revisited again as well.
