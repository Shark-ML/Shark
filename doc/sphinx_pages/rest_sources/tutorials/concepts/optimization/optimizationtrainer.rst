Optimizers and Trainers
=======================


Shark 3 comes with two related but different software architectures that
at a first glance both represent the act of learning, or the application
of learning rules: optimizers and trainers.
This article explains the differences between the two, and how they are
combined with the ErrorFunction and OptimizationTrainer classes.


Optimization in Shark
---------------------

Optimization methods are at the basis of many machine learning problems.
These can be as diverse as minimizing a smooth (regularized) empirical
risk functional, or searching a genotype space for a (local) minimum.

General purpose optimization has been difficult to realize with older
versions of Shark for two reasons: First, the interfaces for direct
(evolutionary) optimization and gradient-based optimization have been
part of the different sub-libraries EALib and ReClaM with incompatible
interfaces. Second, the gradient-based search interfaces had the
consequence that general-purpose optimization algorithms were only
applicable to problems expressed as functions of model parameters.

In Shark 3 all optimization problems inherit the
:doxy:`AbstractObjectiveFunction`
base class, and all optimization strategies are derived from
:doxy:`AbstractOptimizer`.
At this general level there is no explicit link to machine learning
problems yet, such that it is easy to use the library for any type of
optimization problem. Of course, the available methods are of relevance
for machine learning.

In Shark, optimization is modeled as a process of iterative refinement.
For example, such a refinement step can correspond to a generation of an
evolutionary algorithm or to a gradient step. This iterative proceeding
is reflected in the :doxy:`AbstractOptimizer::step()` interface.


Training of Models in Shark
---------------------------

In Shark, predictions about data are made by models, and the parameters
of models are set and modified by trainers, which encode learning rules.
At a first glance, if machine learning problems are formulated as
optimization problems, what then is the difference between an optimizer
(modeled by :doxy:`AbstractOptimizer`) and trainer (modeled by :doxy:`AbstractTrainer`)?
The concepts do indeed differ in the following three aspects:

First, model training is an atomic step, in contrast to an iterative
optimization process. This is reflected by the
:doxy:`AbstractTrainer::train` interface. 

Second, model training is understood as machine learning, thus, learning
from data. Hence, training data needs to be provided to the train
method. In contrast, the more general optimization framework may but
does not need to involve data.

Third, a trainer encodes a specific learning rule, in contrast to a
general optimization strategy. For example, the :doxy:`IRpropPlus`
algorithm is a very general gradient-descent algorithm, which can be
used to minimize any differentiable function, no matter whether it
depends on training data and model parameters or not. In contrast,
training a linear model to do a linear discriminant analysis
(:doxy:`LDA`) is specific to a certain class of models, in this case
linear models, and requires a certain type of training data, in this
case a classification problem. Thus, :doxy:`IRpropPlus` is an
optimizer, while :doxy:`LDA` is a trainer. This entails that the whole
training procedure is known to the trainer and thus very special
optimization strategies can be applied - for example if the problem
is convex and separable.


Model Training as an Optimization Problem
-----------------------------------------

Now let us consider an important mixed case, namely training a feed
forward neural network with gradient descent on the squared error using
:doxy:`IRpropPlus`. This requires the construction of the squared error
of the network output as an objective function which can be optimized
by the :doxy:`IRpropPlus` algorithm. This functionality is provided by
the :doxy:`ErrorFunction` class. The :doxy:`ErrorFunction` knows about
data, a model, and a loss function, and it computes the average loss of
the model output on the given data. Thus, the first way to train the
network is to apply the :doxy:`IRpropPlus` algorithm to an
:doxy:`ErrorFunction` object that knows about the network, a
:doxy:`SquaredLoss` object, and our training data.

However, Shark provides a second mechanism for training the same
network, which amounts to casting the above process into a trainer.
The :doxy:`OptimizationTrainer` class encapsulates the above iterative
training procedure by means of three objects:
An :doxy:`AbstractLoss`, an optimizer, and a stopping
criterion. The last component is encapsulated by the
:doxy:`AbstractStoppingCriterion` interface. This interface allows
the :doxy:`OptimizationTrainer` to implement the optimization loop
and to make training based on iterative optimization a single opaque
step.
