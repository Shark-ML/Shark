Stopping Criteria
=================

This concept tutorial gives some additional information on stopping criteria
in Shark and notably a list of all those implemented.

A stopping criterion determines when optimization processes carried
out by :doc:`optimizers` should halt. Deciding when to stop is rather
difficult.  Typically, we would like to stop when the improvement
between two optimization steps is small, i.e., :math:`|E(t+1)-E(t)|<
\epsilon` for an objective function :math:`E`, for example an error
function computed on some training data, and some criterion for
"smallness" :math:`\epsilon`. However, there is no guarantee that a
solution obtained this way really is close to any local optimum -- it
could as well be just a saddle point or an area with small gradient.

When the optimization is employed for learning, an important aspect of
stopping a trial is early-stopping for regularization (i.e., avoiding
overfitting to the training set). A class of stopping heuristics
therefore monitors the development of the error on a hold-out part of the data
set -- the validation error :math:`V(t)`. The idea is that learning
should be stopped as soon as the validation error is not improving any
more, assuming that from then on overfitting takes over. Even this
error measurement is not perfect, since the validation error often
follows a very noisy path during training.  That fact has led to the
proposal of several different stopping critera, and this tutorial will
introduce those implemented in Shark.



The base class 'AbstractStoppingCriterion<ResultSetT>'
------------------------------------------------------


Stopping criteria are represented by the abstract interface
:doxy:`AbstractStoppingCriterion`. They are templatized on the ``ResultSet`` used:


==========================   =====================================================================
Types                        Description
==========================   =====================================================================
``ResultSet``                Defines the type of the result set used. This can be either the
                             :doxy:`SingleObjectiveResultSet` returned by most optimizers
                             or a :doxy:`ValidatedSingleObjectiveResultSet` that makes the
                             validation error of the current time step available to the algorithm.
==========================   =====================================================================


Every stopping criterion extracts useful information from the result set
at every time step and accumulates the information over time. Thus the
interface consists only of two functions.


============================================   =============================================================================
Method                                         Description
============================================   =============================================================================
``bool stop(ResultSet)``                       Updates the internal statistics and stops when the stopping criterion is met.
``reset()``                                    Resets the internal state before a new trial is started.
============================================   =============================================================================


Applying a stopping criterion is simple. For an iterative optimizer we
can simply write::

  MyStoppingCriterion criterion;
  MyOptimizer opt;
  MyTrainingErrorFunction E;
  opt.init(E);
  do{
      opt.step(E);
  } while( !criterion.stop( opt.solution() ) );

List of Stopping Criteria
-------------------------


.. todo::

    give the exact equations for all criteria in the table. also check the
    code again if they all really adhere to the Prechelt paper, i.e., .



===================================  =====================================================================================
Model                                Description
===================================  =====================================================================================
:doxy:`MaxIterations`                Stops after a fixed number of Iterations.
:doxy:`TrainingError`                Stops when the training error seems to converge, i.e., :math:`E(t-T)-E(t)< \epsilon`.
:doxy:`TrainingProgress`             Tracks the progress of the training error over a period of time, i.e.,
                                     :math:`\text{mean}\{E(t),\dots, E(t-T)\}/ \min_t E(t)< \epsilon`.
:doxy:`ValidatedStoppingCriterion`   Evaluates the validation error and hands the validated result to another criterion.
:doxy:`GeneralizationLoss`           Calculates the quotient :math:`V(t)/\min_t E(t)-1` as a relative measure of the gap
                                     between training and validation error.
:doxy:`GeneralizationQuotient`       Uses the quotient of training progress and generalization loss.
===================================  =====================================================================================
