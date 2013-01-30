

Support Vector Machines: Likelihood-based Model Selection
=========================================================


Please first read the :doc:`svm` tutorial, and possibly also
the :doc:`svmModelSelection` tutorial for traditional approaches
to SVM model selection.



Motivation
----------


The previous tutorial explained that the performance of an SVM classifier
depends on the choice of regularization parameter :math:`C`
and the kernel parameters. There we also presented the most common method
to find SVM hyperparameters: grid search on the cross-validation error.
This is suited for kernels with only
one or two parameters, because a two- or three-dimensional SVM hyperparameter
search space can still be sufficiently covered by a fixed grid of search
points. Using naive heuristics like :doxy:`NestedGridSearch`, where the
search resolution increases iteratively, a four-dimensional SVM hyperparameter
space can maybe still be sampled sufficiently. But we do not get around the
fact that grid search-based SVM model selection suffers from the curse of
dimensionality.

Naturally, much research has been directed toward differentiable
estimates of, substitutes for, or bounds on the generalization error.
A good overview is given in our paper [GlasmachersIgel2010]_. There, we
also presented a novel model selection criterion that is differentiable
(in almost all practical cases) with respect to the regularization and
kernel parameters. In practical experiments, it compared very favorably to
other gradient-based model selection criteria. We consider it the current
state-of-the-art for gradient-based SVM model selection, and especially when
the number of examples is relatively small. In the next paragraphs we explain
how to use this maximum-likelihood based approach to SVM model selection in
Shark. For theoretical details and background, please consult the original article.

You can find the source code for the following example in
:doxy:`CSvmMaxLikelihoodMS.cpp`. (There, one trial is wrapped by the function
``run_one_trial()``, which takes a verbosity preference as
argument. The first trial is carried out verbosely, the 100 aggregating trials
(which take a long time) silently and only the result is being printed.)



The toy problem
---------------


Assume we have a higher- or high-dimensional kernel, for example an
"Automatic Relevance Detection" (ARD) kernel (in Shark the
:doxy:`ARDKernelUnconstrained`), which has one parameter for each input
dimension:

.. math::
  k(x, z) = \exp(\sum_i \gamma_i (x_i - z_i)^2 )

Such a kernel can be useful when the individual features correlate differently
with the labels, hence calling for individual bandwidths :math:`\gamma_i`
per feature (from another angle, learning the ARD kernel bandwidths corresponds
to learning a linear transformation of the input space).

In [GlasmachersIgel2010]_, a toy problem is introduced which well lends itself
to an ARD kernel and the optimization of its parameters. It creates a binary
classification dataset of dimension :math:`2d` in the following way: first, fix
a positive or negative label :math:`y`, i.e., ``1`` or ``0``, respectively. Then,
fill the first :math:`d` dimensions by

.. math::
	y - 0.5 + \mathcal N(0.0,1.0) \enspace ,

that is, produce Gaussian distributed noise around :math:`+0.5` for positive label
and :math:`-0.5` for negative label. The second :math:`d` dimensions are simply filled
with only Gaussian noise :math:`\sim \mathcal N(0.0,1.0)`. Overall, there will be
:math:`d` dimensions which are correlated with the labels and hence informative, and
:math:`d` dimensions which are not correlated with the labels and hence uninformative.

By design, this toy problem is well tailored to an ARD kernel. The ARD kernel
weights corresponding to the uninformative dimensions would best be optimized out
to be zero, since these dimensions on average hold no information relevant to the
classification problem. In the following, we will use our maximum-likelihood model
selection criterion to optimize the hyperparameters of an SVM using an ARD kernel
on such a toy problem. Ideally, the kernel weights will afterwards reflect the
nature of the underlying distribution.



Likelihood-based model selection in Shark
-----------------------------------------


The key class for maximum-likelihood based SVM model selection in Shark
is :doxy:`SvmLogisticInterpretation`, and we include its header. To create
the toy problem via the aptly named ``PamiToy`` distribution, we also include
the header for data distributions; and the gradient-based optimizer "Rprop",
with which we will optimize the SVM hyperparameters under the
:doxy:`SvmLogisticInterpretation` criterion (see source file for complete
list of includes)::

	#include <shark/Data/DataDistribution.h>
	#include <shark/Models/Kernels/ArdKernel.h>
	#include <shark/Algorithms/GradientDescent/Rprop.h>
	#include <shark/ObjectiveFunctions/SvmLogisticInterpretation.h>
	#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>



Creating the toy problem
&&&&&&&&&&&&&&&&&&&&&&&&



We create our problem dataset with :math:`d=5`::

	unsigned int useful_dim = 5;
	unsigned int noise_dim = 5;
	unsigned int total_dim = useful_dim + noise_dim;
	PamiToy problem( useful_dim, noise_dim );
	unsigned int train_size = 500;
	unsigned int test_size = 5000;
	ClassificationDataset train = problem.generateDataset( train_size );
	ClassificationDataset test = problem.generateDataset( test_size );

As usual, we normalize all data such that it has unit variance in the
training set::

	Normalizer<> normalizer;
	NormalizeComponentsUnitVariance<> normalizationTrainer;
	normalizationTrainer.train( normalizer, train.inputs() );
	transformInputs( train, normalizer );
	transformInputs( test, normalizer );

Then create the ARD kernel with appropriate dimensions
(kernel parameter initialization will come later)::

	DenseARDKernel kernel( total_dim, 0.1 );




Data folds and model selection criterion
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&



Before we go ahead and declare our model selection criterion (i.e., objective
funtion), we have to partition the training data into folds before that:
the :doxy:`SvmLogisticInterpretation` class requires to be passed data in the
form of a :doxy:`CVFolds` object, that is, in form of an existing partitioning
for cross-validation. This way, control over the type of data partitioning
(e.g., stratified vs. IID, etc.) strictly remains with the user::

	unsigned int num_folds = 5;
	CVFolds<ClassificationDataset> cv_folds = createCVIID( train, num_folds );

Three essential lines of this tutorial are now setting up the maximum-likelihood
based objective function for model selection::

	bool log_enc_c = true;
	QpStoppingCondition stop(1e-12);
	SvmLogisticInterpretation<> mlms( cv_folds, &kernel, log_enc_c, &stop );

The first line specifies that in this case, we want to allow for unconstrained optimization
of the regularization parameter (i.e., we do not want to bother with the possibility of the
optimizer accidentally driving :math:`C` into the negative half-space). However, ``true``
is also the default, so we could have omitted it had we not passed a custom stopping
criterion. The second line sets up a :doxy:`QpStoppingCondition` for all SVMs that the
SvmLogisticInterpretation will train internally to determine its own objective value.

.. admonition:: Note on the stopping criterion

	Here, the :doxy:`QpStoppingCondition` is
	set to a rather low, or conservative, value for the final KKT violation. In general,
	the computation of the :doxy:`SvmLogisticInterpretation` criterion is somewhat volatile
	and requires high computational accuracy. For that reason, we use a very conservative
	stopping criterion in this tutorial. In a real-world setting this can be relaxed somewhat,
	as long as the signs of the gradient of the :doxy:`SvmLogisticInterpretation` will be correct
	"often enough". To date, we do not have an airtight method to properly choose the stopping
	criterion so that it is loose enough to allow fast optimization, but tight enough to ensure
	a proper optimization path. A well-performing heuristic used in [GlasmachersIgel2010]_ was
	to set the 	maximum number of iterations to 200 times the input dimension. This	proved
	robust enough to have produced the state-of-the-art results given in the paper.

In the last line, we finally find the declaration of our objective function, which takes as
arguments the CVFolds object, kernel, log-encoding information, and the stopping criterion
(optional).



The optimization process
&&&&&&&&&&&&&&&&&&&&&&&&


Now we only need to set a starting point for the optimization process, and we choose
:math:`C=1` and :math:`\gamma_i = 0.5/(2d)` as motivated in the paper. Note that by
convention, the CSvmTrainer stores the regularization parameter :math:`C` last in the
parameter vector, and the SvmLogisticInterpretation honors this convention::

	RealVector start( total_dim+1 );
	if ( log_enc_c ) start( total_dim ) = 0.0; else start( total_dim ) = 1.0; //start at C = 1.0
	for ( unsigned int k=0; k<total_dim; k++ )
		start(k) = 0.5 / total_dim;

One single evaluation of the objective function at this current point looks like this::

	double start_value = mlms.eval( start );

and the value we get from that (on our development machine) is ``0.337388``.

Next, we set up an :doxy:`IRpropPlus` optimizer, choosing the same parameters
for it as in the original paper, except with a lower number of total iterations::

	IRpropPlus rprop;
	double stepsize = 0.1;
	double stop_delta = 1e-3;
	rprop.init( mlms, start, stepsize );
	unsigned int its = 50;

The main process of this tutorial, optimizing the SVM hyperparameters under the
SvmLogisticInterpretation objective function now is straightforward and follows
the general optimization schemes in Shark::

	for (unsigned int i=0; i<its; i++) {
		rprop.step( mlms );
		if ( rprop.maxDelta() < stop_delta )
			break;
	}



Introspection after optimization
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


After the optimization loop, we again query the current objective function value::

	double end_value = mlms.eval( rprop.solution().point );

and the value we get from that (on our development machine) is ``0.335099``, so the
initial parameter guess in this case was already quite good (in terms of the
associated objective function value).

We examine the improved hyperparameter values via::

	std::cout << "        C = " << ( log_enc_c ? exp( rprop.solution().point(total_dim) ) : rprop.solution().point(total_dim) ) << std::endl;
	for ( unsigned int i=0; i<total_dim; i++ )
		std::cout << "        gamma(" << i << ") = " << kernel.parameterVector()(i)*kernel.parameterVector()(i) << std::endl;

and get (on our development machine) the (unencoded, i.e., true, i.e.,
squared-parameter) values::

	C = 1.71335
	gamma(0) = 0.460517
	gamma(1) = 0.0193955
	gamma(2) = 0.0277312
	gamma(3) = 0.0235109
	gamma(4) = 0.0308288
	gamma(5) = 0
	gamma(6) = 0.000977712
	gamma(7) = 0
	gamma(8) = 0.0171233
	gamma(9) = 0


We see that in the majority of cases, the ARD kernel parameters corresponding
to uninformative feature dimensions were learned to be (close to) zero. However,
for some reason, the value of ``gamma(8)`` is almost in the range of its
informative counterparts.



Evaluation after optimization
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


We present two ways of applying the hyperparameters found in the model selection
process to an SVM trainer in order to get the final training and test errors.
In both cases, care must be taken at one spot or another to correctly specify
the encoding style for the regularization parameter, namely the same as previously
used by the SvmLogisticInterpretation object. The relevant lines are below marked
with an ``//Attention`` comment. We recommend the second variant, since
it does not rely on calling ``eval(...)`` on the objective function first.


Option 1: implicit/manual copy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


We of course also want to query the training and test classification errors by
applying the hyperparameters found to a new CSvmTrainer object::

	double C_reg = ( log_enc_c ? exp( rprop.solution().point(total_dim) ) : rprop.solution().point(total_dim) ); //ATTENTION: mind the encoding
	KernelExpansion<RealVector> svm( &kernel, true );
	CSvmTrainer<RealVector> trainer( &kernel, C_reg, log_enc_c ); //encoding does not really matter in this case b/c it does not affect the ctor
	trainer.train( svm, train );
	ZeroOneLoss<unsigned int, RealVector> loss;
	Data<RealVector> output;
	output = svm( train.inputs() );
	train_error = loss.eval( train.labels(), output );
	output = svm( test.inputs() );
	test_error = loss.eval( test.labels(), output );
	std::cout << "    training error:  " <<  train_error << std::endl;
	std::cout << "    test error:      " << test_error << std::endl;

The above code (only) works because we previously issued
``mlms.eval( rprop.solution().point );``, which implicitly copied the kernel
parameters from the RProp solution vector into the kernel function used by the
CSvmTrainer. We obtain

.. code-block:: none

	training error:  0.116
	test error:      0.1374



Option 2: using solution().point
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Alternatively, we can also directly copy all the hyperparameters as found by the optimizer
into the CSvmTrainer directly::

	KernelExpansion<RealVector> svm( &kernel, true );
	CSvmTrainer<RealVector> trainer( &kernel, 0.1, log_enc_c ); //ATTENTION: must be constructed with same log-encoding preference
	trainer.setParameterVector( rprop.solution().point ); //copy best hyperparameters to svm trainer
	trainer.train( svm, train );
	ZeroOneLoss<unsigned int, RealVector> loss;
	Data<RealVector> output;
	output = svm( train.inputs() );
	train_error = loss.eval( train.labels(), output );
	output = svm( test.inputs() );
	test_error = loss.eval( test.labels(), output );
	std::cout << "    training error:  " <<  train_error << std::endl;
	std::cout << "    test error:      " << test_error << std::endl;

And we get the same results

.. code-block:: none

	training error:  0.116
	test error:      0.1374




Repetition over 100 trials
&&&&&&&&&&&&&&&&&&&&&&&&&&


We now examine the distribution of hyperparameter values over several trials on
different realizations of the toy problem distribution. We repeat the experiment
100 times, and note the means and variances of the SVM hyperparameters. This
yields the following results (where the last/11th entry is the regularization
parameter C)::

	avg-param(0)    = 0.0174454  +- 0.000372237
	avg-param(1)    = 0.0243765  +- 0.00276891
	avg-param(2)    = 0.0170669  +- 0.000236762
	avg-param(3)    = 0.0148257  +- 0.000139686
	avg-param(4)    = 0.0175333  +- 0.000225192
	avg-param(5)    = 0.00810077 +- 0.000397033
	avg-param(6)    = 0.00831601 +- 0.000484481
	avg-param(7)    = 0.0134892  +- 0.000909667
	avg-param(8)    = 0.00652671 +- 0.000238294
	avg-param(9)    = 0.00863524 +- 0.000432687
	avg-param(10)   = 1.68555    +- 0.971377

	avg-error-train = 0.12594    +- 0.000294276
	avg-error-test  = 0.137722   +- 4.47983e-05

We see that on average, some tendency exists for the uninformative parameters
to be different from completely zero. At the same time, the :doxy:`SvmLogisticInterpretation`
objective clearly selects a meaningful model. Also note that the mean test error is
well below 14%, which is an excellent value for an SVM on this toy problem.




References
----------

.. [GlasmachersIgel2010] T. Glasmachers and C. Igel. Maximum Likelihood Model Selection
   for 1-Norm Soft Margin SVMs with Multiple Parameters. IEEE Transactions on Pattern
   Analysis and Machine Intelligence, 2010.

