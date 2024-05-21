Creating and Using Subsets of Data
==================================


A common operation with datasets is the creation of specific subsets.
There are different types of subsets, and thus Shark offers a wide
variety of functions and objects for handling subsets.

At this point recall that in Shark offers the data containers
:doxy:`Data`, :doxy:`UnlabeledData`, and :doxy:`LabeledData`. Most of
the functionalities presented in the following refer to all of these
containers. If you are not familiar with Shark data containers then you
way want to work through the tutorial :doc:`Data tutorial <datasets>`
first.


Basics of Subset Generation
-----------------------------------

**The data classes are designed to generate subsets
at the level of batches** and not at the level of single points.
This concept is outlined in the tutorial :doc:`Data tutorial <datasets>`.
Thus for most types of subsets the points inside the set need to be
reordered across the batch structure.

After that, acquisition of subsets is easy and inexpensive in terms of
memory and runtime, as only references to the batches are shared across
the datasets. This is one of the core features of Shark's data containers.
It can save the day, e.g., when performing 10-fold cross-validation.

We start by including the data set header::


	#include <shark/Data/Dataset.h>
	using namespace shark;
	

The following basic functions acquire subsets::


		LabeledData<I,L> dataset;             // our dataset
	
		// create an indexed subset of batches
		std::vector<std::size_t> indices;     // indices of the batches to be contained in the subset
		LabeledData<I,L> subset = dataset.indexedSubset(indices);
	

The functions of course also work with :doxy:`Data` and
:doxy:`UnlabeledData` objects.


Splitting
----------------------------

Splitting is a special type of subset generation where one part of the
dataset is removed from the dataset and returned as a new one. We use
this most often in the generation of training and test sets. There are
two types of splits: The first one is splitting at the level of batches,
we call this operation splicing::


		LabeledData<I,L> remaining_batches = dataset.splice(k);
	

After this call dataset contains the batches [0,...,k-1] and the
remaining part starting from element k is returned as a new dataset.
This is obviously most useful if we already know the batch structure
of the data object.

The second type of splitting is on the level of elements, for example
when the first *k* elements of a file make up the training set, we can
write::


		LabeledData<I,L> remaining_elements = splitAtElement(dataset, k);
	

The semantics are the same as in splice, however if k happens to be in
the middle of a batch, it is split into two parts before applying the
splicing operation.


Cross-Validation
----------------------------

Cross-Validation uses a number of training and validation subsets called
folds. Typically, the data is distributed evenly across the validation
subsets. The training subsets are then constructed as the complements of
the validation sets. A model is trained and validated systematically on
all splits and the mean performance is the cross-validation performance.
Since the elements are usually reshuffled randomly between folds the
whole data container needs to be reorganized. The tools for this can be
included using::


	#include <shark/Data/CVDatasetTools.h>
	

This file provides a bunch of functions for the creation of folds. The
data container is reorganized in this process, which requires an
intermediate copy. This has to be taken into account when using big
data sets. Aside from the reorganization of the data set a new object of
type :doxy:`CVFolds` is created. It stores the number of folds as well
as which batch belongs to which fold. Before we describe the functions
to create the cross validation data set we present a small usage example
that tries to find a good regularization parameter for a given problem.
We assume here the existence of some function `trainProblem` which takes
training and validation set as well as the regularization parameter and
returns the validation error::


		RegressionDataset dataset;
	
		CVFolds<RegressionDataset> folds = createCVSameSize(dataset,4);
	
		double bestValidationError = 1e4;
		double bestRegularization = 0;
		for (double regularization = 1.e-5; regularization < 1.e-3; regularization *= 2) {
			double result = 0;
			for (std::size_t fold = 0; fold != folds.size(); ++fold){ //CV
				// access the fold
				RegressionDataset training = folds.training(fold);
				RegressionDataset validation = folds.validation(fold);
				// train
				result += trainProblem(training, validation, regularization);
			}
			result /= folds.size();
	
			// check whether this regularization parameter leads to better results
			if (result < bestValidationError)
			{
				bestValidationError = result;
				bestRegularization = regularization;
			}
	
			// print status:
			std::cout << regularization << " " << result << std::endl;
		}
	

Now we present the basic splitting functions provided by Shark. they are::


		// Creates partitions of approximately the same size.
		createCVSameSize(data, numberOfPartitions);
	
		// Creates IID drawn partitions of the data set (without replacement).
		createCVIID(data, numberOfPartitions);
	
		// Creates indexed cross-validation sets. For each element the
		// index describes the fold in which the data point acts as a
		// validation example. This function offers maximal control.
		createCVIndexed(data, numberOfPartitions, indices);
	

For the special case of classification there also exists a function
that ensures that all partitions have approximately the same fraction
of examples of each class (i.e., for stratified sampling). The function
supports vector labels with one-hot encoding and integer class labels
(see also :doc:`labels`)::


		createCVSameSizeBalanced(data, numberOfPartitions);
	

.. Caution::

   Note that some of the above operations may subtly change the
   data container from which the partitions were created. For example,
   ``createCVSameSizeBalanced(data, numberOfPartitions);`` will change
   the order of examples in ``data``.


Nested Cross-Validation
----------------------------

Sometimes we want to use a nested Cross-Validation scheme. That is,
after we chose one training and validation set, we want to repeat this
scheme, applying another level of cross-validation. Unfortunately, this
is not directly supported in an efficient manner right now, but we can
handle it using an explicit copy of the training set::


		// as created in the above example
		RegressionDataset training = folds.training(i);
		RegressionDataset validation = folds.validation(i);
		// explicit copy!
		training.makeIndependent();
		// creating a new fold
		CVFolds<RegressionDataset> innerFolds = createCVSameSize(training, numberOfFolds);
	


One-vs-One Partitioning
------------------------------------------------

This is a special subset creation mechanism used in One-vs-One schemes
for multiclass problems. In this case, we often want to look at the
binary classification problems created by all pairs of classes.
For doing so, we first reorganize the dataset such that all elements of
one class are grouped together and every batch contains only elements of
a single class::


		ClassificationDataset data;
		// ...
		repartitionByClass(data);
	

Afterwards, we can create binary subproblems of this set by issuing::


		ClassificationDataset subproblem = binarySubProblem(data, class0, class1);
	

The labels in the returned dataset are not the original class labels,
but are created by setting the label of all elements of ``class0`` to 0
and of ``class1`` to 1.


Element-wise Subsets with DataView
--------------------------------------

Sometimes it is not useful to reorganize the dataset for a subset. This
happens for example if a set of random subsets needs to be generated. In
this case we can us the :doxy:`DataView` class, which wraps a data set
and provides fast random access to the elements as well as efficient
subsets::


	#include <shark/Data/DataView.h>
	
		DataView<ClassificationDataset> view(data);
	
		// creating a random subset from indices
		std::size_t k = 100;
		std::vector<std::size_t> indices(view.size());
		for (std::size_t i=0; i<view.size(); i++) indices[i] = i;
		for (std::size_t i=0; i<k; i++) std::swap(indices[i], indices[rand() % view.size()]);
		indices.resize(k);
		DataView<ClassificationDataset> subset1 = subset(view, indices);
	
		// same functionality in one line
		DataView<ClassificationDataset> subset2 = randomSubset(view, k);
	
