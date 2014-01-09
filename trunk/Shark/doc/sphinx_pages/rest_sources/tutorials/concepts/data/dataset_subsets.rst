Creating and Using Subsets of Data
==================================

On of the most common operations with datasets, represented by the :doxy:`Data`
or :doxy:`LabeledData` classes,  are the creation of different types of datasets.
There are different types of subsets, and thus Shark offers a wide variety
of functions and objects for handling subsets.

Basics of Subset Generation
-----------------------------------

**The data classes are designed to generate subsets
at the level of batches** and not at the level of single points as outlined in the :doc:`Data tutorial <datasets>`.
Thus for most types of subsets the points inside the set need to be reordered across the batch structure.

After that, acquiring subsets is easy and inexpensive in terms of memory and runtime, as only references
to the batches are shared across the datasets. To acquire a subset, the following basic functions can be used::

    LabeledData<I,L> dataset;//our dataset
    //create an indexed subset of batches
    std::vector<std::size_t> indices;//indices of the batches to be contained in the subset
    LabeledData<I,L> subset = indexedSubset(dataset,indices);
    //if also the complement of the set is needed, the call is:
    LabeledData<I,L> complement;
    dataset.indexedSubset(indices,subset,complement);
    //create subsets by considering ranges of batches
    LabeledData<I,L> range1 = rangeSubset(dataset,start,end);//contains batches with indices start,...,end-1
    LabeledData<I,L> range2 = rangeSubset(dataset,end);//contains batches with indices 0,...,end-1

The functions of course also work for :doxy:`Data` and :doxy:`UnlabeledData` objects.

Splitting
----------------------------
Splitting is a special type of subset generation where one part of the dataset is removed from the dataset
and returned as a new one. We use this most often in the generation of training and test sets.
There are two types of splits: The first one is splitting at the level of batches, we call this operation splicing::

  LabeledData<I,L> right = dataset.splice(k);

After this call dataset contains the elements [0,...,k-1] and the right part starting from element k is returned as a new dataset.

The second type of splitting is on the level of elements, for example when the first *k* elements of a file make up the training set,
we can write::

  LabeledData<I,L> right = splitAtElement(dataset,k);

The semantics are the same as in splice, however if k happens to be in the middle of a batch, it is splited in two parts before
applying the splicing operation.

Cross-Validation
----------------------------

Cross-Validation uses a number of training and validation subsets,
or folds. Typically, the data is distributed evenly across the validation subsets.
The training subsets are then constructed as the complements of the validation sets.
The model is trained and validated on all folds and the mean performance is the
cross-validation performance. As the elements are usually reshuffled randomly between the folds,
the whole dataset needs to be reorganized. The tools for this can be included using::

    #include <shark/Data/CVDatasetTools.h>

This file provides a bunch of functions for the creation of folds. When the folds
are created, the dataset is reorganized, which needs an intermediate copy of the
dataset. This has to be taken into account when big datasets are to be used.
Afterwards, aside from reorganizing the dataset, a new object is returned,
:doxy:`CVFolds`. For the dataset, it stores which batches belong to which
training and validation set and how many folds were created. Before we describe
the functions to create the  cross validation dataset, we present a small usage example
which tries to find a good regularization parameter for a given problem. We assume here
the existence of some function `trainProblem` which takes training and validation set as
well as the regularization parameter and returns the validation error::

    RegressionDataset dataset;
    CVFolds<RegressionDataset> folds = createCVSameSize(dataset, numberOfFolds);

    for(double regularization = 0; regularization < 1; regularization += 0.1)
    {
        double result = 0;
        for (unsigned fold = 0; fold != folds.size(); ++fold) {
            // access the fold
            RegressionDataset training = folds.training(i);
            RegressionDataset validation = folds.validation(i);

            // train with your problem and return the optimal value
            result += trainProblem(training, validation, regularization);
        }
        result /= folds.size();

        // remember the best setting
        if (result < bestValidationError)
        {
            bestValidationError = result;
            bestRegularization = regularization;
        }
    }

A slightly more complex example program can be found at :doxy:`CVFolds.cpp`.
Now we present the basic splitting functions provided by Shark. they are::

    // Creates partitions of approximately the same size.
    createCVSameSize(data, numberOfPartitions);

    // Creates IID drawn partitions of the data set (without replacement).
    createCVIID(data, numberOfPartitions);

    // Creates indexed cross-validation sets. For each element the
    // index describes the fold in which the data point acts as a
    // validation example.
    createCVIndexed(data, numberOfPartitions, indices);

For the special case of classification there also exists a function
that ensures that all partitions have approximately the same fraction
of examples of each class (i.e., for stratified sampling). The function supports vector labels with
one-hot encoding and integer class labels (see also :doc:`labels`)::

    createCVSameSizeBalanced(data, numberOfPartitions);

.. Caution::

   Note that some of the above operations may subtly change the
   dataset from which the partitions were created. For example,
   ``createCVSameSizeBalanced(data, numberOfPartitions);`` will
   change the order of examples in ``data``.



Nested Cross-Validation
----------------------------

Sometimes we want to use a nested Cross-Validation scheme. That is, after we chose
one training and validation set, we want to repeat this scheme, applying another
level of cross-validation. Unfortunately, this is not directly supported in an
efficient manner right now, but we can handle it using an explicit copy of
the training set::

    //as created in the above example
    RegressionDataset training = folds.training(i);
    RegressionDataset validation = folds.validation(i);
    //explicit copy!
    training.makeIndependent();
    //creating a new fold
    CVFolds<RegressionDataset> innerFolds = createCVSameSize(training, numberOfFolds);

One-vs-One Partitioning
------------------------------------------------

This is a special subset creation mechanism used in One-vs-One schemes for multiclass problems.
In this case, we often want to look at the binary classification
problems created by all pairs of classes.
For doing so,  we first reorganize the dataset such that all elements of one class are grouped together and
every batch contains only elements of one class::

    repartitionByClass(data);

Afterwards, we can create binary subproblems of this set by issuing::

    RegressionDataset subproblem = binarySubProblem(data,class0,class1);

The labels in the returned dataset are not the original class labels, but are created by
setting the label of all elements of ``class0`` to 0 and ``class1`` to 1.

Elementwise Subsets Using DataView
--------------------------------------

Sometimes it is not useful to reorganize the dataset for a subset. This for example happens if
a set of random subsets needs to be generated. In this case, we can us the :doxy:`DataView` class,
which wraps a dataset and provide fast random access to the elements as well as efficient subsetting::

    DataView<RegressionDataset> view(data);

    //creating a random subset using indices
    std::vector<std::size_t> indices;//somehow fill
    DataView<RegressionDataset> subset1 = subset(view,indices);

    //randomly choosing k elements out of the dataset:
    DataView<RegressionDataset> subset2 = randomSubset(view);
