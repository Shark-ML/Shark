
Creating and Using Subsets of Data
==================================

.. warning::

    This tutorial is compltely outdated and does not reflect way, shark handles subsets anymore.

In this tutorial we will take a closer look at Data
partitioning. Read the basic :ref:`data tutorials <label_for_data_tutorials>`
first if you are not familiar with the :doxy:`Data` containers.

Having read in data, we most often want
to partition them in at least two parts: training and
test data. We use the training data for machine training
and evaluate our model on the test data. Often we also
want to split the training data again in smaller parts,
for example for selecting parameters with cross-validation.
The :doxy:`Data` class and its sub-classes
:doxy:`UnlabeledData` and :doxy:`LabeledData` are perfectly
suited for these tasks, since they can share data very
efficiently via subsets. Creating a subset only costs
the space needed to store the positions of its elements.
Duplication of previously created subsets costs *constant*
space and time, and thus is basically free. This tutorial
will focus on :doxy:`RegressionDataset` as an exemplary
subclass of a :doxy:`Data` container, but the same methods
can be applied to :doxy:`Data` and its sub-classes alike.

These are our initial declarations::

	RegressionDataset data;
	RegressionDataset subset;
	RegressionDataset complement;
	std::vector<std::size_t> indices;
	std::string name;

Subsets of a :doxy:`Data` object are described by indices.
So the index `i` being an element of the indices vector means
that the i-th element will be element of the subset. There is
a whole family of functions for defining and creating subsets
as well as their complements::

    // creation from indices
    data.indexedSubset(indices, subset);
    data.indexedSubset(indices, subset, complement);

    // random subsets of a certain size (with
    // unique elements, drawn without replacement)
    data.randomSubset(size, subset);
    data.randomSubset(size, subset, complement);

    // sub-range [0, end)
    data.rangeSubset(end, subset);
    data.rangeSubset(end, subset, complement);

Of course, we may want to save the created subsets at some point
and pass them around. For this, :doxy:`Data` can store named
subsets. Just choose the name of the subset (a string) and call ::

    data.createNamedSubset(name, indices);
    data.createNamedSubset(name, subset);

.. warning::
    The `subset` object must be a "sibling" object of the `data` object,
    so, it must share its underlying data. This is the case if the `subset`
    object has indeed been created as a subset of the `data` object, and
    none of the two has been modified thereafter. Otherwise this method
    throws an exception.

The existence of a subset can be checked later with ::

    bool result = data.hasNamedSubset(name);

and subsets can be accessed with ::

    data.namedSubset(name, subset);
    data.namedSubset(name, subset, complement);

where the complement of the subset with respect to the full set is calculated
on the fly. Calling :doxy:`Data::namedSubset` with an unknown name causes an
exception.

.. note::
    Note that there are some special names. A subset called "training"
    is returned by :doxy:`LabeledData::trainingSet`, and the subset "test" is
    returned by :doxy:`LabeledData::testSet`. Also the names "fold"+number are
    reserved for cross-validation subsets as described in the follwing.

A data set can be split randomly into training and test subsets
by calling ::

    data.randomSubset(size, subset, complement);
    data.createNamedSubset("training", subset);
    data.createNamedSubset("test", complement);


.. todo::

    also mention splitAfterElement - this is right now only
    done in the hello-world tutorial and clearly also belongs here.


Cross-Validation
----------------------------

Cross-Validation uses a number of training and validation subsets,
or folds. Typically, the data is distributed
evenly across the validation subsets. The training subsets are then
constructed as the complements of the validation sets. The model is
trained and validated on all folds and the mean performance is the
cross-validation performance.

The previously described methods are insufficient or at least
cumbersome for creating cross-validation folds. For this purpose
there exists a header file that offers special tools: ::

    #include <shark/Data/CVDatasetTools.h>

This file provides a bunch of functions for the creation of folds.
The subset definitions are automatically stored as named subsets. ::

    // Creates IID drawn partitions of the data set (without replacement).
    createCVIID(data, numberOfPartitions);

    // Creates partitions of approximately the same size.
    createCVSameSize(data, numberOfPartitions);

    // Creates indexed cross-validation sets. For each element the
    // index describes the fold in which the data point acts as a
    // validation example.
    createCVIndexed(data, numberOfPartitions, indices);

For the special case of classification there also exists a function
that ensures that all partitions have approximately the same fraction
of examples of each class. The function supports vector labels with
one-hot encoding and integer class labels (see also :doc:`labels`) ::

    createCVSameSizeBalanced(data, numberOfPartitions);

As noted previously, the folds have the names "fold"+number.
For forward compatibility these names should be created with
the helper function ::

    std::string name = getCVPartitionName(partitionNumber);

Now, using the folds for cross-validation is simple, for
example in grid search for a good regularization parameter: ::

    RegressionDataset dataset;
    createCVSameSize(dataset, numberOfFolds);

    for(double regularization = 0; regularization < 1; regularization += 0.1)
    {
        double result = 0;
        for (unsigned fold = 0; fold != numberOfFolds; ++fold) {
            // access the fold
            RegressionDataset training;
            RegressionDataset validation;
            dataset.namedSubset(getCVPartitionName(fold), training, validation);

            // train with your problem and return the optimal value
            result += trainProblem(training, validation, regularization);
        }
        result /= numberOfFolds;

        // remember the best setting
        if (result < bestValidationError)
        {
            bestValidationError = result;
            bestRegularization = regularization;
        }
    }

A slightly more complex example program can be found at :doxy:`CVFolds.cpp`
