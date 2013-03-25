Normalization of Input Data
============================================================

This short tutorial will demonstrate how data can be normalized using
Shark. Read the basic :ref:`data tutorials <label_for_data_tutorials>`
first if you are not familiar with the :doxy:`Data` containers.

Shark normalizes data by training a :doxy:`Normalizer` model. Two
different trainers for two different types of normalization are
available. The trainers are :doxy:`NormalizeComponentsUnitInterval` and
:doxy:`NormalizeComponentsUnitVariance`. The first one normalizes
every input dimension to the range [0,1], the other adjusts the variance
of each component to one, and it can optionally remove the mean. This is
no whitening, because correlations remain unchanged.
For whitening, use the :doxy:`PCA`.

In the following we will normalize data to unit variance. First we
have to train our linear model so that it can perform the
normalization::

  #include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
  using namespace shark;

  int main()
  {
    //load data from a file or generate it
    UnlabeledData<RealVector> trainingData = loadData();

    Normalizer<> normalizer;
    NormalizeComponentsUnitVariance<> normalizingTrainer;
    normalizingTrainer.train(normalizer, trainingData);

  }

Now the normalizer is ready to use and we can transform the dataset to get the::

  trainingData = model(trainingData);

This will copy the training data and disconnect it
from previously created subsets of this set. Thus previously created
subsets won't be normalized, but all subsets created afterwards. In
order to apply such a normalization to :doxy:`LabeledData`, the methods
``transformInputs`` and ``transformLabels`` can be used. Of course,
the test data can be normalized as well, mutually or separately. The
following example trains and transforms the labels of a regression task::

  int main()
  {

    //load data somehow from a file or generate it
    LabeledData<RealVector,RealVector> trainingData = loadData();
    std::size_t labelSize = labelDimension(trainingData); //size of label vector

    //train normalizer
    Normalizer<> labelNormalizer;
    NormalizeComponentsUnitVariance<> normalizingTrainer(true);  // true: remove mean
    normalizingTrainer.train(labelNormalizer,trainingData.labels());

    //apply normalizer
    trainingData = transformLabels(trainingData, labelNormalizer);
  }

You can concatenate a normalizer
with another model. This comes handy when a model should be used
to handle a stream of new input data. Only one call to eval is needed
to use the normalization followed by the trained model::

  #include<shark/Models/ConcatenatedModel.h>
  //...

  YourModel model;
  ConcatenatedModel<RealVector,RealVector> completeModel = normalizer >> model;


For a more complex example of how normalization can be used, see the
tutorial about training the :doc:`../../algorithms/extreme_learning_machine` with the
complete example source :doxy:`elmTutorial.cpp`.
