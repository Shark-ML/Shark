============================
Sparse Autoencoder
============================

Background
----------

A sparse autoencoder is an unsupervised learning algorithm using a
neural network, setting the target values equal to the input. In
other words, the autoencoder tries to approximate the identity function.
By imposing additional constraints we can make the network learn
interesting features about the input.

One such constraint is the number of hidden units in the network. If
this is below the number of input variables the network is actually
learning a compressed representation of the input. Another constraint
we will use is to have the average activation of the neurons being low.
This is done by using a regularization term using the KL-divergence:

.. math ::
   KL(\rho \| \rho_j) = \rho \log(\frac{\rho}{\rho_j}) +
   (1 - \rho) \log(\frac{1-\rho}{1-\rho_j})

See the documentation of the :doxy:`SparseFFNetError` class for more info.


Sparse Autoencoder in Shark
---------------------------

Edges in natural images
^^^^^^^^^^^^^^^^^^^^^^^

As an example of finding features with a sparse autoencoder, we will
use natural images as input. We will then see how the autoencoder
discovers edges as a good representation of natural images.

The images we will use are the same as in the `Stanford Tutorial
<http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder>`_.


Generating a training set
^^^^^^^^^^^^^^^^^^^^^^^^^

In order to generate a training set, we will randomly select image patches
of equal size (we use *8x8*) and use each patch as a data point: ::

  #include <shark/Data/Csv.h>
  #include <shark/LinAlg/VectorStatistics.h>
  ...
  using namespace std;
  using namespace shark;
  ...
  const unsigned int w = 512, h = 512;
  const unsigned int numsamples = 10000;
  const unsigned int psize = 8;
  ...
  UnlabeledData<RealVector> images;
  import_csv(images, "directory/containing/the/images.csv");
  vector<RealVector> patches;

  size_t patchesPerImg = numsamples / images.numberOfElements();
  typedef typename UnlabeledData<RealVector>::element_range::iterator ElRef;

  for (ElRef it = images.elements().begin(); it != images.elements().end(); ++it) {
      for (size_t i = 0; i < patchesPerImg; ++i) {
          // Upper left corner of image
          unsigned int ulx = rand() % (w - psize);
          unsigned int uly = rand() % (h - psize);
          // Transform 2d coordinate into 1d coordinate and get the sample
          unsigned int ul = ulx * h + uly;
          RealVector sample(psize * psize);
          const RealVector& img = *it;
          for (int j = 0; j < psize; ++j)
              for (int k = 0; k < psize; ++k)
                  sample(j * psize + k) = img(ul + k + j * h);
          patches.push_back(sample);
      }
  }

  samples = createDataFromRange(patches);

This creates a data set containing *10000* patches of size *8x8* transformed
into a vector with *64* elements. Before we can use this data set we need
to normalize the data points. We first truncate outliers to *+/- 3* standard
deviations, and then normalize to the range :math:`[0.1, 0.9]`. We also need
to make it a regression data set with same input and target values: ::

  // zero mean
  RealVector meanvec = mean(samples);
  for (ElRef it = samples.elements().begin(); it != samples.elements().end(); ++it) {
      *it -= meanvec;
  }
  // Remove outliers outside of +/- 3 standard deviations
  // and normalize to [0.1, 0.9]
  RealVector pstd = 3 * sqrt(variance(samples));
  for (ElRef it = samples.elements().begin(); it != samples.elements().end(); ++it)
  {
      for (size_t idx = 0; idx < it->size(); ++idx) {
          double trunced = max(min(pstd(idx), (*it)(idx)), -pstd(idx)) / pstd(idx);
          (*it)(idx) = (trunced + 1.0) * 0.4 + 0.1;
      }
  }

  RegressionDataset data(samples, samples);


Sparse autoencoder objective function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to create the objective function for the sparse autoencoder,
we first need to create a feed-forward neural network with the correct
layout: ::

  #include <shark/ObjectiveFunctions/SparseFFNetError.h>
  #include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
  #include <shark/ObjectiveFunctions/Regularizer.h>
  #include <shark/ObjectiveFunctions/CombinedObjectiveFunction.h>
  ...
  const unsigned int numhidden = 25;
  const double rho = 0.01;
  const double beta = 6.0;
  const double lambda = 0.0002;
  ...
  FFNet<LogisticNeuron, LogisticNeuron> model;
  model.setStructure(psize * psize, numhidden, psize * psize, true, false, false, true);

We then need to add the sparsity constraint: ::

  SquaredLoss<RealVector> loss;
  SparseFFNetError error(&model, &loss, rho, beta);
  error.setDataset(data);

and weight regularization: ::

  TwoNormRegularizer regularizer(error.numberOfVariables());
  CombinedObjectiveFunction<VectorSpace<double>, double> func;
  func.add(error);
  func.add(lambda, regularizer);

This creates the entire objective function for the sparse autoencoder,
with sparsity constraint and weight regularization.


Training the autoencoder
^^^^^^^^^^^^^^^^^^^^^^^^

In order to train the autoencoder we use the limited memory BFGS (L-BFGS)
algorithm with a line search satisfying the wolfe conditions. We also need
to chose a starting point for the optimization. For this we use values
uniformly taken from :math:`[-r, r]` for the weights and :math:`0` for the
biases, with

.. math ::
    r = \frac{\sqrt{6}}{n_{in} + n_{out} + 1}

where :math:`n_{in}` and :math:`n_{out}` is the number of input and output
values per neuron.

The training is then done as follows: ::

  #include <shark/Algorithms/GradientDescent/LBFGS.h>
  ...
  const unsigned int maxIter = 400;
  ...
  LBFGS optimizer;
  optimizer.lineSearch().lineSearchType() = LineSearch::WolfeCubic;
  optimizer.init(func, startingPoint);

  for (unsigned int i = 0; i < maxIter; ++i) {
      optimizer.step(func);
  }

In our trials we got final error values around 0.8 to 0.9.


Visualizing the autoencoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^

After training, each row of the first weight matrix, :math:`W1`, will
correspond to a feature learned by the autoencoder. To visualize these
features, we export each row as an *8x8* PGM image using the PGM library
of Shark, but first some normalization is done: ::

  boost::format filename("output/feature%d.pgm");

  // Find the mean value for normalization
  double m = 0.0;
  for (size_t i = 0; i < W.size1(); ++i)
      for (size_t j = 0; j < W.size2(); ++j)
          m += W(i,j);
  m /= W.size1() * W.size2();

  // Create feature images
  for (size_t i = 0; i < W.size1(); ++i)
  {
      // Rescale with the mean. Then normalize.
      double top = 0.0;
      RealVector img(W.size2());
      for (size_t j = 0; j < W.size2(); ++j) {
          img(j) = W(i,j); - m;
          top = max(top, img(j));
      }
      img /= top;
      exportPGM((filename % i).str().c_str(), img, psize, psize, true);
  }

After scaling the features to *50x50* images an plotting them next to
each other, we got the following result

.. figure:: ../images/features.*
  :scale: 100%
  :alt: Plot of features learned by the autoencoder


Full example program
--------------------

A complete program performing the above steps is :download:`SparseAETutorial.cpp
<../../../../../examples/Unsupervised/SparseAETutorial.cpp>`.
