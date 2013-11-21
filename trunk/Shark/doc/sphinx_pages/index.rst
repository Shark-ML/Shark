.. toctree::
   :hidden:

   rest_sources/downloads/downloads


   rest_sources/getting_started/installation
   rest_sources/getting_started/troubleshooting
   rest_sources/getting_started/using_the_documentation


   rest_sources/tutorials/tutorials

   rest_sources/tutorials/first_steps/your_programs
   rest_sources/tutorials/first_steps/hello_shark
   rest_sources/tutorials/first_steps/general_optimization_tasks
   rest_sources/tutorials/first_steps/when_to_stop
   rest_sources/tutorials/first_steps/shark_layout


   rest_sources/tutorials/concepts/library_design/designgoals
   rest_sources/tutorials/concepts/library_design/models
   rest_sources/tutorials/concepts/library_design/kernels
   rest_sources/tutorials/concepts/library_design/losses
   rest_sources/tutorials/concepts/library_design/optimizers
   rest_sources/tutorials/concepts/library_design/objective_functions
   rest_sources/tutorials/concepts/library_design/stopping_criteria
   rest_sources/tutorials/concepts/library_design/trainers
   rest_sources/tutorials/concepts/library_design/batches

   rest_sources/tutorials/concepts/library_design/writing_kernels
   rest_sources/tutorials/concepts/library_design/writing_objective_functions

   rest_sources/tutorials/concepts/lin_alg/vector_matrix
   rest_sources/tutorials/concepts/lin_alg/lapack

   rest_sources/tutorials/concepts/data/datasets
   rest_sources/tutorials/concepts/data/labels
   rest_sources/tutorials/concepts/data/import_data
   rest_sources/tutorials/concepts/data/dataset_subsets
   rest_sources/tutorials/concepts/data/normalization

   rest_sources/tutorials/concepts/optimization/optimizationtrainer
   rest_sources/tutorials/concepts/optimization/directsearch
   rest_sources/tutorials/concepts/optimization/conventions_derivatives

   rest_sources/tutorials/concepts/misc/statistics
   rest_sources/tutorials/concepts/misc/factory
   rest_sources/tutorials/concepts/misc/serialization


   rest_sources/tutorials/algorithms/pca
   rest_sources/tutorials/algorithms/lda
   rest_sources/tutorials/algorithms/linearRegression
   rest_sources/tutorials/algorithms/nearestNeighbor
   rest_sources/tutorials/algorithms/kmeans
   rest_sources/tutorials/algorithms/LASSO

   rest_sources/tutorials/algorithms/ffnet
   rest_sources/tutorials/algorithms/sparse_ae
   rest_sources/tutorials/algorithms/extreme_learning_machine

   rest_sources/tutorials/algorithms/cart
   rest_sources/tutorials/algorithms/rf

   rest_sources/tutorials/algorithms/svm
   rest_sources/tutorials/algorithms/svmModelSelection
   rest_sources/tutorials/algorithms/svmLikelihoodModelSelection
   rest_sources/tutorials/algorithms/lkc-mkl
   rest_sources/tutorials/algorithms/linear-svm
   rest_sources/tutorials/algorithms/kta

   rest_sources/tutorials/algorithms/cma
   rest_sources/tutorials/algorithms/mocma

   rest_sources/tutorials/algorithms/rbm_module
   rest_sources/tutorials/algorithms/binary_rbm

   rest_sources/tutorials/algorithms/quadratic_programs


   rest_sources/tutorials/concepts/misc/versatile_classification


   rest_sources/tutorials/for_developers/codingconvention
   rest_sources/tutorials/for_developers/the_build_system
   rest_sources/tutorials/for_developers/development_environment
   rest_sources/tutorials/for_developers/effective_ublas
   rest_sources/tutorials/for_developers/writing_tutorials
   rest_sources/tutorials/for_developers/managing_the_documentation
   rest_sources/tutorials/for_developers/issuing_a_release


   rest_sources/faq/faq


   rest_sources/showroom/showroom


   rest_sources/about_shark/news
   rest_sources/about_shark/todo
   rest_sources/about_shark/copyright


Summary
=======

.. note::

     This is Shark 3.0 beta. See the :doc:`news <rest_sources/about_shark/news>` for more information.

**SHARK is a fast, modular, feature-rich open-source C++ machine learning library**.
It provides methods for linear and nonlinear optimization, kernel-based learning
algorithms, neural networks, and various other machine learning techniques (see the
feature list :ref:`below <label_for_feature_list>`).
It serves as a powerful toolbox for real world applications as well as research.
Shark depends on `Boost <http://www.boost.org>`_ and `CMake <http://www.cmake.org/>`_.
It is compatible with Windows, Solaris, MacOS X, and Linux. Shark is licensed under
`GPLv3 <http://gplv3.fsf.org/>`_.

For an overview over the previous major release of Shark (2.0) we
refer to:

.. container:: cibox

  Christian Igel, Verena Heidrich-Meisner, and Tobias Glasmachers.
  `Shark <http://jmlr.csail.mit.edu/papers/v9/igel08a.html>`_.
  Journal of Machine Learning Research 9, pp. 993-996, 2008.
  [:ref:`Bibtex <label_for_citing_shark>`]


Where to start
%%%%%%%%%%%%%%

In the menu above, click on "Getting started", or use this direct link to the
:doc:`installation instructions <rest_sources/getting_started/installation>`.
After installation, there is a guide to the different documentation pages available
:doc:`here <rest_sources/getting_started/using_the_documentation>`.


Why Shark?
%%%%%%%%%%

**Speed and flexibility**

Shark provides an excellent trade-off between flexibility and
ease-of-use on the one hand, and computational efficiency on the other.

**One for all**

Shark offers numerous algorithms from various machine learning and
computational intelligence domains in a way that they can be easily
combined and extended.

**Unique features**

Shark comes with a lot of powerful algorithms that are to our best
knowledge not implemented in any other library, for example in the
domains of model selection and training of binary and multi-class SVMs,
or evolutionary single- and multi-objective optimization.


.. _label_for_feature_list:

Selected features
%%%%%%%%%%%%%%%%%

Shark currently supports:

* Supervised learning

  * Linear discriminant analysis (LDA), Fisher--LDA
  * Naive Bayes classifier (supporting generic distributions)
  * Linear regression
  * Support vector machines (SVMs) for one-class, binary and true
    multi-category classification as well as regression; includes fast variants for linear kernels.
  * Feed-forward and recurrent multi-layer artificial neural networks
  * Radial basis function networks
  * Regularization networks as well as Gaussian processes for regression
  * Iterative nearest neighbor classification and regression
  * Decision trees and random forests

* Unsupervised learning

  * Principal component analysis
  * Restricted Boltzmann machines (including many state-of-the-art
    learning algorithms)
  * Hierarchical clustering
  * Data structures for efficient distance-based clustering

* Evolutionary algorithms

  * Single-objective optimization (e.g., CMA--ES)
  * Multi-objective optimization (in particular, highly efficient
    algorithms for computing as well as approximating the contributing hypervolume)

* Fuzzy systems

* Basic linear algebra and optimization algorithms

