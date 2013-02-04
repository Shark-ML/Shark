Trainers
========

Trainers offer one of the simplest interfaces in Shark, but also form
a very diverse group of algorithms for a range of different settings
and applications. Each trainer represents a one-step solver for one
certain machine learning problem. Usually, it will adapt the parameters
of a model such that it represents the solution to some machine learning
or data processing problem given a certain data set.

Some trainers are very simple -- for example those training a linear
model such that it normalizes the components of all examples in a data
set to unit variance. Other trainers are quite complex, for example for
some multi-class support vector machines. But in most cases, trainers
are used to reach analytical solutions to relatively simple problems
which are not stated as an iterative optimization problem underneath.



The base class 'AbstractTrainer<ModelT, LabelTypeT>'
----------------------------------------------------

:doxy:`AbstractTrainer` is the base interface for trainers for supervised
learning problems. It is templatized with respect to the type of model it
trains and the type of label in the data set. The trainer then defines the
following types in its public interface:


==========================   =======================================
Types                        Description
==========================   =======================================
``ModelType``                The type of model the trainer optimizes
``InputType``                The type of inputs the model takes
``LabelType``                The type of the labels in the data set
==========================   =======================================


A trainer offers the following methods:


=========================================================   =================================================
Method                                                      Description
=========================================================   =================================================
``train(ModelType&, LabeledData<InputType, LabelType>)``    Solves the problem and sets the model parameters
``configure(PropertyTree)``                                 Configures the trainer
``std::string name()``                                      Returns the trainer's name
=========================================================   =================================================


Usage of trainers is equally straightforward::

  MyModel model;
  MyTrainer trainer;
  MyDataset data;
  trainer.configure(...);      //optional
  trainer.train(model, data);  //model now represents the solution to the problem.



The base class 'AbstractUnsupervisedTrainer<ModelT>'
----------------------------------------------------


:doxy:`AbstractUnsupervisedTrainer` is the base interface for trainers for
unsupervised learning poblems. It only needs to know about the model type,
and offers a typedef for the data format:


==========================   ==============================================
Types                        Description
==========================   ==============================================
``ModelType``                Type of model which the trainer optimizes
``InputType``                Type of inputs the model takes
==========================   ==============================================


These trainers also offer the following methods:


=====================================================   ================================================
Method                                                  Description
=====================================================   ================================================
``train(ModelType&, UnlabeledData<InputType>)``         Solves the problem and stores it in the model.
``configure(PropertyTree)``                             Configures the trainer
``std::string name()``                                  Returns the name of the trainer
=====================================================   ================================================




List of trainers
----------------


We first list the unsupervised trainers in Shark. Many
of these operate on models for data normalization.


========================================  ========================  ============================================================
Trainer                                     Model                     Description
========================================  ========================  ============================================================
:doxy:`NormalizeComponentsUnitInterval`   :doxy:`Normalizer`        Trains a linear model to normalize the components of data
                                                                    to the unit interval.
:doxy:`NormalizeComponentsUnitVariance`   :doxy:`Normalizer`        Trains a linear model to normalize the components of data
                                                                    to unit variance.
:doxy:`NormalizeComponentsWhitening`      :doxy:`LinearModel`       Trains a linear model to whiten the data (uncorrelate all
                                                                    components, and normalize to unit variance).
:doxy:`PCA`                               :doxy:`LinearModel`       Trains a linear model for an orthogonal transformation
                                                                    such that the data is decorrelated and the new coordinate
                                                                    axes point in the successive directions of highest variance.
:doxy:`NormalizeKernelUnitVariance`       :doxy:`ScaledKernel`      Trains the scaling factor of a ScaledKernel such that the
                                                                    data has unit variance in its induced feature space. Note
                                                                    how this trainer operates on a kernel rather than a (linear)
                                                                    model.
:doxy:`OneClassSvmTrainer`                :doxy:`KernelExpansion`   Trains a one-class SVM.
========================================  ========================  ============================================================



List of supervised trainers:



========================================  ========================================   ===================================================================
Trainer                                     Model                                      Description
========================================  ========================================   ===================================================================
:doxy:`CARTTrainer`                       :doxy:`CARTClassifier`                     Trains a classification and regression tree
:doxy:`FisherLDA`                         :doxy:`LinearModel`                        Finds a subspace which maximizes the class label variance.
:doxy:`KernelMeanClassifier`              :doxy:`KernelExpansion`                    Computes the class means in the kernel induced feature
                                                                                     space and generates a classifier which assigns the points
                                                                                     to the class of the nearest mean.
:doxy:`LDA`                               :doxy:`LinearClassifier`                   Assumes all classes to have Gaussian distributions with
                                                                                     different means but same covariance matrix and generates
                                                                                     a classifier based on this assumption.
:doxy:`LinearRegression`                  :doxy:`LinearModel`                        Finds the best linear regression model for the labels.
:doxy:`NBClassifierTrainer`               :doxy:`NBClassifier`                       Trains a standard naive-Bayes classifier.
:doxy:`OptimizationTrainer`               all                                        Combines the elements of a given learning problem -- optimizer,
                                                                                     model, error function and stopping criterion -- into a trainer.
:doxy:`Perceptron`                        :doxy:`KernelExpansion`                    Kernelized perceptron -- tries to find a separating hyperplane of
                                                                                     the data in the feature space induced by the kernel.
:doxy:`RFTrainer`                         :doxy:`RFClassifier`                       Implements a random forest of CART trees.
:doxy:`SigmoidFitRpropNLL`                :doxy:`SigmoidModel`                       Optimizes the parameters of a sigmoid to fit a validation
                                                                                     dataset via backpropagation on the negative log-likelihood.
:doxy:`SigmoidFitPlatt`                   :doxy:`SigmoidModel`                       Optimizes the parameters of a sigmoid to fit a validation
                                                                                     dataset with Platt's method.
:doxy:`AbstractSvmTrainer`                :doxy:`KernelExpansion`                    Base class for all support vector machine trainers.
:doxy:`MissingFeatureSvmTrainer`          :doxy:`MissingFeaturesKernelExpansion`     Trainer for binary SVMs supporting missing features.
:doxy:`CSvmTrainer`                       :doxy:`KernelExpansion`                    Trainer for binary SVMs, with one-norm regularization.
:doxy:`EpsilonSvmTrainer`                 :doxy:`KernelExpansion`                    Trains an epsilon-SVM for regression.
:doxy:`RegularizationNetworkTrainer`      :doxy:`KernelExpansion`                    ADD SHORT DESCRIPTION
:doxy:`McSvmOVATrainer`                   :doxy:`KernelExpansion`                    Trains a one-vs-all multiclass SVM.
:doxy:`McSvmCSTrainer`                    :doxy:`KernelExpansion`                    Multiclass SVM as defined by Cramer & Singer
:doxy:`McSvmWWTrainer`                    :doxy:`KernelExpansion`                    Multiclass SVM as defined by Weston & Watkins
:doxy:`McSvmADMTrainer`                   :doxy:`KernelExpansion`                    ADD SHORT DESCRIPTION
:doxy:`McSvmLLWTrainer`                   :doxy:`KernelExpansion`                    Multiclass SVM as defined by Lee, Lin, and Wahba
:doxy:`McSvmMMRTrainer`                   :doxy:`KernelExpansion`                    Multiclass SVM using maximum margin regression.
:doxy:`McSvmATSTrainer`                   :doxy:`KernelExpansion`                    ADD SHORT DESCRIPTION
:doxy:`McSvmATMTrainer`                   :doxy:`KernelExpansion`                    ADD SHORT DESCRIPTION
:doxy:`AbstractLinearSvmTrainer`          :doxy:`LinearModel`                        Base class for all linear-SVM trainers
:doxy:`LinearMcSvmOVATrainer`             :doxy:`LinearModel`                        ADD SHORT DESCRIPTION
:doxy:`LinearMcSvmCSTrainer`              :doxy:`LinearModel`                        ADD SHORT DESCRIPTION
:doxy:`LinearMcSvmWWTrainer`              :doxy:`LinearModel`                        ADD SHORT DESCRIPTION
:doxy:`LinearMcSvmADMTrainer`             :doxy:`LinearModel`                        ADD SHORT DESCRIPTION
:doxy:`LinearMcSvmLLWTrainer`             :doxy:`LinearModel`                        ADD SHORT DESCRIPTION
:doxy:`LinearMcSvmATMTrainer`             :doxy:`LinearModel`                        ADD SHORT DESCRIPTION
:doxy:`LinearMcSvmATSTrainer`             :doxy:`LinearModel`                        ADD SHORT DESCRIPTION
========================================  ========================================   ===================================================================
