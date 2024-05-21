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

List of Classes
---------------------------------
* :doxy:`Supervised Trainers<supervised_trainer>`,
* :doxy:`Unsupervised Trainers<unsupervised_trainer>`



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
``DatasetType``              The type of dataset that the trainer expects,
			     this is `LabeledData<InputType, LabelType>`
==========================   =======================================


A trainer offers the following methods:


=========================================================   =================================================
Method                                                      Description
=========================================================   =================================================
``train(ModelType&, LabeledData<InputType, LabelType>)``    Solves the problem and sets the model parameters
``std::string name()``                                      Returns the trainer's name
=========================================================   =================================================


Usage of trainers is equally straightforward::

  MyModel model;
  MyTrainer trainer;
  MyDataset data;
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
``DatasetType``              The type of dataset that the trainer expects,
			     this is `UnlabeledData<InputType>`
==========================   ==============================================


These trainers also offer the following methods:


=====================================================   ================================================
Method                                                  Description
=====================================================   ================================================
``train(ModelType&, UnlabeledData<InputType>)``         Solves the problem and stores it in the model.
``std::string name()``                                  Returns the name of the trainer
=====================================================   ================================================


Weighting
----------------------------------------------------

Both variants of trainers also come in a version with support for weighting,
:doxy:`AbstractWeightedTrainer` and :doxy:`AbstractWeightedUnsupervisedTrainer`. Both support an additional
verion of train which can use a :doxy:`WeightedUnlabeledData` or :doxy:`WeightedLabeledData` object which adds
an additional weighting factor for each element. This factor must be positive, but does not need to sum to one.
Using a weighted dataset with all weights equal will lead to the same result as not weighting the dataset.

Both classes declare a new typedef

==========================   ==============================================
Types                        Description
==========================   ==============================================
``WeightedDatasetType``      The weighted dataset that the trainer expects,
			     this is `WeightedUnlabeledData<InputType>`
			     or `WeightedLabeledData<InputType,LabeledType>`
==========================   ==============================================

and the following version of train, additionally to the known versions:

=====================================================   ================================================
Method                                                  Description
=====================================================   ================================================
``train(ModelType&, WeightedDatasetType)``              Solves the weighted problem and stores it in the model.
=====================================================   ================================================
