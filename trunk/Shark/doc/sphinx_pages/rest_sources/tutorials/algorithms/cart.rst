==========================================
Classification And Regression Trees (CART)
==========================================

Background
----------

CART is a decision tree algorithm for both classification and
regression. It was first described by [Breiman1984]_. It is a recursive
algorithm, which partitions the training dataset by doing binary splits.
It is a conceptual simple decision tree algorithm, and performs 
OK in practice.

CART in Shark
----------------------------------------

Sample classification problem: Protein fold prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us consider the same bioinformatics problem as in the
:doc:`nearestNeighbor` tutorial, namely the prediction of the
secondary structure of proteins. The goal is to assign a protein to
one out of 27 SCOP fold types [DingDubchak2001a]_.  We again consider
the descriptions of amino-acid sequences provided by
[DamoulasGirolami2008a]_.  The data :download:`C.csv <../../../../../examples/Supervised/data/C.csv>`
provide a description of the amino-acid compositions of 695 proteins
together with the corresponding fold type. Each row corresponds to a
protein.  After downloading the data :download:`C.csv <../../../../../examples/Supervised/data/C.csv>` we
can read, inspect and split the data as described in the
:doc:`nearestNeighbor` tutorial: ::

  #include <shark/Data/Csv.h>
  #include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

  using namespace shark;
  using namespace std;


  // read data
  ClassificationDataset dataTrain, dataTest, data;
  import_csv(data, "C.csv", LAST_COLUMN, " ", "#");

  cout << "number of data points: " << data.size() << " " 
       << "number of classes: " << numberOfClasses(data) << " " 
       << "input dimension: " << inputDimension(data) << endl;

  // split data into training and test set
  data.rangeSubset(311, dataTrain, dataTest);

Model and learning algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We include the :doxy:`CARTTrainer` class ::

  #include <shark/Algorithms/Trainers/CARTTrainer.h>

and define the trainer :: 

  CARTTrainer trainer;

and the CART model to be trained by the CART trainer: ::

  CARTClassifier model;

The model is built by calling: ::

    trainer.train(model, dataTrain);

Evaluating the model
^^^^^^^^^^^^^^^^^^^^

After training the model we can evaluate it.  As a performance
measure, we consider the standard 0-1 loss.  The corresponding risk is
the probability of error, the empirical risk is the average
classification error.  When measured on set of sample patterns, it
simply computes the fraction of wrong predictions.
We define the loss for ``unsigned integer`` labels and
create a new data container for the predictions of our model: ::

	Data<unsigned int> prediction;
	ZeroOneLoss<unsigned int, RealVector> loss;

	
Let's apply the classifier to the training and the test data: ::

	prediction = model.eval(dataTrain.inputs());
	cout << "CART on training set accuracy: " << 1. - loss.eval(dataTrain.labels(), prediction) << endl;
	prediction = model.eval(dataTest.inputs());
	cout << "CART on test set accuracy:     " << 1. - loss.eval(dataTest.labels(), prediction) << endl;

Of course, the accuracy is given by one minus the error.
In this example, CART performs slightly better than the LDA and much
better than random guessing.

Full example program
--------------------

The full example program is 
:doxy:`CARTTutorial.cpp`.



References
----------

.. [Breiman1984] L. Breiman, J. H. Friedman, R. A. Olshen, C. J.  Stone.
            Classification and Regression Trees.
            Wadsworth and Brooks,  
            1984
