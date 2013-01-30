==========================================
Random Forest
==========================================

Background
----------

Random Forest is a decision tree algorithm for both classification and
regression. It was first described by [Breiman2001]_. It is a recursive
algorithm which partitions the training dataset by doing binary splits like CART.
It builds an ensemble of classifiers, and uses randomization to reduce the variance
between the trees of the ensemble. It performs remarkable well in practice, and 
is widely used.


Random Forest in Shark
----------------------------------------

Sample classification problem: Protein fold prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us consider the same bioinformatics problem as in the
:doc:`nearestNeighbor`, :doc:`lda` and :doc:`cart` tutorial, namely the prediction of the
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
  dataTrain.pack();
  dataTest.pack();
  
Model and learning algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We include the :doxy:`RFTrainer` class ::

  #include <shark/Algorithms/Trainers/RFTrainer.h>

and define the trainer :: 

  RFTrainer trainer;

and the Random Forest model to be trained by the corresponding trainer: ::

  RFClassifier model;

The model is trained by calling: ::

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
	cout << "RF on training set accuracy: " << 1. - loss.eval(dataTrain.labels(), prediction) << endl;
	prediction = model.eval(dataTest.inputs());
	cout << "RF on test set accuracy:     " << 1. - loss.eval(dataTest.labels(), prediction) << endl;

Of course, the accuracy is given by one minus the error.
In this example, Random Forest outperforms LDA, KNN, and CART. 


Parameters of the trainer
^^^^^^^^^^^^^^^^^^^^^^^^^

The trainer has some properties that can be set to tweak the learning process. 
All parameters have meaningful default values. The parameters are set by the 
following methods: ::

    trainer.setMTry(size_t mtry);

MTry controls the number of random attribute to try at each inner node of each tree. ::

    trainer.setNTrees(size_t nTrees)

NTrees, controls the number of trees to be built. Typically this would be 100+. ::

    trainer.setNodeSize(size_t nodeSize)

NodeSize, controls the maximum nodesize, before a node is classified as a leaf. Lowering this
value, makes the trees in the ensemble larger, and increasing this value, makes the trees smaller. ::

    trainer.setOOBratio(double ratio)

OOBRatio controls the ratio determining  the number of OOB (out-of-bag) samples is sampled from the training dataset.

Full example program
--------------------

The full example program is 
:doxy:`RFTutorial.cpp`.



References
----------

.. [Breiman2001] L. Breiman.
            Random Forests.
            Machine Learning, vol. 45, issue 1, p. 5-32,
            2001
            

