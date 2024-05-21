==========================================
Random Forest
==========================================

Background
----------

Random Forest is a decision tree algorithm for both classification and
regression. It was first described by [Breiman2001]_. It is a recursive
algorithm which partitions the training dataset by doing binary splits based on the labels of the data.
It builds an ensemble of models, and uses randomization to increase the variance
between the trees of the ensemble. It performs remarkable well in practice, and 
is widely used.


Random Forest in Shark
--------------------------------------

For this tutorial, the following includes are needed::


	#include <shark/Data/Csv.h> //importing the file
	#include <shark/Algorithms/Trainers/RFTrainer.h> //the random forest trainer
	#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> //zero one loss for evaluation
	

Sample classification problem: Protein fold prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us consider the same bioinformatics problem as in the
:doc:`nearestNeighbor` and :doc:`lda` tutorial, namely the prediction of the
secondary structure of proteins. The goal is to assign a protein to
one out of 27 SCOP fold types [DingDubchak2001a]_.  We again consider
the descriptions of amino-acid sequences provided by
[DamoulasGirolami2008a]_.  The data :download:`C.csv <../../../../../examples/Supervised/data/C.csv>`
provide a description of the amino-acid compositions of 695 proteins
together with the corresponding fold type. Each row corresponds to a
protein.  After downloading the data :download:`C.csv <../../../../../examples/Supervised/data/C.csv>` we
can read, inspect and split the data as described in the
:doc:`nearestNeighbor` tutorial::


		ClassificationDataset data;
		importCSV(data, "data/C.csv", LAST_COLUMN, ' ');
	
		//Split the dataset into a training and a test dataset
		ClassificationDataset dataTest = splitAtElement(data,311);
		
  
Model and learning algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can use the :doxy:`RFTrainer` class to train a :doxy:`RFClassifier` which represents a Random Forest::


		RFTrainer<unsigned int> trainer;
		RFClassifier<unsigned int> model;
		trainer.train(model, data);
		

Evaluating the model
^^^^^^^^^^^^^^^^^^^^

After training the model we can evaluate it.  As a performance
measure, we consider the standard 0-1 loss.  The corresponding risk is
the probability of error, the empirical risk is the average
classification error.  When measured on set of sample patterns, it
simply computes the fraction of wrong predictions::


		ZeroOneLoss<> loss;
		auto prediction = model(data.inputs());
		cout << "Random Forest on training set accuracy: " << 1. - loss.eval(data.labels(), prediction) << endl;
	
		prediction = model(dataTest.inputs());
		cout << "Random Forest on test set accuracy:     " << 1. - loss.eval(dataTest.labels(), prediction) << endl;
		

Of course, the accuracy is given by one minus the error.
In this example, Random Forest outperforms LDA and KNN. 


Parameters of the trainer
^^^^^^^^^^^^^^^^^^^^^^^^^

The trainer has some properties that can be set to tweak the learning process. 
All parameters have meaningful default values. The parameters are set by the 
following methods:

======================================= =====================================
Method					Effect
=======================================	=====================================
``setMTry(size_t mtry)``   		Number of random attribute to try at 
					each inner node of each tree.
``setNTrees(size_t nTrees)``		Number of trees to be built. 
					Typically this would be 100+.
``setNodeSize(size_t nodeSize)``	The maximum nodesize, before a node 
					is classified as a leaf. Lowering this
					value makes the trees in the ensemble 
					larger and increasing this value 
					makes the trees smaller.
``setOOBratio(double ratio)``		Controls the ratio determining 
					the number of OOB (out-of-bag)
					samples is sampled from the training 
					dataset.
=======================================	=====================================				

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
            

