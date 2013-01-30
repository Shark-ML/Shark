Extreme Learning Machine
========================

During the last few years, the Extreme Learning Machine (ELM) got a lot of attention.
Shark does not have a class tailored to ELMs but supports all building blocks to create one.
If you need the ELM very often it might be advisable to put the following code blocks inside
a class so that you can use it easily.

In theory an ELM is not more than a simple Feed Forward neural network where the hidden
weights are not trained. It is possible to put all the weights into the :doxy:`FFNet` of Shark,
but it is not straightforward to create the parameter vector for the FFNet. Instead we use an
easier construction for the ELM in this tutorial which concatenates several models to the
ELM.

For the ELM, we need the following include files: ::

  #include <shark/Models/LinearModel.h>
  #include <shark/Models/ConcatenatedModel.h>
  #include <shark/Models/FFNet.h>
  #include <shark/Rng/Normal.h>
  #include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
  #include <shark/Algorithms/Trainers/LinearRegression.h>
  
To prevent that single inputs with high variance dominate the behaviour of the ELM, the inputs are required to 
have zero mean and unit variance. So we need a model which applies this transformation to the data. 
This is a linear transformation which can be learned by an affine linear model using the proper trainer ::

  Normalizer<> normalizer(true);
  NormalizeComponentsUnitVariance<> normalizingTrainer;
  normalizingTrainer.train(normalizer,data.inputs());
  
Now we can create the hidden layer of the ELM. For this we abuse the :doxy:`FFNet`. We define it to have no hidden 
neurons, so the weights between input and sigmoidal output neurons are the random weights of the ELM. 
Next we call :doxy:`initRandomUniform` to initialize the ELM with uniform weights. You should never forget 
to initialize the seed for the random number generator befor using it! ::

  Rng::seed(42);
  FFNet<LogisticNeuron,LogisticNeuron> elmHidden;
  elmHidden.setStructure(inputDim,0,hiddenNeurons);
  initRandomUniform(elmHidden,0,1);

Next we must concatenate the normalizer with the hidden units ::

  ConcatenatedModel<RealVector,RealVector> elm = normalizer >> elmHidden;

Now that we have the hidden Layer of the ELM, we can train the output layer. Since the output weights of the ELM
are linear, we can use linear regression to train them. The best way to do is, is to propagate the training data
forward through the hidden layer and use the outputs as inputdata for the linear regression ::

  RegressionDataset transformedData(elm(data.inputs()),data.labels());
  LinearModel<> elmOutput;
  LinearRegression trainer;
  trainer.train(elmOutput, transformedData);
  
After training is done, the last thing we have to do is concatenate the hidden units with the outputs and present
our shiny new elm to the world! ::
  
  elm = normalizer >> elmHidden >> elmOutput;
  
The downside of this technique is, that in the end  4 objects are needed for a working ELM: the normalizer, 
the hidden units, the output networks and the concatenated model which put the pieces together. So it
is advisable to put these parts into a class. Of course putting all weights together into a single FFNet is
faster, but is also not straight forward and error prone. The difference in speed can be ignored for most 
applications.

Full example program
--------------------

The full example program is  :doxy:`elmTutorial.cpp`.
