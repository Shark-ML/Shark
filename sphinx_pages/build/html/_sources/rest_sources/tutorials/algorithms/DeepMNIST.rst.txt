Data Shapes and Deep Convolutional Architectures
========================================================
In this tutorial, we will train a deep convolutional neural network on the MNIST
dataset.This tutorial is an adaptation of the tensorflow 1.4 
`Deep MNIST for Experts <https://www.tensorflow.org/versions/r1.4/get_started/mnist/pros>`_ tutorial
and the whole code file can be found :doxy:`here<MNISTForExperts.cpp>`.
In this tutorial, we want to use a deep neural network architecture based on convolutions.
For this, we have to introduce a new concept: :doxy:`Shapes <Shape>`. All of the computations are performed in floating point precision
by using ``FloatVector`` instead of ``RealVector`` as template arguments.

For this tutorial, we need the following includes::


	#include <shark/Data/SparseData.h>//for reading in the images as sparseData/Libsvm format
	#include <shark/Models/LinearModel.h>//single dense layer
	#include <shark/Models/ConvolutionalModel.h>//single convolutional layer
	#include <shark/Models/PoolingLayer.h> //pooling after convolution
	#include <shark/Models/ConcatenatedModel.h>//for stacking layers
	#include <shark/Algorithms/GradientDescent/Adam.h>// The Adam optimization algorithm
	#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h> //classification loss
	#include <shark/ObjectiveFunctions/ErrorFunction.h> //Error function for optimization
	#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> //evaluation for testing
	using namespace shark;
	

Shapes
------------------
In shark, most input data is converted to vector or scalar form. For example an 28x28 image is encoded as a vector with 784 entries. Its class
label is a single integral number. This
means that the model taking the image as input has no information about the image size or the number of classes.
Further, if the input is a convolutional model, its output may have a different number of rows, columns and channels. Keeping
track of change of dimensionality is an errorprone task which should be avoided.
Therefore, in this tutorial we will introduce the concept of a :doxy:`Shape`. A shape adds information about the input or output types
in a vectorial format. For example, images have a 3-d shape: ``height x width x channels``. The class label has a 1-d shape: the number
of classes. The batchsize is not stored in the shape but separately in each batch. 
By default, when loading a dataset from a file, we have to adapt the shape to its actual dimensionalities::


		LabeledData<FloatVector,unsigned int> data;
		importSparseData( data, argv[1] , 784 , 100);
		std::cout<<"input shape:"<< data.inputShape()<<std::endl;
		std::cout<<"output shape:"<< data.labelShape()<<std::endl;
		data.inputShape() = {28,28,1}; //store shape for model creation
		std::cout<<"input shape:"<< data.inputShape()<<std::endl;
	

When loading the MNIST dataset, this prints

.. code-block:: none

	input shape:(784)
	output shape:(10)
	input shape:(28, 28, 1)
		
Most models in shark do not care about the exact shape of objects they get as input or return as outputs. 
For example a linear model can be initialized to return a vector with shape (2352) or a multi-channel image (28,28,3).
Similarly when a model returns an image (28,28,3) as output shape, the next model can interpret it as (2352), as long as dimensions match, this
is not a problem.


Deep Convolutional Neural Networks
------------------------------------------

Lets create our neural network!
We choose the layers of the neural networks as:

* :doxy:`Convolutional layer<Conv2DModel>` with 32 filters of size 5x5 and ReLu :doxy:`activation function<activations>`
* :doxy:`Max-pooling<PoolingLayer>` of size 2x2
* Convolutional layer with 64 filters of size 5x5 and ReLu activation function
* Max-pooling of size 2x2
* A :doxy:`Dense-layer<LinearModel>` with 1024 output neurons, bias term and ReLu activation function
* A final classification layer with one neuron for each class, bias term and linear activation function

In shark, this is implemented as::


		Conv2DModel<FloatVector, RectifierNeuron> conv1(data.inputShape(), {32, 5, 5});
		PoolingLayer<FloatVector> pooling1(conv1.outputShape(), {2, 2}, Pooling::Maximum, Padding::Valid);
		Conv2DModel<FloatVector, RectifierNeuron> conv2(pooling1.outputShape(), {64, 5, 5});
		PoolingLayer<FloatVector> pooling2(conv2.outputShape(), {2, 2}, Pooling::Maximum, Padding::Valid);
		LinearModel<FloatVector, RectifierNeuron> dense1(pooling2.outputShape(), 1024, true);
		LinearModel<FloatVector> dense2(dense1.outputShape(), data.labelShape(), true);
		auto model = conv1 >> pooling1 >> conv2 >> pooling2 >> dense1 >> dense2;
	


In the next step, we define our loss and error function for optimization. Again, we use minibatch learning with the batchsize 100 defined
during loading of the dataset::


		CrossEntropy<unsigned int, FloatVector> loss;
		ErrorFunction<FloatVector> error(data, &model, &loss, true);
	

And optimize it using the Adam algorithm::


		std::size_t iterations = 20001;
		initRandomNormal(model,0.0001); //init model
		Adam<FloatVector> optimizer;
		optimizer.setEta(0.0001);//learning rate of the algorithm
		error.init();
		optimizer.init(error);
		std::cout<<"Optimizing model "<<std::endl;
		for(std::size_t i = 0; i != iterations; ++i){
			optimizer.step(error);
			if(i  % 100 == 0){//print out timing information and training error
				ZeroOneLoss<unsigned int, FloatVector> classificationLoss;
				double error = classificationLoss(data.labels(),model(data.inputs()));
				std::cout<<i<<" "<<optimizer.solution().value<<" "<<error<<std::endl;
			}
		}
	

What next?
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The next tutorials covers :doc:`autoencoders`