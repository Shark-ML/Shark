//###begin<includes>
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
//###end<includes>

int main(int argc, char **argv)
{
	if(argc < 2) {
		std::cerr << "usage: " << argv[0] << " path/to/mnist_subset.libsvm" << std::endl;
		return 1;
	}
	
	//Step1: load data, adapt shapes
//###begin<data>
	LabeledData<FloatVector,unsigned int> data;
	importSparseData( data, argv[1] , 784 , 100);
	std::cout<<"input shape:"<< data.inputShape()<<std::endl;
	std::cout<<"output shape:"<< data.labelShape()<<std::endl;
	data.inputShape() = {28,28,1}; //store shape for model creation
	std::cout<<"input shape:"<< data.inputShape()<<std::endl;
//###end<data>

	//Step 2: define model
//###begin<model_creation>
	Conv2DModel<FloatVector, RectifierNeuron> conv1(data.inputShape(), {32, 5, 5});
	PoolingLayer<FloatVector> pooling1(conv1.outputShape(), {2, 2}, Pooling::Maximum, Padding::Valid);
	Conv2DModel<FloatVector, RectifierNeuron> conv2(pooling1.outputShape(), {64, 5, 5});
	PoolingLayer<FloatVector> pooling2(conv2.outputShape(), {2, 2}, Pooling::Maximum, Padding::Valid);
	LinearModel<FloatVector, RectifierNeuron> dense1(pooling2.outputShape(), 1024, true);
	LinearModel<FloatVector> dense2(dense1.outputShape(), data.labelShape(), true);
	auto model = conv1 >> pooling1 >> conv2 >> pooling2 >> dense1 >> dense2;
//###end<model_creation>
	
//###begin<objfunct>	
	CrossEntropy<unsigned int, FloatVector> loss;
	ErrorFunction<FloatVector> error(data, &model, &loss, true);
//###end<objfunct>	
	
	//Step 4 set up optimizer and run optimization
//###begin<training>	
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
//###end<training>
}

