//###begin<includes>
#include <shark/Data/Pgm.h> //for exporting the learned filters
#include <shark/Data/SparseData.h>//for reading in the images as sparseData/Libsvm format
#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers
#include <shark/ObjectiveFunctions/ErrorFunction.h> //the error function for minibatch training
#include <shark/Algorithms/GradientDescent/Adam.h>// The Adam optimization algorithm
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h> // squared loss used for regression
#include <shark/ObjectiveFunctions/Regularizer.h> //L2 regulariziation
//###end<includes>

using namespace std;
using namespace shark;

int main(int argc, char **argv)
{	
	if(argc < 2) {
		cerr << "usage: " << argv[0] << " path/to/mnist_subset.libsvm" << endl;
		return 1;
	}
	std::size_t hidden1 = 200;
	std::size_t hidden2 = 100;
	std::size_t iterations = 10000;
	double regularisation = 0.01;
	
	LabeledData<RealVector,unsigned int> data;
	importSparseData( data, argv[1], 784 );
	
	std::size_t numElems = data.numberOfElements();
	for(std::size_t i = 0; i != numElems; ++i){
		for(std::size_t j = 0; j != 784; ++j){
			if(data.element(i).input(j) > 0.5){
				data.element(i).input(j) = 1;
			}else{
				data.element(i).input(j) = 0;
			}
		}
	}
	std::size_t inputs = dataDimension(data.inputs());
	
//###begin<model_creation>
	//We use a dense lienar model with rectifier activations
	typedef LinearModel<RealVector, RectifierNeuron> DenseLayer;
	
	//build encoder network
	DenseLayer encoder1(inputs,hidden1);
	DenseLayer encoder2(encoder1.outputShape(),hidden2);
	auto encoder = encoder1 >> encoder2;
	
	//build decoder network
	DenseLayer decoder1(encoder2.outputShape(), encoder2.inputShape());
	DenseLayer decoder2(encoder1.outputShape(), encoder1.inputShape());
	auto decoder = decoder1 >> decoder2;
	
	//Setup autoencoder model
	auto autoencoder = encoder >> decoder;
//###end<model_creation>
//###begin<objective>		
	//create the objective function as a regression problem
	LabeledData<RealVector,RealVector> trainSet(data.inputs(),data.inputs());//labels identical to inputs
	SquaredLoss<RealVector> loss;
	ErrorFunction<> error(trainSet, &autoencoder, &loss, true);//we enable minibatch learning
	TwoNormRegularizer<> regularizer(error.numberOfVariables());
	error.setRegularizer(regularisation,&regularizer);
	initRandomNormal(autoencoder,0.01);
//###end<objective>	
	//set up optimizer
//###begin<optimizer>
	Adam<> optimizer;
	error.init();
	optimizer.init(error);
	std::cout<<"Optimizing model "<<std::endl;
	for(std::size_t i = 0; i != iterations; ++i){
		optimizer.step(error);
		if(i  % 100 == 0)
			std::cout<<i<<" "<<optimizer.solution().value<<std::endl;
	}
	autoencoder.setParameterVector(optimizer.solution().point);
//###end<optimizer>
//###begin<visualize>
	exportFiltersToPGMGrid("features",encoder1.matrix(),28,28);
//###end<visualize>
}
