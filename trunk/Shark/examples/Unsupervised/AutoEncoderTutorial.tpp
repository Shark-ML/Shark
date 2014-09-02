
//###begin<includes>
#include <shark/Data/Pgm.h> //for exporting the learned filters
#include <shark/Data/SparseData.h>//for reading in the images as sparseData/Libsvm format
#include <shark/Models/Autoencoder.h>//normal autoencoder model
#include <shark/Models/TiedAutoencoder.h>//autoencoder with tied weights
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>// the RProp optimization algorithm
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h> // squared loss used for regression
#include <shark/ObjectiveFunctions/Regularizer.h> //L2 regulariziation
//###end<includes>

using namespace std;
using namespace shark;

//training of an auto encoder with one hidden layer
//###begin<function>
template<class AutoencoderModel>
AutoencoderModel trainAutoencoderModel(
	UnlabeledData<RealVector> const& data,//the data to train with
	std::size_t numHidden,//number of features in the autoencoder
	std::size_t iterations, //number of iterations to optimize
	double regularisation//strength of the regularisation
){
//###end<function>
	
	
	
//###begin<model>	
	//create the model
	std::size_t inputs = dataDimension(data);
	AutoencoderModel model;
	model.setStructure(inputs, numHidden);
	initRandomUniform(model,-0.1*std::sqrt(1.0/inputs),0.1*std::sqrt(1.0/inputs));
//###end<model>	

//###begin<objective>		
	//create the objective function
	LabeledData<RealVector,RealVector> trainSet(data,data);//labels identical to inputs
	SquaredLoss<RealVector> loss;
	ErrorFunction<RealVector,RealVector> error(trainSet, &model, &loss);
	TwoNormRegularizer regularizer(error.numberOfVariables());
	error.setRegularizer(regularisation,&regularizer);
//###end<objective>	
	//set up optimizer
//###begin<optimizer>
	IRpropPlusFull optimizer;
	optimizer.init(error);
	std::cout<<"Optimizing model: "+model.name()<<std::endl;
	for(std::size_t i = 0; i != iterations; ++i){
		optimizer.step(error);
		std::cout<<i<<" "<<optimizer.solution().value<<std::endl;
	}
//###end<optimizer>
	model.setParameterVector(optimizer.solution().point);
	return model;
	
}
int main(int argc, char **argv)
{	
	if(argc < 2) {
		cerr << "usage: " << argv[0] << " path/to/mnist_subset.libsvm" << endl;
		return 1;
	}
	std::size_t numHidden = 200;
	std::size_t iterations = 200;
	double regularisation = 0.01;
	
	LabeledData<RealVector,unsigned int> train;
	importSparseData( train, argv[1] );
	
	std::size_t numElems = train.numberOfElements();
	for(std::size_t i = 0; i != numElems; ++i){
		for(std::size_t j = 0; j != 784; ++j){
			if(train.element(i).input(j) > 0.5){
				train.element(i).input(j) = 1;
			}else{
				train.element(i).input(j) = 0;
			}
		}
	}
	
	//###begin<main>
	typedef Autoencoder<LogisticNeuron, LogisticNeuron> Autoencoder1;
	typedef TiedAutoencoder<LogisticNeuron, LogisticNeuron> Autoencoder2;
	typedef Autoencoder<DropoutNeuron<LogisticNeuron>, LogisticNeuron> Autoencoder3;
	typedef TiedAutoencoder<DropoutNeuron<LogisticNeuron>, LogisticNeuron> Autoencoder4;
	
	Autoencoder1 net1 = trainAutoencoderModel<Autoencoder1>(train.inputs(),numHidden,iterations,regularisation);
	Autoencoder2 net2 = trainAutoencoderModel<Autoencoder2>(train.inputs(),numHidden,iterations,regularisation);
	Autoencoder3 net3 = trainAutoencoderModel<Autoencoder3>(train.inputs(),numHidden,iterations,regularisation);
	Autoencoder3 net4 = trainAutoencoderModel<Autoencoder3>(train.inputs(),numHidden,iterations,regularisation);

	exportFiltersToPGMGrid("features1",net1.encoderMatrix(),28,28);
	exportFiltersToPGMGrid("features2",net2.encoderMatrix(),28,28);
	exportFiltersToPGMGrid("features3",net3.encoderMatrix(),28,28);
	exportFiltersToPGMGrid("features4",net4.encoderMatrix(),28,28);
	//###end<main>
}
