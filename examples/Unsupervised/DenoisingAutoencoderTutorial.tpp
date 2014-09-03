
//###begin<includes>
#include <shark/Data/Pgm.h> //for exporting the learned filters
#include <shark/Data/SparseData.h>//for reading in the images as sparseData/Libsvm format
#include <shark/Models/Autoencoder.h>//normal autoencoder model
#include <shark/Models/TiedAutoencoder.h>//autoencoder with tied weights
#include <shark/Models/ImpulseNoiseModel.h>//noise source to corrupt the inputs
#include <shark/Models/ConcatenatedModel.h>//to concatenate the noise with the model
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>// the Rprop optimization algorithm
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
	double regularisation,//strength of the regularisation
	double noiseStrength // strength of the added noise
){
//###end<function>
//###begin<model>	
	//create the model
	std::size_t inputs = dataDimension(data);
	AutoencoderModel baseModel;
	baseModel.setStructure(inputs, numHidden);
	initRandomUniform(baseModel,-0.1*std::sqrt(1.0/inputs),0.1*std::sqrt(1.0/inputs));
	ImpulseNoiseModel noise(noiseStrength,0.0);//set an input pixel with probability p to 0
	ConcatenatedModel<RealVector,RealVector> model = noise>> baseModel;
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
	return baseModel;
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
	double noiseStrengt = 0.5;
	
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

	Autoencoder1 net1 = trainAutoencoderModel<Autoencoder1>(train.inputs(),numHidden,iterations,regularisation,noiseStrengt);
	Autoencoder2 net2 = trainAutoencoderModel<Autoencoder2>(train.inputs(),numHidden,iterations,regularisation,noiseStrengt);

	exportFiltersToPGMGrid("features1",net1.encoderMatrix(),28,28);
	exportFiltersToPGMGrid("features2",net2.encoderMatrix(),28,28);
	//###end<main>
}
