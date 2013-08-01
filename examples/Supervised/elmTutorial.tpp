//headers needed for ELM
//###begin<includes>
#include <shark/Models/LinearModel.h>
#include <shark/Models/ConcatenatedModel.h>
#include <shark/Models/FFNet.h>
#include <shark/Rng/Normal.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
#include <shark/Algorithms/Trainers/LinearRegression.h>
//###end<includes>

//header needed for the problem
#include <shark/Data/DataDistribution.h>

//just for evaluation of the ELM
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <iostream>
using namespace std;
using namespace shark;

//In this example an extreme learning machine is constructed and
//trained.  An ELM is a neural net with one single hidden layer. The
//weight of the hidden neurons are random and only the outputs are
//trained.  This makes the learning problem much easier since the
//remaining linear weights form a convex problem.  in principle, the
//ELM can be constructed using a simple FFNet. But this would mean
//that we had to calculate the weight vector for the FFNet - which is
//not trivial. Instead we will construct the ELM out of one FFNet and
//two linear networks. That's a bit slower but usually not a problem

///Our problem:
///z = sin(x)/x+y+noise
/// @cond EXAMPLE_SYMBOLS
class Problem:public LabeledDataDistribution<RealVector,RealVector>{
public:
    void draw(RealVector& input, RealVector& label)const
	{
		input.resize(2);
		label.resize(1);
		input(0) = Rng::uni(-5, 5);
		input(1) = Rng::uni(-5, 5);
		if(input(0) != 0)
			label(0) = sin(input(0)) / input(0) + input(1) + Rng::gauss(0.0, 0.1);
		else
			label(0) = 1 + input(1) + Rng::gauss(0.0, 0.1);
	}
};
/// @endcond

int main(){
	//change these constants for your own problem
	size_t hiddenNeurons = 17;
	size_t numSamples = 1000;
	unsigned int randomSeed = 42;
	
	//configure random number generator
	Rng::seed(randomSeed);
	
	//create the regression problem
	Problem problem;
	RegressionDataset data = problem.generateDataset(numSamples);
	size_t inputDim = inputDimension(data);
	
	//usually an elm uses zero mean unit variance inputs. so we should
	//normalize the data first
	//###begin<normalization>
	Normalizer<> normalizer;
	NormalizeComponentsUnitVariance<> normalizingTrainer(true);
	normalizingTrainer.train(normalizer,data.inputs());
	//###end<normalization>
	
	//now we construct the hidden layer of the elm
	//we use an FFNet with no hidden layer. Instead of hiddens, we abuse the output neurons
	//for our purposes. This setup also uses bias neurons for the hidden layer
	//###begin<FFNetStructure>
	FFNet<LogisticNeuron,LogisticNeuron> elmHidden;
	elmHidden.setStructure(inputDim,0,hiddenNeurons);
	initRandomUniform(elmHidden,0,1);
	//###end<FFNetStructure>
	
	//now we concatenate the hidden units with the normalizer and get the
	//full hidden layer
	//###begin<normalizer_concatenate>
	ConcatenatedModel<RealVector,RealVector> elm = normalizer >> elmHidden;
	//###end<normalizer_concatenate>
	
	//We need to train the linear part. in this simple example we use the elm standard
	//technique: linear regression. For this we need to propagate the data first 
	// through the hidden layer.
	//###begin<train>
	RegressionDataset transformedData=transformInputs(data,elm);
	LinearModel<> elmOutput;
	LinearRegression trainer;
	trainer.train(elmOutput,transformedData);
	//###end<train>

	//finally we can construct the complete elm which can than be used for evaluation
	//###begin<elm>
	elm = normalizer >> elmHidden >> elmOutput;
	//###end<elm>
	//to test whether everything works, we will evaluate the elm and the elmOutput layer separately
	//both results should be identical
	SquaredLoss<> loss;
	double outputResult = loss(transformedData.labels(),elmOutput(transformedData.inputs()));
	double elmResult = loss(transformedData.labels(),elm(data.inputs()));

	cout<<"Results"<<std::endl;
	cout<<"============"<<std::endl;
	cout<<"output Layer: "<< outputResult<<std::endl;
	cout<<"ELM: "<< elmResult<<std::endl;
}
