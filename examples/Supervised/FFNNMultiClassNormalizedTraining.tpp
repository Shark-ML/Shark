#include<shark/Data/Dataset.h>
#include<shark/Data/DataDistribution.h>
#include<shark/Models/FFNet.h>
#include<shark/Models/Softmax.h>
#include<shark/Models/ConcatenatedModel.h>
#include<shark/Algorithms/GradientDescent/Rprop.h>
#include<shark/ObjectiveFunctions/ErrorFunction.h>
#include<shark/ObjectiveFunctions/Loss/NegativeClassificationLogLikelihood.h>
#include<shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

using namespace shark;
using namespace std;

// data generating distribution for our toy
// multi-category classification problem
/// @cond EXAMPLE_SYMBOLS
class Problem : public LabeledDataDistribution<RealVector, unsigned int>
{
private:
	double m_noise;
public:
	Problem(double noise):m_noise(noise){}
	void draw(RealVector& input, unsigned int& label)const
	{
		label = Rng::discrete(0, 4);
		input.resize(2);
		input(0) = m_noise * Rng::gauss() + 3.0 * std::cos((double)label);
		input(1) = m_noise * Rng::gauss() + 3.0 * std::sin((double)label);
	}
};
/// @endcond

int main(){
	//get problem data
	Problem problem(1.0);
	LabeledData<RealVector,unsigned int> training = problem.generateDataset(1000);
	
	std::size_t inputs=inputDimension(training);
	std::size_t outputs = numberOfClasses(training);
	std::size_t hiddens = 10;
	unsigned numberOfSteps = 1000;

	//create network
	//to train the network with the NegativeClassificationLogLikelihood, we need to
	//normalize the sum of the outputs to 1. Here we use Softmax
	//which first calculates the exponential of the neuron activation and than normalises this
	//this together with linear outputs of the networks leads in conjunction with the 
	//NegativeClassificationLogLikelihood to a well known loss function which is also
	//known as CrossEntropy. 
	//This is also implemented in shark and the same example using the CrossEntropy can be
	//seen in FFNNMultiClassCrossEntropy.cpp. The implementation of CrossEntropy is numerically
	//more stable than this and should be used. However this example shows an easy way to implement 
	//such or similar normalisations for arbitrary error functions.
	FFNet<LogisticNeuron,LinearNeuron> network;
	Softmax normalisation(outputs);
	//concatenate both models so that the output of network is normalised
	ConcatenatedModel<RealVector,RealVector> model = network >>normalisation;
	
	//set model structure and initialize
	network.setStructure(inputs,hiddens,outputs);
	initRandomUniform(model,-0.1,0.1);
	
	//create error function
	NegativeClassificationLogLikelihood loss;
	ErrorFunction<RealVector,unsigned int> error(training,&model,&loss);
	
	// loss for evaluation
	// The zeroOneLoss for multiclass problems assigns the class to the highest output
	ZeroOneLoss<unsigned int, RealVector> loss01; 

	// evaluate initial network
	Data<RealVector> prediction = network(training.inputs());
	cout << "classification error before learning:\t" << loss01.eval(training.labels(), prediction) << endl;

	//initialize Rprop
	IRpropPlus optimizer;
	optimizer.init(error);
	
	for(unsigned step = 0; step != numberOfSteps; ++step) 
		optimizer.step(error);

	// evaluate solution found by training
	model.setParameterVector(optimizer.solution().point); // set weights to weights found by learning
	prediction = network(training.inputs());
	cout << "classification error after learning:\t" << loss01.eval(training.labels(), prediction) << endl;
}
