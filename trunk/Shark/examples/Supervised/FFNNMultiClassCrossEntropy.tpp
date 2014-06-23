#include<shark/Data/Dataset.h>
#include<shark/Data/DataDistribution.h>
#include<shark/Models/FFNet.h>
#include<shark/Algorithms/GradientDescent/Rprop.h>
#include<shark/ObjectiveFunctions/ErrorFunction.h>
#include<shark/ObjectiveFunctions/Loss/CrossEntropy.h>
#include<shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include<shark/Models/FFNet.h>

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
	LabeledData<RealVector,unsigned int> test = problem.generateDataset(100);
	
	std::size_t inputs=inputDimension(training);
	std::size_t outputs = numberOfClasses(training);
	std::size_t hiddens = 10;
	unsigned numberOfSteps = 1000;

	//create network and initialize weights random uniform
	FFNet<LogisticNeuron,LinearNeuron> network;
	network.setStructure(inputs,hiddens,outputs);
	initRandomUniform(network,-0.1,0.1);
	
	//create error function
	CrossEntropy loss;
	ErrorFunction<RealVector,unsigned int> error(training,&network,&loss);
	
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
	network.setParameterVector(optimizer.solution().point); // set weights to weights found by learning
	prediction = network(training.inputs());
	cout << "classification error after learning:\t" << loss01(training.labels(), prediction) << endl;
}
