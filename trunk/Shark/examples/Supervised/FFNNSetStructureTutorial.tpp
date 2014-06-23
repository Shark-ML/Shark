#include<shark/Data/Dataset.h>
#include<shark/Models/FFNet.h>
#include<shark/Algorithms/GradientDescent/Rprop.h>
#include<shark/ObjectiveFunctions/ErrorFunction.h>
#include<shark/ObjectiveFunctions/Loss/NegativeClassificationLogLikelihood.h>
#include<shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include<shark/Models/FFNet.h>

using namespace shark;
using namespace std;

// define xor benchmark problem
LabeledData<RealVector,unsigned int> xorProblem(){
	//the 2D xor Problem has 4 patterns, (0,0), (0,1), (1,0), (1,1)
	vector<RealVector> inputs(4,RealVector(2));
	//the result is 1 if both inputs have a different value, and 0 otherwise
	vector<unsigned int> labels(4);
	
	unsigned k = 0;
	for(unsigned i=0; i < 2; i++){
		for(unsigned j=0; j < 2; j++){
			inputs[k](0) = i;
			inputs[k](1) = j;
			labels[k] = (i+j) % 2;
			k++;
		}
	}
	LabeledData<RealVector,unsigned int> dataset= createLabeledDataFromRange(inputs,labels);
	return dataset;
}

int main(){
	//create network and initialize weights random uniform
	FFNet<LogisticNeuron,LogisticNeuron> network;

	//first three parameters are number of input, hidden and output neurons
	//the next parameter sets the stucture: we choose here the default structure without any shortcuts
	//the last parameter switches the bias neuron on
	network.setStructure(2, 3, 1, FFNetStructures::Normal,true);

	initRandomUniform(network,-0.1,0.1);
	
	//get problem data
	LabeledData<RealVector,unsigned int> dataset = xorProblem();
	
	//create error function
	NegativeClassificationLogLikelihood loss; // surrogate loss for training
	ErrorFunction<RealVector,unsigned int> error(dataset,&network,&loss);
	
	// loss for evaluation
	ZeroOneLoss<unsigned int, RealVector> loss01(0.5); // classification error, output is thresholded at 1/2

	// evaluate initial network
	Data<RealVector> prediction = network(dataset.inputs());
	cout << "classification error before learning:\t" << loss01.eval(dataset.labels(), prediction) << endl;

	//initialize Rprop
	IRpropPlus optimizer;
	optimizer.init(error);
	unsigned numberOfSteps = 1000;
	for(unsigned step = 0; step != numberOfSteps; ++step) 
		optimizer.step(error);

	// evaluate solution found by training
	network.setParameterVector(optimizer.solution().point); // set weights to weights found by learning
	prediction = network(dataset.inputs());
	cout << "classification error after learning:\t" << loss01.eval(dataset.labels(), prediction) << endl;
}
