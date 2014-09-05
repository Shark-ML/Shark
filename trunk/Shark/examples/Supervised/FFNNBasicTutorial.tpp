//###begin<includes>
#include<shark/Models/FFNet.h> //the feed forward neural network
#include<shark/Algorithms/GradientDescent/Rprop.h> //resilient propagation as optimizer
#include<shark/ObjectiveFunctions/Loss/CrossEntropy.h> // loss during training
#include<shark/ObjectiveFunctions/ErrorFunction.h> //error function to connect data model and loss
#include<shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> //loss for test performance

//evaluating probabilities
#include<shark/Models/Softmax.h> //transforms model output into probabilities  
#include<shark/Models/ConcatenatedModel.h> //provides operator >> for concatenating models
//###end<includes>

using namespace shark;
using namespace std;

// define xor benchmark problem
//###begin<problem>
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
	LabeledData<RealVector,unsigned int> dataset = createLabeledDataFromRange(inputs,labels);
	return dataset;
}
//###end<problem>
int main(){
	//create network and initialize weights random uniform
	//###begin<network_topology>
	unsigned numInput=2;
	unsigned numHidden=4;
	unsigned numOutput=1;
	FFNet<LogisticNeuron,LinearNeuron> network;
	network.setStructure(numInput,numHidden,numOutput,FFNetStructures::Normal,true);
	//###end<network_topology>
	
	//###begin<train>
	//get problem data
	LabeledData<RealVector,unsigned int> dataset = xorProblem();
	
	//create error function
	CrossEntropy loss; // surrogate loss for training
	ErrorFunction<RealVector,unsigned int> error(dataset,&network,&loss);
	
	//initialize Rprop and initialize the network randomly
	initRandomUniform(network,-0.1,0.1);
	IRpropPlus optimizer;
	optimizer.init(error);
	unsigned numberOfSteps = 1000;
	for(unsigned step = 0; step != numberOfSteps; ++step) 
		optimizer.step(error);
		
	//###end<train>

	// loss for evaluation
	ZeroOneLoss<unsigned int, RealVector> loss01(0.5); // classification error, output is thresholded at 1/2

	// evaluate solution found by training
	network.setParameterVector(optimizer.solution().point); // set weights to weights found by learning
	Data<RealVector> prediction = network(dataset.inputs());
	cout << "classification error after learning:\t" << loss01.eval(dataset.labels(), prediction) << endl;
	
	//###begin<probability>
	cout<<"probabilities:"<<std::endl;
	Softmax probabilty(1);
	for(std::size_t i = 0; i != 4; ++i){
		cout<< (network>>probabilty)(dataset.element(i).input)<<std::endl;
	}
	//###end<probability>
}
