//###begin<includes>
//used for training the RBM
#include <shark/Unsupervised/RBM/BinaryRBM.h>
#include <shark/Algorithms/GradientDescent/SteepestDescent.h>

//the problem
#include <shark/Unsupervised/RBM/Problems/BarsAndStripes.h>

//for evaluation
#include <shark/Unsupervised/RBM/analytics.h>
#include <iostream>
//###end<includes>

using namespace shark;
using namespace std;

//initialize weights of rbm to be between -0.1 and 0.1
void initializeWeights(BinaryRBM& rbm){
	RealVector weights(rbm.numberOfParameters());
	for(size_t i = 0; i != weights.size(); ++i){
		weights(i) = Rng::uni(-0.1,0.1);
	}
	rbm.setParameterVector(weights);
}

int main(){

	//we first create the problem. in this tutorial, we use BarsAndStripes
	//###begin<problem>
	BarsAndStripes problem;
	UnlabeledData<RealVector> data = problem.data();
	//###end<problem>
	
	//some constants needed for training
	//###begin<RBM>
	size_t numberOfHidden = 32;//hidden units of the rbm
	size_t numberOfVisible = problem.inputDimension();//visible units of the inputs

	//create rbm with simple binary units
	BinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(numberOfVisible,numberOfHidden);
	//###end<RBM>
	
	//create derivative to optimize the rbm
	//we want a simple vanilla CD-1
	//###begin<cd>
	BinaryCD cd(&rbm);
	cd.setK(1);
	cd.setData(data);
	//###end<cd>

	//generate optimizer
	//###begin<optimizer>
	SteepestDescent optimizer;
	optimizer.setMomentum(0);
	optimizer.setLearningRate(0.1);
	//###end<optimizer>
	
	//now we train the rbm and evaluate the mean negative log-likelihood at the end
	//###begin<train>
	unsigned int numIterations = 1000;//iterations for training
	unsigned int numTrials = 10;//number of trials for training
	double meanResult = 0;
	for(unsigned int trial = 0; trial != numTrials; ++trial) {
		initializeWeights(rbm);
		optimizer.init(cd);

		for(unsigned int iteration = 0; iteration != numIterations; ++iteration) {
			optimizer.step(cd);
		}
		//evaluate exact likelihood after training. this is only possible for small problems!
		double likelihood = negativeLogLikelihood(rbm,data);
		std::cout<<trial<<" "<<likelihood<<std::endl;
		meanResult +=likelihood;
	}
	meanResult /= numTrials;
	//###end<train>

	//print the mean performance
	//###begin<output>
	cout << "RESULTS: " << std::endl;
	cout << "======== " << std::endl;
	cout << "mean negative log likelihood: " << meanResult << std::endl;
	//###end<output>

}

