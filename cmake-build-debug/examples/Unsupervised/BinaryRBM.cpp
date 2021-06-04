//used for training the RBM
#include <shark/Unsupervised/RBM/BinaryRBM.h>
#include <shark/Algorithms/GradientDescent/SteepestDescent.h>

//the problem
#include <shark/Unsupervised/RBM/Problems/BarsAndStripes.h>

//for evaluation
#include <shark/Unsupervised/RBM/analytics.h>
#include <iostream>

using namespace shark;
using namespace std;

int main(){

	//we first create the problem. in this tutorial, we use BarsAndStripes
	BarsAndStripes problem;
	UnlabeledData<RealVector> data = problem.data();
	
	//some constants needed for training
	size_t numberOfHidden = 32;//hidden units of the rbm
	size_t numberOfVisible = problem.inputDimension();//visible units of the inputs

	//create rbm with simple binary units
	BinaryRBM rbm(random::globalRng);
	rbm.setStructure(numberOfVisible,numberOfHidden);
	
	//create derivative to optimize the rbm
	//we want a simple vanilla CD-1
	BinaryCD cd(&rbm);
	cd.setK(1);
	cd.setData(data);

	//generate optimizer
	SteepestDescent<> optimizer;
	optimizer.setMomentum(0);
	optimizer.setLearningRate(0.1);
	
	//now we train the rbm and evaluate the mean negative log-likelihood at the end
	unsigned int numIterations = 1000;//iterations for training
	unsigned int numTrials = 10;//number of trials for training
	double meanResult = 0;
	for(unsigned int trial = 0; trial != numTrials; ++trial) {
		initRandomUniform(rbm, -0.1,0.1);
		cd.init();
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

	//print the mean performance
	cout << "RESULTS: " << std::endl;
	cout << "======== " << std::endl;
	cout << "mean negative log likelihood: " << meanResult << std::endl;

}

