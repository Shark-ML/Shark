#include <shark/Data/SparseData.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>
#include <shark/ObjectiveFunctions/Regularizer.h>

#include <shark/Algorithms/GradientDescent/LBFGS.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/Models/LinearModel.h>

#include <shark/Core/Timer.h>
#include <iostream>
using namespace shark;
using namespace std;

int main(int argc, char **argv) {
	ClassificationDataset data;
	importSparseData(data, "mnist",0,8192);
	double alpha = 0.1;
	CrossEntropy<unsigned int, RealVector> loss;
	LinearClassifier<> model;
	
	//Setting up the problem
	model.decisionFunction().setStructure(inputDimension(data),numberOfClasses(data),true);
	TwoNormRegularizer regularizer;
	ErrorFunction error(data,&model.decisionFunction(),&loss);
	error.setRegularizer(alpha,&regularizer);
	
	//solving
	Timer time;
	LBFGS optimizer;
	optimizer.init(error);
	while(error.evaluationCounter()<200){
		optimizer.step(error);
	}
	model.setParameterVector(optimizer.solution().point);
	double time_taken = time.stop();
	
	cout << "Cross-Entropy: " << loss(data.labels(),model.decisionFunction()(data.inputs()))<<std::endl;
	cout << "Time:\n" << time_taken << endl;
}