#include <iostream>
#include <shark/Data/SparseData.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Algorithms/Trainers/RFTrainer.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>

#include <shark/Core/Timer.h>

using namespace shark;
using namespace std;

int main(int argc, char **argv) {
	LabeledData<RealVector,unsigned int> data;
	importSparseData(data, "covtype");
	data.shuffle();
	ClassificationDataset test = splitAtElement(data,400000);

	RFClassifier<unsigned int> model;
	RFTrainer<unsigned int> trainer(true,true);
		
	Timer time;
	trainer.train(model, data);
	double time_taken = time.stop();
		
	ZeroOneLoss<> loss;
	cout <<  time_taken <<" "<< 1.0 - loss(data.labels(),model(data.inputs()))<< " "<< 1.0 - loss(test.labels(),model(test.inputs()))<<std::endl;
}