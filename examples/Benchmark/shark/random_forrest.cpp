#include <shark/Data/SparseData.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/Trainers/RFTrainer.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>

#include <shark/Core/Timer.h>
#include <iostream>
using namespace shark;
using namespace std;

int main(int argc, char **argv) {
	LabeledData<RealVector,unsigned int> data;
	importSparseData(data, "cod-rna",0,8192);

	RFClassifier model;
	RFTrainer trainer;
		
	Timer time;
	trainer.train(model, data);
	double time_taken = time.stop();
		
	ZeroOneLoss<unsigned int,RealVector> loss;
	cout <<  time_taken <<" "<< loss(data.labels(),model(data.inputs()))<<std::endl;
}