#include <shark/Data/SparseData.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>

#include <shark/Core/Timer.h>
#include <iostream>
using namespace shark;
using namespace std;

int main(int argc, char **argv) {
	LabeledData<RealVector,unsigned int> data;
	importSparseData(data, "cod-rna",0,8192);

	GaussianRbfKernel<> kernel(1.0);
	for(double C = 0.01; C <= 1; C*=10){
		KernelClassifier<RealVector> model;
		CSvmTrainer<RealVector> trainer(&kernel,C, true);
		
		Timer time;
		trainer.train(model, data);
		double time_taken = time.stop();
		
		ZeroOneLoss<> loss;
		cout << C <<"  " <<  time_taken <<" "<< loss(data.labels(),model(data.inputs()))<<std::endl;
	}
}