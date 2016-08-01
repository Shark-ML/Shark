#include <shark/Data/SparseData.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>

#include <shark/Core/Timer.h>
#include <iostream>
using namespace shark;
using namespace std;

int main(int argc, char **argv) {
	LabeledData<CompressedRealVector,unsigned int> data_sparse;
	importSparseData(data_sparse, "rcv1_train.binary",0,8192);

	for(double C = 1; C <= 100000; C*=10){
		LinearClassifier<CompressedRealVector> model;
		LinearCSvmTrainer<CompressedRealVector> trainer(C,true);
		
		Timer time;
		trainer.train(model, data_sparse);
		double time_taken = time.stop();
		
		ZeroOneLoss<> loss;
		cout << C <<"  " <<  time_taken <<" "<< loss(data_sparse.labels(),model(data_sparse.inputs()))<<std::endl;
	}
}