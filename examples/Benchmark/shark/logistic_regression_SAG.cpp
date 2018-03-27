#include <shark/Data/SparseData.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>
#include <shark/Algorithms/Trainers/LinearSAGTrainer.h>

#include <shark/Core/Timer.h>
#include <iostream>
using namespace shark;
using namespace std;


template<class InputType>
void run(LabeledData<InputType,unsigned int> const& data, double alpha, unsigned int epochs){
	CrossEntropy<unsigned int, RealVector> loss;
	LinearClassifier<InputType> model;
	
	
	LinearSAGTrainer<InputType,unsigned int> trainer(&loss,alpha);
	trainer.setEpochs(epochs);
	
	Timer time;
	trainer.train(model, data);
	double time_taken = time.stop();
	
	cout << "Cross-Entropy: " << loss(data.labels(),model.decisionFunction()(data.inputs()))<<std::endl;
	cout << "Time:\n" << time_taken << endl;
}
int main(int argc, char **argv) {
	ClassificationDataset data_dense;
	importSparseData(data_dense, "mnist",0,8192);
	data_dense = transformLabels(data_dense, [](unsigned int y){ return y%2;});
	LabeledData<CompressedRealVector,unsigned int> data_sparse;
	importSparseData(data_sparse, "rcv1_train.binary",0,8192);
	
	double alpha = 0.1;
	run(data_dense, alpha, 200);
	run(data_sparse, alpha, 2000);
	
}