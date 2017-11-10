#include <iostream>
#include <shark/Data/SparseData.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Algorithms/Trainers/RFTrainer.h>
#include <shark/Algorithms/Trainers/CARTTrainer.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>

#include <shark/Core/Timer.h>

using namespace shark;
using namespace std;

int main(int argc, char **argv) {
	//~ {
		//~ LabeledData<RealVector,unsigned int> data;
		//~ importSparseData(data, "covtype");
		//~ data.shuffle();
		//~ ClassificationDataset test = splitAtElement(data,400000);

		//~ RFClassifier<unsigned int> model;
		//~ RFTrainer<unsigned int> trainer(true,true);
			
		//~ Timer time;
		//~ trainer.train(model, data);
		//~ double time_taken = time.stop();
			
		//~ ZeroOneLoss<> loss;
		//~ cout <<  time_taken <<" "<< 1.0 - loss(data.labels(),model(data.inputs()))<< " "<< 1.0 - loss(test.labels(),model(test.inputs()))<<std::endl;
	//~ }
	{
		RegressionDataset data;
		importSparseData(data, "cod-rna");
		CARTree<RealVector> model;
		CARTTrainer<RealVector> trainer;
		//~ trainer.setNTrees(100);
		//~ trainer.setMTry(inputDimension(data));
		Timer time;
		trainer.train(model, data);
		double time_taken = time.stop();
			
		SquaredLoss<> loss;
		cout <<  time_taken <<" "<< loss(data.labels(),model(data.inputs()))<<std::endl;
	}
}