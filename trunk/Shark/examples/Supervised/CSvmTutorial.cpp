#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Data/DataDistribution.h>

using namespace shark;
using namespace std;

int main(int argc, char** argv)
{
	// experiment settings
	unsigned int ell = 500;     // number of training data point
	unsigned int tests = 10000; // number of test data points
	double C = 1000.0;          // regularization parameter
	double gamma = 0.5;         // kernel bandwidth parameter
	bool bias = true;           // use bias/offset parameter

	GaussianRbfKernel<> kernel(gamma); // Gaussian kernel
	KernelExpansion<RealVector> ke(&kernel, bias); // (affine) linear function in kernel-induced feature space

	ZeroOneLoss<unsigned int, RealVector> loss; // 0-1 loss

	// generate dataset
	Chessboard problem; // artificial benchmark data
	ClassificationDataset training = problem.generateDataset(ell);
	ClassificationDataset test = problem.generateDataset(tests);

	// define the machine
	CSvmTrainer<RealVector> trainer(&kernel, C);

//	// ADDITIONAL/ADVANCED SVM SOLVER OPTIONS:
//	//to use "double" as kernel matrix cache type internally instead of float:
//	CSvmTrainer<RealVector, double> trainer(&kernel, C);
//	//to keep non-support vectors after training:
//	trainer.sparsify() = false;
//	//to relax or tighten the stopping criterion from 1e-3 (here, tightened to 1e-6)
//	trainer.stoppingCondition().minAccuracy = 1e-6;
//	//to set the cache size to 128MB for double (16**6 times sizeof(double), when double was selected as cache type above)
//	//or to 64MB for float (16**6 times sizeof(float), when the CSvmTrainer is declared without second template argument)
//	trainer.setCacheSize( 0x1000000 );

	// train the machine
	cout << "Algorithm: " << trainer.name() << "\ntraining ..." << flush; // Shark algorithms know their names
	trainer.train(ke, training);
	cout << "\n  number of iterations: " << trainer.solutionProperties().iterations;
	cout << "\n  dual value: " << trainer.solutionProperties().value;
	cout << "\n  training time: " << trainer.solutionProperties().seconds << " seconds\ndone." << endl;

	// evaluate
	Data<RealVector> output;  // real-valued output of the machine
	output = ke(training.inputs()); // evaluate on training set
	double train_error = loss.eval(training.labels(), output);
	cout << "training error:\t" <<  train_error << endl;
	output = ke(test.inputs()); // evaluate on test set
	double test_error = loss.eval(test.labels(), output);
	cout << "test error:\t" << test_error << endl;
}
