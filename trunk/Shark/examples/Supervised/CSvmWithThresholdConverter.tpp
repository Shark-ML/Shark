#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Models/Converter.h>
#include <shark/Models/ConcatenatedModel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Data/Dataset.h>
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
	KernelExpansion<RealVector> ke(&kernel, bias); // 
	ThresholdConverter conv;
	ConcatenatedModel<RealVector, unsigned int> svm(&ke, &conv);

	// generate dataset
	Chessboard problem; // artificial benchmark data
	ClassificationDataset training = problem.generateDataset(ell);
	ClassificationDataset test = problem.generateDataset(tests);

	// define the machine
	CSvmTrainer<RealVector> trainer(&kernel, C); //mtqp

	// train the machine
	cout << "METHOD: " << trainer.name() << "\ntraining ..." << flush;
	trainer.train(ke, training);
	cout << "\n  number of iterations: " << trainer.solutionProperties().iterations;
	cout << "\n  training time: " << trainer.solutionProperties().seconds << " seconds\ndone." << endl;

	// evaluate
	ZeroOneLoss<unsigned int, unsigned int> loss;
	Data<unsigned int> output = svm(training.inputs());
	double train_error = loss.eval(training.labels(), output);
	cout << "training error:\t" <<  train_error << endl;
	output = svm(test.inputs());
	double test_error = loss.eval(test.labels(), output);
	cout << "test error:\t" << test_error << endl;
}
