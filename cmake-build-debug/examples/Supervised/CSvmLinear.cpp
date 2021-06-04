
#include <shark/Data/Dataset.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Data/DataDistribution.h>


using namespace shark;
using namespace std;


	typedef RealVector VectorType;
	// or:
	// typedef CompressedRealVector VectorType;


int main(int argc, char** argv)
{
	// experiment settings
	unsigned int ell = 500;      // number of training data point
	unsigned int tests = 10000;  // number of test data points
	double C = 1.0;              // regularization parameter

	// generate dataset
	PamiToy problem; // artificial benchmark data
	LabeledData<VectorType, unsigned int> training;
	training = problem.generateDataset(ell);
	LabeledData<VectorType, unsigned int> test;
	test = problem.generateDataset(tests);

	// define the model
	LinearClassifier<VectorType> model;

	// define the machine
	LinearCSvmTrainer<VectorType> trainer(C, false);

	// train the machine
	cout << "Algorithm: " << trainer.name() << "\ntraining ..." << flush; // Shark algorithms know their names
	trainer.train(model, training);
	cout << "\n  number of iterations: " << trainer.solutionProperties().iterations;
	cout << "\n  dual value: " << trainer.solutionProperties().value;
	cout << "\n  training time: " << trainer.solutionProperties().seconds << " seconds\ndone." << endl;

	// evaluate
	ZeroOneLoss<unsigned int> loss;
	Data<unsigned int> output = model(training.inputs());
	double train_error = loss.eval(training.labels(), output);
	cout << "training error:\t" <<  train_error << endl;
	output = model(test.inputs());
	double test_error = loss.eval(test.labels(), output);
	cout << "test error:\t" << test_error << endl;
}
