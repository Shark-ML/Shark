
//###begin<includes>
#include <shark/Data/Dataset.h>
#include <shark/Models/LinearClassifier.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
//###end<includes>
#include <shark/Data/DataDistribution.h>


using namespace shark;
using namespace std;


//###begin<vectortype>
	typedef RealVector VectorType;
	// or:
	// typedef CompressedRealVector VectorType;
//###end<vectortype>


int main(int argc, char** argv)
{
	// experiment settings
	unsigned int ell = 500;      // number of training data point
	unsigned int tests = 10000;  // number of test data points
	double C = 1.0;              // regularization parameter

	// generate dataset
	PamiToy problem; // artificial benchmark data
//###begin<traindata>
	LabeledData<VectorType, unsigned int> training;
//###end<traindata>
	training = problem.generateDataset(ell);
	LabeledData<VectorType, unsigned int> test;
	test = problem.generateDataset(tests);

	// define the model
//###begin<model>
	LinearClassifier<VectorType> model;
//###end<model>

	// define the machine
//###begin<trainer>
	LinearCSvmTrainer<VectorType> trainer(C);
//###end<trainer>

	// train the machine
	cout << "Algorithm: " << trainer.name() << "\ntraining ..." << flush; // Shark algorithms know their names
//###begin<training>
	trainer.train(model, training);
//###end<training>
	cout << "\n  number of iterations: " << trainer.solutionProperties().iterations;
	cout << "\n  dual value: " << trainer.solutionProperties().value;
	cout << "\n  training time: " << trainer.solutionProperties().seconds << " seconds\ndone." << endl;

	// evaluate
//###begin<evaluation>
	ZeroOneLoss<unsigned int> loss;
	Data<unsigned int> output = model(training.inputs());
	double train_error = loss.eval(training.labels(), output);
	cout << "training error:\t" <<  train_error << endl;
//###end<evaluation>
	output = model(test.inputs());
	double test_error = loss.eval(test.labels(), output);
	cout << "test error:\t" << test_error << endl;
}
