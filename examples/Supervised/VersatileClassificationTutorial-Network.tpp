
//###begin<skeleton>
#include <shark/Data/Dataset.h>
#include <shark/Data/Csv.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
//###end<skeleton>

//###begin<Network-includes>
#include <shark/Models/FFNet.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/Algorithms/StoppingCriteria/MaxIterations.h>
#include <shark/Algorithms/Trainers/OptimizationTrainer.h>
//###end<Network-includes>


//###begin<skeleton>
using namespace shark;

int main()
{
	// Load data, use 70% for training and 30% for testing.
	// The path is hard coded; make sure to invoke the executable
	// from a place where the data file can be found. It is located
	// under [shark]/examples/Supervised/data.
	ClassificationDataset traindata, testdata;
	import_csv(traindata, "data/quickstartData.csv", LAST_COLUMN, ' ');
	testdata = splitAtElement(traindata, 70 * traindata.numberOfElements() / 100);
//###end<skeleton>

//###begin<Network>
	typedef FFNet<LogisticNeuron, LogisticNeuron> ModelType; // sigmoid transfer function for hidden and output neurons
	ModelType model;
	size_t N = inputDimension(traindata);
	size_t M = 10;
	model.setStructure(N, M, 2);         // N inputs (depends on the data),
	                                     // M hidden neurons (depends on problem difficulty),
	                                     // and two output neurons (two classes).
	initRandomUniform(model, -0.1, 0.1); // initialize with small random weights
	CrossEntropy trainloss;              // differentiable loss for neural network training
	ErrorFunction<RealVector, unsigned int> error(&model, &trainloss, traindata);
	IRpropPlus optimizer;                // gradient-based optimization algorithm
	MaxIterations<> stop(100);           // stop optimization after 100 Rprop steps
	OptimizationTrainer<ModelType, unsigned int> trainer(&error, &optimizer, &stop);
//###end<Network>

//###begin<skeleton>
	trainer.train(model, traindata);

//###begin<real-prediction>
	Data<RealVector> prediction = model(testdata.inputs());
//###end<real-prediction>

//###begin<real-loss>
	ZeroOneLoss<unsigned int, RealVector> loss;
//###end<real-loss>
	double error_rate = loss(testdata.labels(), prediction);

	std::cout << "model: " << model.name() << std::endl
		<< "trainer: " << trainer.name() << std::endl
		<< "test error rate: " << error_rate << std::endl;
}
//###end<skeleton>
