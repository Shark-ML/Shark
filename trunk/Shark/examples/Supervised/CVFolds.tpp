//header needed for cross validation
//###begin<includetools>
#include <shark/Data/CVDatasetTools.h>
//###end<includetools>

//headers needed for our test problem
#include <shark/Data/DataDistribution.h>
#include <shark/Models/FFNet.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/Regularizer.h>
#include <shark/ObjectiveFunctions/CombinedObjectiveFunction.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/Rng/Uniform.h>

//we use an artifical learning problem
#include <shark/Data/DataDistribution.h>

using namespace shark;
using namespace std;

///In this example, you will learn to create and use partitions
///for cross validation.
///This tutorial describes a handmade solution which does not use the Crossvalidation error function
///which is also provided by shark. We do this, because it gives a better on what Cross Validation does.

///The Test Problem receives the regularization parameter and a dataset
///and returns the errror. skip to the main if you are not interested
///in the problem itself. But here you can also see how to create
///regularized error functions. so maybe it's still worth taking a look ;)
double trainProblem(const RegressionDataset& training, RegressionDataset const& validation, double regularization){
	//we create a feed forward network with 20 hidden neurons.
	//this should create a lot of overfitting when not regularized
	FFNet<LogisticNeuron,LogisticNeuron> network;
	network.setStructure(1,20,1);

	//initialize with random weights between -5 and 5
	//todo: find better initialization for cv-error...
	RealVector startingPoint(network.numberOfParameters());
	Uniform<> uni( Rng::globalRng, -5, 5 );
	std::generate(startingPoint.begin(),startingPoint.end(),uni);

	//the error function is a combination of MSE and 2-norm error
	SquaredLoss<> loss;
	ErrorFunction<> error(&network,&loss);
	TwoNormRegularizer regularizer;

	//combine both functions
	CombinedObjectiveFunction<RealVector, double> regularizedError;
	regularizedError.add(error);
	regularizedError.add(regularization,regularizer);


	//now train for a number of iterations using Rprop
	error.setDataset(training);
	IRpropPlus optimizer;
	//initialize with our predefined point, since
	//the combined function can't propose one.
	optimizer.init(regularizedError,startingPoint);
	for(unsigned iter = 0; iter != 5000; ++iter)
	{
		optimizer.step(regularizedError);
	}

	//validate and return the error without regularization
	error.setDataset(validation);
	return error(optimizer.solution().point);
}


/// What is Cross Validation(CV)? In Cross Validation the dataset is partitioned in
/// several validation data sets. For a given validation dataset the remainder of the dataset
/// - every other validation set - forms the training part. During every evaluation of the error function, 
/// the problem is solved using the training part and the final error is computed using the validation part.
/// The mean of all validation sets trained this way is the final error of the solution found.
/// This quite complex procedure is used to minimize the bias introduced by the dataset itself and makes
/// overfitting of the solution harder.
int main(){
	//we first create the problem. in this simple tutorial,
	//it's only the 1D wave function sin(x)/x + noise
	Wave wave;
//###begin<cvselection>
	RegressionDataset dataset;
//###end<cvselection>
	dataset = wave.generateDataset(100);

	//now we want to create the cv folds. For this, we
	//use the CVDatasetTools.h. There are a few functions
	//to create folds. in this case, we create 4
	//partitions with the same size. so we have 75 train
	//and 25 validation data points
//###begin<cvselection>
	CVFolds<RegressionDataset> folds = createCVSameSize(dataset,4);
//###end<cvselection>

	//now we want to use the folds to find the best regularization
	//parameter for our problem. we use a grid search to accomplish this
//###begin<cvselection>
	double bestValidationError = 1e5;
	double bestRegularization = 0;
	for (double regularization = 0; regularization < 1.e-4; regularization +=1.e-5) {
		double result = 0;
		for (std::size_t fold = 0; fold != folds.size(); ++fold){ //CV
			// access the fold
			RegressionDataset training = folds.training(fold);
			RegressionDataset validation = folds.validation(fold);
			// train
			result += trainProblem(training, validation, regularization);
		}
		result /= folds.size();

		// check whether this regularization parameter leads to better results
		if (result < bestValidationError)
		{
			bestValidationError = result;
			bestRegularization = regularization;
		}

		// print status:
		std::cout << regularization << " " << result << std::endl;
	}
//###end<cvselection>

	// print the best value found
	cout << "RESULTS: " << std::endl;
	cout << "======== " << std::endl;
	cout << "best validation error: " << bestValidationError << std::endl;
	cout << "best regularization:   " << bestRegularization<< std::endl;
}
