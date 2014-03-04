#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Data/DataDistribution.h>

//###begin<additional_includes>
#include <shark/ObjectiveFunctions/CrossValidationError.h>
#include <shark/Algorithms/DirectSearch/GridSearch.h>
#include <shark/Algorithms/JaakkolaHeuristic.h>
//###end<additional_includes>

using namespace shark;
using namespace std;


int main()
{
	// problem definition
	Chessboard prob;
	ClassificationDataset dataTrain = prob.generateDataset(200);
	ClassificationDataset dataTest= prob.generateDataset(10000);

	// SVM setup
	//###begin<setup>
	GaussianRbfKernel<> kernel(0.5, true); //unconstrained?
	KernelClassifier<RealVector> svm;
	bool offset = true;
	bool unconstrained = true;
	CSvmTrainer<RealVector> trainer(&kernel, 1.0, offset,unconstrained);
	//###end<setup>

	// cross-validation error
	//###begin<cv_error>
	const unsigned int N= 5;  // number of folds
	ZeroOneLoss<unsigned int> loss;
	CVFolds<ClassificationDataset> folds = createCVSameSizeBalanced(dataTrain, N);
	CrossValidationError<KernelClassifier<RealVector>, unsigned int> cvError(
		folds, &trainer, &svm, &trainer, &loss
	);
	//###end<cv_error>


	// find best parameters

	// use Jaakkola's heuristic as a starting point for the grid-search
	//###begin<jaakkola>
	JaakkolaHeuristic ja(dataTrain);
	double ljg = log(ja.gamma());
	//###end<jaakkola>
	cout << "Tommi Jaakkola says gamma = " << ja.gamma() << " and ln(gamma) = " << ljg << endl;

	//###begin<grid_configure>
	GridSearch grid;
	vector<double> min(2);
	vector<double> max(2);
	vector<size_t> sections(2);
	min[0] = ljg-4.; max[0] = ljg+4; sections[0] = 17;  // kernel parameter gamma
	min[1] = 0.0; max[1] = 10.0; sections[1] = 11;	   // regularization parameter C
	grid.configure(min, max, sections);
	//###end<grid_configure>
	//###begin<grid_train>
	grid.step(cvError);
	//###end<grid_train>

	// train model on the full dataset
	//###begin<train_optimal_params>
	trainer.setParameterVector(grid.solution().point);
	trainer.train(svm, dataTrain);
	//###end<train_optimal_params>
	cout << "grid.solution().point " << grid.solution().point << endl;
	cout << "C =\t" << trainer.C() << endl;
	cout << "gamma =\t" << kernel.gamma() << endl;

	// evaluate
	Data<unsigned int> output = svm(dataTrain.inputs());
	double train_error = loss.eval(dataTrain.labels(), output);
	cout << "training error:\t" << train_error << endl;
	output = svm(dataTest.inputs());
	double test_error = loss.eval(dataTest.labels(), output);
	cout << "test error: \t" << test_error << endl;
}
