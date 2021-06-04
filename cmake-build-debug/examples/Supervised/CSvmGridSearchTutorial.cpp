#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Data/DataDistribution.h>

#include <shark/ObjectiveFunctions/CrossValidationError.h>
#include <shark/Algorithms/DirectSearch/GridSearch.h>
#include <shark/Algorithms/JaakkolaHeuristic.h>

using namespace shark;
using namespace std;


int main()
{
	// problem definition
	Chessboard prob;
	ClassificationDataset dataTrain = prob.generateDataset(200);
	ClassificationDataset dataTest= prob.generateDataset(10000);

	// SVM setup
	GaussianRbfKernel<> kernel(0.5, true); //unconstrained?
	KernelClassifier<RealVector> svm;
	bool offset = true;
	bool unconstrained = true;
	CSvmTrainer<RealVector> trainer(&kernel, 1.0, offset,unconstrained);

	// cross-validation error
	const unsigned int K = 5;  // number of folds
	ZeroOneLoss<unsigned int> loss;
	CVFolds<ClassificationDataset> folds = createCVSameSizeBalanced(dataTrain, K);
	CrossValidationError<KernelClassifier<RealVector>, unsigned int> cvError(
		folds, &trainer, &svm, &trainer, &loss
	);


	// find best parameters

	// use Jaakkola's heuristic as a starting point for the grid-search
	JaakkolaHeuristic ja(dataTrain);
	double ljg = log(ja.gamma());
	cout << "Tommi Jaakkola says gamma = " << ja.gamma() << " and ln(gamma) = " << ljg << endl;

	GridSearch grid;
	vector<double> min(2);
	vector<double> max(2);
	vector<size_t> sections(2);
	// kernel parameter gamma
	min[0] = ljg-4.; max[0] = ljg+4; sections[0] = 9;
	// regularization parameter C
	min[1] = 0.0; max[1] = 10.0; sections[1] = 11;
	grid.configure(min, max, sections);
	grid.step(cvError);

	// train model on the full dataset
	trainer.setParameterVector(grid.solution().point);
	trainer.train(svm, dataTrain);
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
