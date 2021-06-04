#include <shark/Algorithms/Trainers/Budgeted/KernelBudgetedSGDTrainer.h> // the KernelBudgetedSGD trainer
#include <shark/Algorithms/Trainers/Budgeted/MergeBudgetMaintenanceStrategy.h> // the strategy the trainer will use 
#include <shark/Data/DataDistribution.h> //includes small toy distributions
#include <shark/Models/Kernels/GaussianRbfKernel.h> //the used kernel for the SVM
#include <shark/ObjectiveFunctions/Loss/HingeLoss.h> // the loss we want to use for the SGD machine
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> //used for evaluation of the classifier

using namespace shark;
using namespace std;


// data generating distribution for our toy
// multi-category classification problem
class myProblem : public LabeledDataDistribution<RealVector, unsigned int>
{
public:
    void draw(RealVector& input, unsigned int& label)const
    {
        label = random::discrete(random::globalRng, 0, 4);
        input.resize(1);
        input(0) = random::gauss(random::globalRng) + 3.0 * label;
    }
};
/// @endcond




int main(int argc, char** argv)
{
	// experiment settings
	unsigned int ell = 500;     // number of training data point
	unsigned int tests = 10000; // number of test data points
	double gamma = 0.5;         // kernel bandwidth parameter
	double C = 1.0;          // regularization parameter
	bool bias = false;           // use bias/offset parameter
	size_t budgetSize = 16;     // our model shall contain at most 16 vectors
	size_t epochs = 5;      // we want to run 5 epochs


	GaussianRbfKernel<> kernel(gamma); // Gaussian kernel
	KernelClassifier<RealVector> kernelClassifier; // (affine) linear function in kernel-induced feature space

	// generate dataset
	Chessboard problem; // artificial benchmark data
	ClassificationDataset trainingData = problem.generateDataset(ell);
	ClassificationDataset testData = problem.generateDataset(tests);

	// define the machine
	HingeLoss hingeLoss; // define the loss we want to use while training
	// as the budget maintenance strategy we choose the merge strategy
	MergeBudgetMaintenanceStrategy<RealVector> *strategy = new MergeBudgetMaintenanceStrategy<RealVector>();
	KernelBudgetedSGDTrainer<RealVector> kernelBudgetedSGDtrainer(&kernel, &hingeLoss, C, bias, false, budgetSize, strategy);        // create the trainer
	kernelBudgetedSGDtrainer.setEpochs(epochs);      // set the epochs number

	// train the machine
	std::cout << "Training the " << kernelBudgetedSGDtrainer.name() << " on the problem with a budget of " << budgetSize << " and " << epochs << " Epochs..." << std::endl; // Shark algorithms know their names
	kernelBudgetedSGDtrainer.train(kernelClassifier, trainingData);
	Data<RealVector> supportVectors = kernelClassifier.decisionFunction().basis();  // get a pointer to the support vectors of the model
	size_t nSupportVectors = supportVectors.numberOfElements();     // get number of support vectors
	std::cout << "We have " << nSupportVectors << " support vectors in our model.\n";   // report

	// evaluate
	ZeroOneLoss<unsigned int> loss; // 0-1 loss
	Data<unsigned int> output = kernelClassifier(trainingData.inputs()); // evaluate on training set
	double train_error = loss.eval(trainingData.labels(), output);
	cout << "training error:\t" <<  train_error << endl;
	output = kernelClassifier(testData.inputs()); // evaluate on test set
	double test_error = loss.eval(testData.labels(), output);
	cout << "test error:\t" << test_error << endl;
}

