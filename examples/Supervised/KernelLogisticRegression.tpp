
#include <shark/LinAlg/Base.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Algorithms/Trainers/KernelSGDTrainer.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>
#include <iostream>


using namespace shark;


// data generating distribution for our toy
// multi-category classification problem
/// @cond EXAMPLE_SYMBOLS
class Problem : public LabeledDataDistribution<RealVector, unsigned int>
{
public:
	void draw(RealVector& input, unsigned int& label)const
	{
		label = Rng::discrete(0, 4);
		input.resize(1);
		input(0) = Rng::gauss() + 3.0 * label;
	}
};
/// @endcond

int main()
{
	std::cout << "kernel logistic regression example program" << std::endl;

	// experiment settings
	unsigned int ell = 1000;
	unsigned int tests = 1000;
	double C = 10.0;
	double gamma = 0.5;

	// generate a very simple dataset with a little noise
	Problem problem;
	ClassificationDataset training = problem.generateDataset(ell);
	ClassificationDataset test = problem.generateDataset(tests);

	// kernel function
	GaussianRbfKernel<> kernel(gamma);

	// classifier model
	KernelClassifier<RealVector> classifier;

	// loss measuring classification errors
	ZeroOneLoss<unsigned int> loss;

	// loss measuring training errors
	CrossEntropy crossentropy;

	// machine training
	KernelSGDTrainer<RealVector> trainer(&kernel, &crossentropy, C, false);
	trainer.train(classifier, training);

	// evaluation
	Data<unsigned int> output = classifier(training.inputs());
	double train_error = loss.eval(training.labels(), output);
	std::cout << "training error: " << train_error << std::endl;
	output = classifier(test.inputs());
	double test_error = loss.eval(test.labels(), output);
	std::cout << "    test error: " << test_error << std::endl;
}
