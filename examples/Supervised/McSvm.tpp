#include <cstdio>

#include <shark/LinAlg/Base.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Models/Converter.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Algorithms/Trainers/McSvmOVATrainer.h>
#include <shark/Algorithms/Trainers/McSvmCSTrainer.h>
#include <shark/Algorithms/Trainers/McSvmWWTrainer.h>
#include <shark/Algorithms/Trainers/McSvmLLWTrainer.h>
#include <shark/Algorithms/Trainers/McSvmADMTrainer.h>
#include <shark/Algorithms/Trainers/McSvmATSTrainer.h>
#include <shark/Algorithms/Trainers/McSvmATMTrainer.h>
#include <shark/Algorithms/Trainers/McSvmMMRTrainer.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>


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
	unsigned int i;

	// experiment settings
	unsigned int classes = 5;
	unsigned int ell = 30;
	unsigned int tests = 100;
	double C = 10.0;
	double gamma = 0.5;

	// generate a very simple dataset with a little noise
	Problem problem;
	ClassificationDataset training = problem.generateDataset(ell);
	ClassificationDataset test = problem.generateDataset(tests);

	// kernel function
	GaussianRbfKernel<> kernel(gamma);

	// SVM kernel expansions with real vector-valued
	// output are converted to class labels
	ArgMaxConverter<KernelExpansion<RealVector> >  svm_no_bias;
	ArgMaxConverter<KernelExpansion<RealVector> >  svm_with_bias;
	svm_no_bias.decisionFunction().setStructure(false,classes);
	svm_with_bias.decisionFunction().setStructure(true,classes);

	// loss measuring classification errors
	ZeroOneLoss<unsigned int> loss;

	// There are 8 trainers for multi-class SVMs in Shark:
	AbstractSvmTrainer<RealVector, unsigned int>* trainer[8];
	trainer[0] = new McSvmOVATrainer<RealVector>(&kernel, C);
	trainer[1] = new McSvmCSTrainer<RealVector>(&kernel, C);
	trainer[2] = new McSvmWWTrainer<RealVector>(&kernel, C);
	trainer[3] = new McSvmLLWTrainer<RealVector>(&kernel, C);
	trainer[4] = new McSvmADMTrainer<RealVector>(&kernel, C);
	trainer[5] = new McSvmATSTrainer<RealVector>(&kernel, C);
	trainer[6] = new McSvmATMTrainer<RealVector>(&kernel, C);
	trainer[7] = new McSvmMMRTrainer<RealVector>(&kernel, C);

	std::printf("SHARK multi-class SVM example - training 16 machines:\n");
	for (i=0; i<8; i++)
	{
		trainer[i]->train(svm_no_bias.decisionFunction(), training);
		Data<unsigned int> output = svm_no_bias(training.inputs());
		double train_error = loss.eval(training.labels(), output);
		output = svm_no_bias(test.inputs());
		double test_error = loss.eval(test.labels(), output);

		std::printf("[%2d] %10s %s    iterations=%10d    time=%9.4g seconds    training error=%9.4g    test error=%9.4g\n",
				(2*i+1),
				trainer[i]->name().c_str(),
				"without bias",
				(int)trainer[i]->solutionProperties().iterations,
				trainer[i]->solutionProperties().seconds,
				train_error,
				test_error);

		trainer[i]->train(svm_with_bias.decisionFunction(), training);
		output = svm_with_bias(training.inputs());
		train_error = loss.eval(training.labels(), output);
		output = svm_with_bias(test.inputs());
		test_error = loss.eval(test.labels(), output);

		std::printf("[%2d] %10s %s    iterations=%10d    time=%9.4g seconds    training error=%9.4g    test error=%9.4g\n",
				(2*i+2),
				trainer[i]->name().c_str(),
				"with bias   ",
				(int)trainer[i]->solutionProperties().iterations,
				trainer[i]->solutionProperties().seconds,
				train_error,
				test_error);
	}
	
	//clean up
	for(std::size_t i = 0; i < 8; ++i){
		delete trainer[i];
	}
}
