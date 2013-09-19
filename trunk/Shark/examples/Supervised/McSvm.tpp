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

	// SVM us kernel classifiers
	KernelClassifier<RealVector>  svm;

	// loss measuring classification errors
	ZeroOneLoss<unsigned int> loss;

	// There are 8 trainers for multi-class SVMs in Shark which can train with or without bias:
	AbstractSvmTrainer<RealVector, unsigned int>* trainer[16];
	trainer[0] = new McSvmOVATrainer<RealVector>(&kernel, C,false);
	trainer[1] = new McSvmCSTrainer<RealVector>(&kernel, C,false);
	trainer[2] = new McSvmWWTrainer<RealVector>(&kernel, C,false);
	trainer[3] = new McSvmLLWTrainer<RealVector>(&kernel, C,false);
	trainer[4] = new McSvmADMTrainer<RealVector>(&kernel, C,false);
	trainer[5] = new McSvmATSTrainer<RealVector>(&kernel, C,false);
	trainer[6] = new McSvmATMTrainer<RealVector>(&kernel, C,false);
	trainer[7] = new McSvmMMRTrainer<RealVector>(&kernel, C,false);
	trainer[8] = new McSvmOVATrainer<RealVector>(&kernel, C,true);
	trainer[9] = new McSvmCSTrainer<RealVector>(&kernel, C,true);
	trainer[10] = new McSvmWWTrainer<RealVector>(&kernel, C,true);
	trainer[11] = new McSvmLLWTrainer<RealVector>(&kernel, C,true);
	trainer[12] = new McSvmADMTrainer<RealVector>(&kernel, C,true);
	trainer[13] = new McSvmATSTrainer<RealVector>(&kernel, C,true);
	trainer[14] = new McSvmATMTrainer<RealVector>(&kernel, C,true);
	trainer[15] = new McSvmMMRTrainer<RealVector>(&kernel, C,true);

	std::printf("SHARK multi-class SVM example - training 16 machines:\n");
	for (i=0; i<16; i++)
	{
		trainer[i]->train(svm, training);
		Data<unsigned int> output = svm(training.inputs());
		double train_error = loss.eval(training.labels(), output);
		output = svm(test.inputs());
		double test_error = loss.eval(test.labels(), output);

		std::printf(
			"[%2d] %10s %s    iterations=%10d    time=%9.4g seconds    training error=%9.4g    test error=%9.4g\n",
			(2*i+1),
			trainer[i]->name().c_str(),
			trainer[i]->trainOffset()? "with bias   ":"without bias",
			(int)trainer[i]->solutionProperties().iterations,
			trainer[i]->solutionProperties().seconds,
			train_error,
			test_error
		);
	}
	
	//clean up
	for(std::size_t i = 0; i < 16; ++i){
		delete trainer[i];
	}
}
