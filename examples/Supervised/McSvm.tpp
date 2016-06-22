#include <cstdio>
#include <tuple>

#include <shark/LinAlg/Base.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Models/Converter.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
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

	// SVM kernel classifiers
	KernelClassifier<RealVector>  svm;

	// loss measuring classification errors
	ZeroOneLoss<unsigned int> loss;

	// There are 9 trainers for multi-class SVMs in Shark which can train with or without bias:
	std::tuple<std::string,McSvm,bool> machines[18] ={
		{"OVA", McSvm::OVA,false},
		{"CS", McSvm::CS,false},
		{"WW",McSvm::WW,false},
		{"LLW",McSvm::LLW,false},
		{"ADM",McSvm::ADM,false},
		{"ATS",McSvm::ATS,false},
		{"ATM",McSvm::ATM,false},
		{"MMR",McSvm::MMR,false},
		{"ReinforcedSvm",McSvm::ReinforcedSvm,false},
		{"OVA", McSvm::OVA,true},
		{"CS", McSvm::CS,true},
		{"WW",McSvm::WW,true},
		{"LLW",McSvm::LLW,true},
		{"ADM",McSvm::ADM,true},
		{"ATS",McSvm::ATS,true},
		{"ATM",McSvm::ATM,true},
		{"MMR",McSvm::MMR,true},
		{"ReinforcedSvm",McSvm::ReinforcedSvm,true}
	};

	std::printf("SHARK multi-class SVM example - training 18 machines:\n");
	for (int i=0; i<18; i++)
	{
		CSvmTrainer<RealVector> trainer(&kernel, C, std::get<2>(machines[i]));
		trainer.setMcSvmType(std::get<1>(machines[i]));
		trainer.train(svm, training);
		Data<unsigned int> output = svm(training.inputs());
		double train_error = loss.eval(training.labels(), output);
		output = svm(test.inputs());
		double test_error = loss.eval(test.labels(), output);

		std::cout<<std::get<0>(machines[i])<<(trainer.trainOffset()? " w bias   ":" w/o bias");
		std::cout<<"\ttraining error="<<train_error;
		std::cout<<"\ttest error="<<test_error;
		std::cout<<"\titerations="<<trainer.solutionProperties().iterations;
		std::cout<<"\ttime="<<trainer.solutionProperties().seconds<<std::endl;
		
		
	}
}
