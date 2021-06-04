#include <cstdio>
#include <tuple>

#include <shark/LinAlg/Base.h>
#include <shark/Core/Random.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>
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
		label = random::discrete(random::globalRng, 0, 4);
		input.resize(1);
		input(0) = random::gauss(random::globalRng) + 3.0 * label;
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
		std::make_tuple("OVA", McSvm::OVA,false),
		std::make_tuple("CS", McSvm::CS,false),
		std::make_tuple("WW",McSvm::WW,false),
		std::make_tuple("LLW",McSvm::LLW,false),
		std::make_tuple("ADM",McSvm::ADM,false),
		std::make_tuple("ATS",McSvm::ATS,false),
		std::make_tuple("ATM",McSvm::ATM,false),
		std::make_tuple("MMR",McSvm::MMR,false),
		std::make_tuple("ReinforcedSvm",McSvm::ReinforcedSvm,false),
		std::make_tuple("OVA", McSvm::OVA,true),
		std::make_tuple("CS", McSvm::CS,true),
		std::make_tuple("WW",McSvm::WW,true),
		std::make_tuple("LLW",McSvm::LLW,true),
		std::make_tuple("ADM",McSvm::ADM,true),
		std::make_tuple("ATS",McSvm::ATS,true),
		std::make_tuple("ATM",McSvm::ATM,true),
		std::make_tuple("MMR",McSvm::MMR,true),
		std::make_tuple("ReinforcedSvm",McSvm::ReinforcedSvm,true)
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
