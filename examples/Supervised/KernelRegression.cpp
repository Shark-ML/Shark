#include <shark/LinAlg/Base.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Algorithms/Trainers/EpsilonSvmTrainer.h>
#include <shark/Algorithms/Trainers/RegularizationNetworkTrainer.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>


using namespace shark;


int main()
{
	// experiment settings
	unsigned int ell = 200;
	unsigned int tests = 10000;
	double C = 10.0;
	double gamma = 1.0 / C;
	double epsilon = 0.03;

	GaussianRbfKernel<> kernel(0.1);
	SquaredLoss<> loss;

	// generate dataset
	Wave problem;
	RegressionDataset training = problem.generateDataset(ell);
	RegressionDataset test = problem.generateDataset(tests);

	// define the machines
	KernelExpansion<RealVector> svm[2] = {
		KernelExpansion<RealVector>(true),
		KernelExpansion<RealVector>(false)
	};

	// define the corresponding trainers
	AbstractTrainer<KernelExpansion<RealVector> >* trainer[2];
	trainer[0] = new EpsilonSvmTrainer<RealVector>(&kernel, C, epsilon);
	trainer[1] = new RegularizationNetworkTrainer<RealVector>(&kernel, gamma);

	for (unsigned int i=0; i<2; i++)
	{
		std::cout<<"METHOD"<<(i+1) <<" "<< trainer[i]->name().c_str()<<std::endl;
		std::cout<<"training ..."<<std::flush;
		trainer[i]->train(svm[i], training);
		std::cout<<"done"<<std::endl;

		Data<RealVector> output = svm[i](training.inputs());
		double train_error = loss.eval(training.labels(), output);
		std::cout<<"training error: "<<train_error<<std::endl;
		output = svm[i](test.inputs());
		double test_error = loss.eval(test.labels(), output);
		std::cout<<"    test error: "<<test_error<<"\n\n";
	}

	delete trainer[0];
	delete trainer[1];
}
