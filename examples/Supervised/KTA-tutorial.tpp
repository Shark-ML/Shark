
#include <shark/Data/DataDistribution.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/ObjectiveFunctions/KernelTargetAlignment.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>

using namespace shark;
using namespace std;

int main(int argc, char** argv)
{
	// generate dataset
	Chessboard problem;          // artificial benchmark problem
	LabeledData<RealVector, unsigned int> data = problem.generateDataset(1000);

	// define the family of kernel functions
	double gamma = 0.5;          // initial guess of the parameter value
	GaussianRbfKernel<RealVector> kernel(gamma);

	// set up kernel target alignment as a function of the kernel parameters
	// on the given data
	KernelTargetAlignment<RealVector> kta(&kernel);
	kta.setDataset(data);

	// optimize parameters for best alignment
	IRpropPlus rprop;
	rprop.init(kta);
	cout << "initial parameter: " << kernel.gamma() << endl;
	for (size_t i=0; i<50; i++)
	{
		rprop.step(kta);
		cout << "parameter after step " << (i+1) << ": " << kernel.gamma() << endl;
	}
	cout << "final parameter: " << kernel.gamma() << endl;
}
