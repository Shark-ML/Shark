
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/MultiTaskKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Data/DataDistribution.h>

using namespace shark;
using namespace std;


// RealVector input with task index
typedef MultiTaskSample<RealVector> InputType;


// Multi-task problem with up to three tasks.
class MultiTaskProblem : public LabeledDataDistribution<InputType, unsigned int>
{
public:
	MultiTaskProblem()
	{
		m_task[0] = true;
		m_task[1] = true;
		m_task[2] = true;
	}

	void setTasks(bool task0, bool task1, bool task2)
	{
		m_task[0] = task0;
		m_task[1] = task1;
		m_task[2] = task2;
	}

	void draw(InputType& input, unsigned int& label) const
	{
		size_t taskindex = 0;
		do {
			taskindex = Rng::uni(0, 2);
		} while (! m_task[taskindex]);
		double x1 = Rng::gauss();
		double x2 = 3.0 * Rng::gauss();
		unsigned int y = (x1 > 0.0) ? 1 : 0;
		double alpha = 0.05 * M_PI * taskindex;
		input.input.resize(2);
		input.input(0) = cos(alpha) * x1 - sin(alpha) * x2;
		input.input(1) = sin(alpha) * x1 + cos(alpha) * x2;
		input.task = taskindex;
		label = y;
	}

protected:
	bool m_task[3];
};


int main(int argc, char** argv)
{
	// experiment settings
	unsigned int ell_train = 1000;    	// number of training data point from tasks 0 and 1
	unsigned int ell_test = 1000;     	// number of test data points from task 2
	double C = 1.0;                   		// regularization parameter
	double gamma = 0.5;               	// kernel bandwidth parameter

	// generate data
	MultiTaskProblem problem;
	problem.setTasks(true, true, false);
	LabeledData<InputType, unsigned int> training = problem.generateDataset(ell_train);
	problem.setTasks(false, false, true);
	LabeledData<InputType, unsigned int> test = problem.generateDataset(ell_test);

	// merge all inputs into a single data object
	Data<InputType> data(ell_train + ell_test);
	for (size_t i=0; i<ell_train; i++) 
		data.element(i) = training.inputs().element(i);
	for (size_t i=0; i<ell_test; i++) 
		data.element(ell_train + i) = test.inputs().element(i);

	// create kernel objects
	GaussianRbfKernel<RealVector> inputKernel(gamma);   // Gaussian kernel on inputs
	GaussianTaskKernel<RealVector> taskKernel(          // task similarity kernel
			data,         // all inputs with task indices, no labels
			3,            // total number of tasks
			inputKernel,  // base kernel for input similarity
			gamma);       // bandwidth for task similarity kernel
	MultiTaskKernel<RealVector> multiTaskKernel(&inputKernel, &taskKernel);

	// train the SVM
	KernelExpansion<InputType> ke(&multiTaskKernel, false);
	CSvmTrainer<InputType> trainer(&multiTaskKernel, C);
	cout << "training ..." << endl;
	trainer.train(ke, training);
	cout << "done." << endl;

	ZeroOneLoss<unsigned int, RealVector> loss;
	Data<RealVector> output;

	// evaluate training performance
	output = ke(training.inputs());
	double trainError = loss.eval(training.labels(), output);
	cout << "training error:\t" <<  trainError << endl;

	// evaluate its transfer performance
	output = ke(test.inputs());
	double testError = loss.eval(test.labels(), output);
	cout << "test error:\t" << testError << endl;
}
