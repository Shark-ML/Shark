
#include <shark/Data/Dataset.h>
#include <shark/Models/LinearClassifier.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Data/DataDistribution.h>
//###begin<includes>
#include <shark/Algorithms/Trainers/McSvmOVATrainer.h>
//###end<includes>

using namespace shark;


double const noise = 1.0;
typedef CompressedRealVector VectorType;
typedef CompressedRealMatrix MatrixType;
typedef CompressedRealMatrixRow RowType;


// data generating distribution for our toy
// multi-category classification problem
/// @cond EXAMPLE_SYMBOLS
class Problem : public LabeledDataDistribution<VectorType, unsigned int>
{
public:
	void draw(VectorType& input, unsigned int& label)const
	{
		label = Rng::discrete(0, 4);
		input.resize(1000002);
		input(1000000) = noise * Rng::gauss() + 3.0 * std::cos((double)label);
		input(1000001) = noise * Rng::gauss() + 3.0 * std::sin((double)label);
	}
};
/// @endcond


int main(int argc, char** argv)
{
	if (argc != 4)
	{
		std::cout << "required parameters: ell C epsilon" << std::endl;
		return 1;
	}

	// experiment settings
	unsigned int ell = std::atoi(argv[1]);
	double C = std::atof(argv[2]);
	double epsilon = std::atof(argv[3]);
	unsigned int tests = 10000;
	std::cout << "ell=" << ell << std::endl;
	std::cout << "C=" << C << std::endl;
	std::cout << "epsilon=" << epsilon << std::endl;

	// generate a very simple dataset with a little noise
	Problem problem;
	LabeledData<VectorType, unsigned int> training = problem.generateDataset(ell);
	LabeledData<VectorType, unsigned int> test = problem.generateDataset(tests);

	// define the model
	LinearClassifier<VectorType > svm;

	// train the machine
	std::cout << "machine training ..." << std::endl;
//###begin<trainer>
	LinearMcSvmOVATrainer<VectorType> trainer(C, epsilon);
//###end<trainer>
	trainer.train(svm, training);
	std::cout << "done." << std::endl;

	// loss measuring classification errors
	ZeroOneLoss<unsigned int> loss;

	Data<unsigned int> output = svm(training.inputs());
	double train_error = loss.eval(training.labels(), output);
	output = svm(test.inputs());
	double test_error = loss.eval(test.labels(), output);
	std::cout <<"training error= "<< train_error <<"    test error= "<<  test_error<<std::endl;
}
