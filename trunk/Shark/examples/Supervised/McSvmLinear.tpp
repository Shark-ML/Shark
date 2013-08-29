#include <shark/Algorithms/Pegasos.h>
#include <shark/Algorithms/Trainers/McSvmLLWTrainer.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Models/Converter.h>
#include <shark/Models/ConcatenatedModel.h>
#include <shark/Models/LinearModel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

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
		std::cout << "required parameters: ell lambda epsilon" << std::endl;
		return 1;
	}

	// experiment settings
	unsigned int dim = 1000002;
	unsigned int classes = 5;
	unsigned int ell = std::atoi(argv[1]);
	double lambda = std::atof(argv[2]);
	double epsilon = std::atof(argv[3]);
	unsigned int tests = 10000;
	std::cout <<"ell="<< ell<<std::endl;
	std::cout <<"lambda="<<  lambda<<std::endl;
	std::cout <<"epsilon="<< epsilon<<std::endl;

	// generate a very simple dataset with a little noise
	Problem problem;
	LabeledData<VectorType, unsigned int> training = problem.generateDataset(ell);
	LabeledData<VectorType, unsigned int> test = problem.generateDataset(tests);

	// define the model
	ArgMaxConverter<LinearModel<VectorType, RealVector> >svm;
	svm.decisionFunction().setStructure(dim,classes,false,true);

	// train the machine
	std::cout << "machine training ..." << std::endl;
	LinearMcSvmLLWTrainer trainer(1.0 / (lambda * ell), epsilon);
	trainer.train(svm.decisionFunction(), training);
	std::cout << "done." << std::endl;

	// loss measuring classification errors
	ZeroOneLoss<unsigned int> loss;

	Data<unsigned int> output = svm(training.inputs());
	double train_error = loss.eval(training.labels(), output);
	output = svm(test.inputs());
	double test_error = loss.eval(test.labels(), output);
	std::cout <<"training error= "<< train_error <<"    test error= "<<  test_error<<std::endl;
}
