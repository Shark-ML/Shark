

//###begin<skeleton>
#include <shark/Data/Dataset.h>
#include <shark/Data/Csv.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
//###end<skeleton>

//###begin<SVM-includes>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
//###end<SVM-includes>


//###begin<skeleton>
using namespace shark;

int main()
{
	// Load data, use 70% for training and 30% for testing.
	// The path is hard coded; make sure to invoke the executable
	// from a place where the data file can be found. It is located
	// under [shark]/examples/Supervised/data.
	ClassificationDataset traindata, testdata;
	import_csv(traindata, "data/quickstartData.csv", LAST_COLUMN, ' ');
	testdata = splitAtElement(traindata, 70 * traindata.numberOfElements() / 100);
//###end<skeleton>

//###begin<SVM>
	double gamma = 1.0;         // kernel bandwidth parameter
	double C = 10.0;            // regularization parameter
	GaussianRbfKernel<RealVector> kernel(gamma);
	KernelClassifier<RealVector> model(&kernel);
	CSvmTrainer<RealVector> trainer(
			&kernel,
			C,
			true); /* true: train model with offset */
//###end<SVM>

//###begin<skeleton>
	trainer.train(model, traindata);

	Data<unsigned int> prediction = model(testdata.inputs());

	ZeroOneLoss<unsigned int> loss;
	double error_rate = loss(testdata.labels(), prediction);

	std::cout << "model: " << model.name() << std::endl
		<< "trainer: " << trainer.name() << std::endl
		<< "test error rate: " << error_rate << std::endl;
}
//###end<skeleton>
