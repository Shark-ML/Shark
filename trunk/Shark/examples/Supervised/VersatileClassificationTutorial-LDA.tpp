
//###begin<skeleton>
#include <shark/Data/Dataset.h>
#include <shark/Data/Csv.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
//###end<skeleton>

//###begin<LDA-includes>
#include <shark/Models/LinearClassifier.h>
#include <shark/Algorithms/Trainers/LDA.h>
//###end<LDA-includes>


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

	// TODO: define a model and a trainer
//###end<skeleton>

//###begin<LDA>
	LinearClassifier<> model;
	LDA trainer;
//###end<LDA>

//###begin<skeleton>
//###begin<remove>
	trainer.train(model, traindata);
//###end<remove>

//###begin<int-prediction>
	Data<unsigned int> prediction = model(testdata.inputs());
//###end<int-prediction>

	ZeroOneLoss<unsigned int> loss;
	double error_rate = loss(testdata.labels(), prediction);

	std::cout << "model: " << model.name() << std::endl
//###begin<remove>
		<< "trainer: " << trainer.name() << std::endl
//###end<remove>
		<< "test error rate: " << error_rate << std::endl;
}
//###end<skeleton>
//