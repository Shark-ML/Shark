
//###begin<skeleton>
#include <shark/Data/Dataset.h>
#include <shark/Data/Csv.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
//###end<skeleton>

//###begin<RF-includes>
#include <shark/Models/Trees/RFClassifier.h>
#include <shark/Algorithms/Trainers/RFTrainer.h>
//###end<RF-includes>


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

//###begin<RF>
	RFClassifier model;
	RFTrainer trainer;
//###end<RF>

//###begin<skeleton>
	trainer.train(model, traindata);

//###begin<real-prediction>
	Data<RealVector> prediction = model(testdata.inputs());
//###end<real-prediction>

//###begin<real-loss>
	ZeroOneLoss<unsigned int, RealVector> loss;
//###end<real-loss>
	double error_rate = loss(testdata.labels(), prediction);

	std::cout << "model: " << model.name() << std::endl
		<< "trainer: " << trainer.name() << std::endl
		<< "test error rate: " << error_rate << std::endl;
}
//###end<skeleton>
