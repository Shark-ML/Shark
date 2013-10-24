
//###begin<skeleton>
#include <shark/Data/Dataset.h>
#include <shark/Data/Csv.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
//###end<skeleton>

//###begin<NN-includes>
#include <shark/Models/Trees/KDTree.h>
#include <shark/Models/NearestNeighborClassifier.h>
#include <shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h>
//###end<NN-includes>


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

//###begin<NN>
	unsigned int k = 3;   // number of neighbors
	KDTree<RealVector> tree(traindata.inputs());
	TreeNearestNeighbors<RealVector, unsigned int> algorithm(traindata, &tree);
	NearestNeighborClassifier<RealVector> model(&algorithm, k);
//###end<NN>

//###begin<skeleton>
	Data<unsigned int> prediction = model(testdata.inputs());

	ZeroOneLoss<unsigned int> loss;
	double error_rate = loss(testdata.labels(), prediction);

	std::cout << "model: " << model.name() << std::endl
		<< "test error rate: " << error_rate << std::endl;
}
//###end<skeleton>
