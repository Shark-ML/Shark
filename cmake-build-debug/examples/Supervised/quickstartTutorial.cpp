#include <shark/Data/Csv.h>
#include <shark/Algorithms/Trainers/LDA.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
using namespace shark;


#include <iostream>
using namespace std;


int main(int argc, char **argv){
	//create a Dataset from the file "quickstartData"
	if(argc < 2) {
		cerr << "usage: " << argv[0] << " (filename)" << endl;
		exit(EXIT_FAILURE);
	}
	ClassificationDataset data;
	try {
		importCSV(data, argv[1], LAST_COLUMN, ' ');
	} 
	catch (...) {
		cerr << "unable to read data from file " <<  argv[1] << endl;
		exit(EXIT_FAILURE);
	}
	
	//create a test and training partition of the data
	ClassificationDataset test = splitAtElement(data,static_cast<std::size_t>(0.8*data.numberOfElements()));
	
	//create a classifier for the problem
	LinearClassifier<> classifier;
	//create the lda trainer
	LDA lda;
	//train the classifier using the training portion of the Data
	lda.train(classifier,data);

	ZeroOneLoss<> loss;
	Data<unsigned int> predictions = classifier(test.inputs());
	double error = loss(test.labels(),predictions);
	
	//print results
	cout << "RESULTS: " << endl;
	cout << "========\n" << endl;
	cout << "test data size: " << test.numberOfElements() << endl;
	cout << "error rate: " << error << endl;
}
