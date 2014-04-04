//###begin<includes>
#include <shark/Data/Csv.h>
#include <shark/Algorithms/Trainers/LDA.h>
using namespace shark;
//###end<includes>


#include <iostream>
using namespace std;


int main(int argc, char **argv){
	//create a Dataset from the file "quickstartData"
	//###begin<load_data>
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
	//###end<load_data>
	
	//create a test and training partition of the data
	//###begin<split_data>
	ClassificationDataset test = splitAtElement(data,static_cast<std::size_t>(0.8*data.numberOfElements()));
	//###end<split_data>
	
	//###begin<objects>
	//create a classifier for the problem
	LinearClassifier<> classifier;
	//create the lda trainer
	LDA lda;
	//###end<objects>
	//train the classifier using the training portion of the Data
	//###begin<train>
	lda.train(classifier,data);
	//###end<train>


	//now use the test data to evaluate the model
	//loop over all points of the test set
	//be aware that in this example a single point consists of an input and a label
	//this code here is just for illustration purposes
	//###begin<eval>
	unsigned int correct = 0;
	BOOST_FOREACH(ClassificationDataset::element_reference point, test.elements()){
		unsigned int result = classifier(point.input);
		if (result == point.label){
			correct++;
		}
	}
	//###end<eval>
	
	//print results
	//###begin<outputs>
	cout << "RESULTS: " << endl;
	cout << "========\n" << endl;
	cout << "test data size: " << test.numberOfElements() << endl;
	cout << "correct classification: "<< correct << endl;
	cout << "error rate: " << 1.0 - double(correct)/test.numberOfElements() << endl;
	//###end<outputs>
}
