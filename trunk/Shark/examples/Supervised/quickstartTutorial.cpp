#include <shark/Data/Csv.h>
#include <shark/Algorithms/Trainers/LDA.h>
#include <iostream>

using namespace shark;
using namespace std;

int main(){
	//create a Dataset from the file "quickstartData"
	ClassificationDataset data;
	import_csv(data, "data/quickstartData.csv", LAST_COLUMN, " ");
	
	//create a test and training partition of the data
	ClassificationDataset test = splitAtElement(data,static_cast<std::size_t>(0.8*data.numberOfElements()));

	//create a classifier for the problem
	LinearClassifier classifier;
	//create the lda trainer
	LDA lda;
	//train the classifier using the training portion of the Data
	lda.train(classifier,data);


	//now use the test data to evaluate the model
	unsigned int correct = 0;
	//loop over all points of the test set
	//be aware that in this example a single point consists of an input and a label
	//this code here is just for illustration purposes
	BOOST_FOREACH(ClassificationDataset::element_reference point, test.elements()){
		unsigned int result = classifier(point.input);
		if (result == point.label){
			correct++;
		}
	}

	//print results
	cout << "RESULTS: " << endl;
	cout << "========\n" << endl;
	cout << "test data size: " << test.numberOfElements() << endl;
	cout << "correct classification: "<< correct << endl;
	cout << "error rate: " << 1.0 - double(correct)/test.numberOfElements() << endl;
}
