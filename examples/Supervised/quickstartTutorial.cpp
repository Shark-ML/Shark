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
	ClassificationDataset test = splitAfterElement(data,static_cast<std::size_t>(0.8*data.numberOfElements()));

	//create a classifier for the problem
	LinearClassifier classifier;
	//create the lda trainer
	LDA lda;
	//train the classifier using the training portion of the Data
	lda.train(classifier,data);


	//now use the test data to evaluate the model
	unsigned int correct = 0;
	for (size_t i = 0; i != test.numberOfElements(); ++i ) {
		//operator() returns the result from the classifier given the i-th datapoint of the testset
		unsigned int result = classifier(test(i).input);
		if (result == test(i).label){
			correct++;
		}
	}

	//print results
	cout<<"RESULTS: "<<endl;
	cout<<"======== "<<endl << endl;
	cout<<"test data size: " << test.size() <<endl;
	cout<<"correct classification: "<< correct<<endl;
	cout<<"error rate: " << 1.0- double(correct)/test.numberOfElements()<<endl;
}
