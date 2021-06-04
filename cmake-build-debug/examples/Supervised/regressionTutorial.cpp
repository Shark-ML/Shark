#include <shark/Data/Csv.h>
#include <shark/Algorithms/GradientDescent/CG.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Models/LinearModel.h>

#include <iostream>

using namespace shark;
using namespace std;


//loads a pair of files
RegressionDataset loadData(const std::string& dataFile,const std::string& labelFile){
	//we first load two separate data files for the training inputs and the labels of the data point
	Data<RealVector> inputs;
	Data<RealVector> labels;
	try {
		importCSV(inputs, dataFile, ' ');
		importCSV(labels, labelFile, ' ');
	} catch (...) {
		cerr << "Unable to open file " <<  dataFile << " and/or " << labelFile << ". Check paths!" << endl;
		exit(EXIT_FAILURE);
	}
	//now we create a complete dataset which represents pairs of inputs and labels
	RegressionDataset data(inputs, labels);
	return data;
}

int main(){
	//load some data set and split a test set from the dataset. The first 80% of data points are training data.
	RegressionDataset data = loadData("data/regressionInputs.csv","data/regressionLabels.csv");
	RegressionDataset test = splitAtElement(data,static_cast<std::size_t>(0.8*data.numberOfElements()));
	
	//a linear model with as many in and outputs as the data has
	LinearModel<> model(inputDimension(data), labelDimension(data));
	
	//the squared loss can be used to calculate the mean squared error of the data and the model
	//the ErrorFunction brings model, loss and data together and so automates evaluation
	SquaredLoss<> loss;
	ErrorFunction<> errorFunction(data, &model,&loss);

	CG<> optimizer;
	errorFunction.init();
	optimizer.init(errorFunction);
	for(int i = 0; i != 100; ++i)
	{
		optimizer.step(errorFunction);
	}
	//copy solution parameters into model
	model.setParameterVector(optimizer.solution().point);
	
	//save training error
	double trainingError = optimizer.solution().value;

	//evaluate test error
	Data<RealVector> predictions = model(test.inputs());
	double testError = loss.eval(test.labels(),predictions);
	
	//print the results
	cout << "RESULTS: " << endl;
	cout << "======== \n" << endl;
	cout << "training error " << trainingError << endl;
	cout << "test error: " << testError << endl;
}
