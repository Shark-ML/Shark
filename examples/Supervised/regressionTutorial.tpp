//###begin<includes>
#include <shark/Data/Csv.h>
#include <shark/Algorithms/GradientDescent/CG.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Models/LinearModel.h>
//###end<includes>

#include <iostream>

using namespace shark;
using namespace std;


//loads a pair of files
//###begin<load>
RegressionDataset loadData(const std::string& dataFile,const std::string& labelFile){
	//we first load two separate data files for the trianing inputs and the labels of the data point
	Data<RealVector> inputs;
	Data<RealVector> labels;
	import_csv(inputs, dataFile, "\t");
	import_csv(labels, labelFile, "\t");
	//now we create a complete dataset which represents pairs of inputs and labels.
	RegressionDataset data(inputs, labels);
	return data;
}
//###end<load>

int main(){
	//load some data set and split a test set from the dataset. The first 80% of data points are training data.
	//###begin<split>
	RegressionDataset data = loadData("data/regressionInputs.csv","data/regressionLabels.csv");
	RegressionDataset test = splitAtElement(data,static_cast<std::size_t>(0.8*data.numberOfElements()));
	//###end<split>
	
	//a linear model with as many in and outputs as the data has
	//###begin<model>
	LinearModel<> model(inputDimension(data), labelDimension(data));
	//###end<model>
	
	//the squared loss can be used to calculate the mean squared error of the data and the model
	//the ErrorFunction brings model, loss and data together and so automates evaluation
	//###begin<error_function>
	SquaredLoss<> loss;
	ErrorFunction<RealVector,RealVector> errorFunction(&model,&loss);
	errorFunction.setDataset(data);
	//###end<error_function>

	//###begin<optimize>
	CG optimizer;
	optimizer.init(errorFunction);
	for(int i = 0; i != 100; ++i)
	{
		optimizer.step(errorFunction);
	}
	//###end<optimize>
	
	//save training error
	double trainingError = optimizer.solution().value;

	//evaluate test error
	//###begin<test_error>
	model.setParameterVector(optimizer.solution().point);
	Data<RealVector> predictions = model(test.inputs());
	double testError = loss.eval(test.labels(),predictions);
	//###end<test_error>
	
	//print the results
	//###begin<output>
	cout << "RESULTS: " << endl;
	cout << "======== \n" << endl;
	cout << "training error " << trainingError << endl;
	cout << "test error: " << testError << endl;
	//###end<output>
}
