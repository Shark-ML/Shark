#include <shark/Data/Csv.h>
#include <shark/Algorithms/Trainers/LinearRegression.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Core/Timer.h>
#include <iostream>
using namespace shark;
using namespace std;

int main(int argc, char **argv) {
	RegressionDataset data;
	importCSV(data, "blogData_train.csv", LAST_COLUMN,1,',','#', 2<<16);

	LinearRegression trainer;
	LinearModel<> model;
	
	Timer time;
	trainer.train(model, data);
	double time_taken = time.stop();

	SquaredLoss<> loss;
	cout << "Residual sum of squares:" << loss(data.labels(),model(data.inputs()))<<std::endl;
	cout << "Time:\n" << time_taken << endl;
	cout << time_taken << endl;
}