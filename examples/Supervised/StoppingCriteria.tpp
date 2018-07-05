//###begin<includes>
#include <shark/Data/Csv.h>
#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers to a feed forward neural network
#include <shark/Algorithms/GradientDescent/Rprop.h> //Optimization algorithm
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h> //Loss used for training
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> //The real loss for testing.
#include <shark/Algorithms/Trainers/OptimizationTrainer.h> // Trainer wrapping iterative optimization
#include <shark/Algorithms/StoppingCriteria/MaxIterations.h> //A simple stopping criterion that stops after a fixed number of iterations
#include <shark/Algorithms/StoppingCriteria/TrainingError.h> //Stops when the algorithm seems to converge
#include <shark/Algorithms/StoppingCriteria/GeneralizationQuotient.h> //Uses the validation error to track the progress
#include <shark/Algorithms/StoppingCriteria/ValidatedStoppingCriterion.h> //Adds the validation error to the value of the point
//###end<includes>

#include <iostream>

using namespace shark;
using namespace std;

//this program demonstrates the effect of different stopping criteria on the performance of a neural network.
//###begin<experiment>
template<class T>
double experiment(
	AbstractModel<RealVector, RealVector>& network, 
	AbstractStoppingCriterion<T> & stoppingCriterion,
	ClassificationDataset const& trainingset, 
	ClassificationDataset const& testset
){
	initRandomUniform(network,-0.1,0.1);

	//The Cross Entropy maximises the activation of the cth output neuron 
	// compared to all other outputs for a sample with class c.
	CrossEntropy<unsigned int, RealVector> loss;

	//we use IRpropPlus for network optimization
	Rprop<> optimizer;
	
	//create an optimization trainer and train the model
	OptimizationTrainer<AbstractModel<RealVector, RealVector>,unsigned int > trainer(&loss, &optimizer, &stoppingCriterion);
	trainer.train(network, trainingset);
	
	//evaluate the performance on the test set using the classification loss we choose 0.5 as threshold since Logistic neurons have values between 0 and 1.
	
	ZeroOneLoss<unsigned int, RealVector> loss01(0.5);
	Data<RealVector> predictions = network(testset.inputs()); 
	return loss01(testset.labels(),predictions);
}
//###end<experiment>
int main(){
	//load the diabetes dataset shuffle its entries and split it in training, validation and test set.
	//###begin<load>
	ClassificationDataset data;
	importCSV(data, "data/diabetes.csv",LAST_COLUMN, ',');
	data = shuffle(data);
	ClassificationDataset test = splitAtElement(data,static_cast<std::size_t>(0.75*data.numberOfElements()));
	ClassificationDataset validation = splitAtElement(data,static_cast<std::size_t>(0.66*data.numberOfElements()));
	//###end<load>
	
	//###begin<network>
	LinearModel<RealVector,LogisticNeuron> layer1(inputDimension(data),10); 
	LinearModel<RealVector> layer2(10,numberOfClasses(data));
	ConcatenatedModel<RealVector> network = layer1 >> layer2;
	//###end<network>
	
	//simple stopping criterion which allows for n iterations (here n = 10,100,500)
	//###begin<max_iter>
	MaxIterations<> maxIterations(10);
	double resultMaxIterations1 = experiment(network, maxIterations,data,test);
	maxIterations.setMaxIterations(100);
	double resultMaxIterations2 = experiment(network, maxIterations,data,test);
	maxIterations.setMaxIterations(500);
	double resultMaxIterations3 = experiment(network, maxIterations,data,test);
	//###end<max_iter>
	
	//###begin<train_error>
	TrainingError<> trainingError(10,1.e-5);
	double resultTrainingError = experiment(network, trainingError,data,test);
	//###end<train_error>
	
	//for the validated stopping criteria we need to define an error function using the validation set
	//###begin<generalization_quotient>
	CrossEntropy<unsigned int, RealVector> loss;
	ErrorFunction<> validationFunction(validation,&network,&loss);
	GeneralizationQuotient<> generalizationQuotient(10,0.1);
	ValidatedStoppingCriterion validatedLoss(&validationFunction,&generalizationQuotient);
	double resultGeneralizationQuotient = experiment(network, validatedLoss,data,test);
	//###end<generalization_quotient>
	
	//print the results
	//###begin<output>
	cout << "RESULTS: " << endl;
	cout << "======== \n" << endl;
	cout << "10 iterations   : " << resultMaxIterations1 << endl;
	cout << "100 iterations : " << resultMaxIterations2 << endl;
	cout << "500 iterations : " << resultMaxIterations3 << endl;
	cout << "training Error : " << resultTrainingError << endl;
	cout << "generalization Quotient : " << resultGeneralizationQuotient << endl;
	//###end<output>
}
