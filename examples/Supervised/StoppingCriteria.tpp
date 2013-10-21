//###begin<includes>
#include <shark/Data/Csv.h>
#include <shark/Models/FFNet.h> //Feed forward neural network class
#include <shark/Algorithms/GradientDescent/Rprop.h> //Optimization algorithm
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h> //Loss used for training
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> //The real loss for testing.
#include <shark/ObjectiveFunctions/ErrorFunction.h> //The usual error function
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
double experiment(AbstractStoppingCriterion<T> & stoppingCriterion, ClassificationDataset const& trainingset, ClassificationDataset const& testset){
	//create a feed forward neural network with one layer of 10 hidden neurons and one output for every class
	FFNet<LogisticNeuron,LinearNeuron> network;
	network.setStructure(inputDimension(trainingset),10,numberOfClasses(trainingset));
	initRandomUniform(network,-0.1,0.1);

	//The Cross Entropy maximises the activation of the cth output neuron compared to all other outputs for a sample with class c.
	CrossEntropy loss;
	ErrorFunction<RealVector,unsigned int> errorFunction(&network,&loss);

	//we use IRpropPlus for network optimization
	IRpropPlus optimizer;
	
	//create an optimization trainer and train the model
	OptimizationTrainer<FFNet<LogisticNeuron,LinearNeuron>,unsigned int > trainer(&errorFunction, &optimizer, &stoppingCriterion);
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
	import_csv(data, "data/diabetes.csv",LAST_COLUMN, ' ');
	data.shuffle();
	ClassificationDataset test = splitAtElement(data,static_cast<std::size_t>(0.75*data.numberOfElements()));
	ClassificationDataset validation = splitAtElement(data,static_cast<std::size_t>(0.66*data.numberOfElements()));
	//###end<load>
	
	//simple stopping criterion which allows for n iterations (here n = 10,100,500)
	//###begin<max_iter>
	MaxIterations<> maxIterations(10);
	double resultMaxIterations1 = experiment(maxIterations,data,test);
	maxIterations.setMaxIterations(100);
	double resultMaxIterations2 = experiment(maxIterations,data,test);
	maxIterations.setMaxIterations(500);
	double resultMaxIterations3 = experiment(maxIterations,data,test);
	//###end<max_iter>
	
	//###begin<train_error>
	TrainingError<> trainingError(10,1.e-5);
	double resultTrainingError = experiment(trainingError,data,test);
	//###end<train_error>
	
	//for the validated stopping criteria we need to define an error function using the validation set
	//###begin<generalization_quotient>
	FFNet<LogisticNeuron,LogisticNeuron> network;
	network.setStructure(inputDimension(data),10,numberOfClasses(data));
	CrossEntropy loss;
	ErrorFunction<RealVector,unsigned int> validationFunction(&network,&loss);
	validationFunction.setDataset(validation);
	
	
	GeneralizationQuotient<> generalizationQuotient(10,0.1);
	ValidatedStoppingCriterion validatedLoss(&validationFunction,&generalizationQuotient);
	double resultGeneralizationQuotient = experiment(validatedLoss,data,test);
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
