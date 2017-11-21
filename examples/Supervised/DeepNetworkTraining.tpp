//###begin<includes>
//the model
#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers
//training the  model
#include <shark/ObjectiveFunctions/ErrorFunction.h>//error function, allows for minibatch training
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h> // loss used for supervised training
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> // loss used for evaluation of performance
#include <shark/Algorithms/GradientDescent/Adam.h> //optimizer: simple gradient descent.
#include <shark/Data/SparseData.h> //optimizer: simple gradient descent.

using namespace shark;
//###end<includes>

int main(int argc, char **argv)
{
	if(argc < 2) {
		std::cerr << "usage: " << argv[0] << " path/to/mnist_subset.libsvm" << std::endl;
		return 1;
	}
	std::size_t hidden1 = 200;
	std::size_t hidden2 = 100;
	std::size_t iterations = 10000;
	
	LabeledData<RealVector,unsigned int> data;
	importSparseData( data, argv[1] );
	data.shuffle(); //shuffle data randomly
	auto test = splitAtElement(data, 70 * data.numberOfElements() / 100);//split a test set
	std::size_t inputs = inputDimension(data);
	std::size_t numClasses = numberOfClasses(data);
	
//###begin<model_creation>
	//We use a dense linear model with rectifier activations
	typedef LinearModel<RealVector, RectifierNeuron> DenseLayer;
	
	//build the network
	DenseLayer layer1(inputs,hidden1);
	DenseLayer layer2(layer1.outputShape(),hidden2);
	LinearModel<RealVector> output(layer2.outputShape(),numClasses);
	auto network = layer1 >> layer2 >> output;
//###end<model_creation>
//###begin<supervised_training>	
	//create the supervised problem. 
	CrossEntropy loss;
	ErrorFunction error(data, &network, &loss, true);//enable minibatch training
	
	//optimize the model
	std::cout<<"training network"<<std::endl;
	Adam optimizer;
	error.init();
	optimizer.init(error);
	for(std::size_t i = 0; i != iterations; ++i){
		optimizer.step(error);
		std::cout<<i<<" "<<optimizer.solution().value<<std::endl;
	}
	network.setParameterVector(optimizer.solution().point);
//###end<supervised_training>
	
	//evaluation
	ZeroOneLoss<unsigned int,RealVector> loss01;
	Data<RealVector> predictionTrain = network(data.inputs());
	std::cout << "classification error,train: " << loss01.eval(data.labels(), predictionTrain) << std::endl;
	
	Data<RealVector> prediction = network(test.inputs());
	std::cout << "classification error,test: " << loss01.eval(test.labels(), prediction) << std::endl;
	
}
