//###begin<includes>
//noisy AutoencoderModel model and deep network
#include <shark/Models/FFNet.h>// neural network for supervised training
#include <shark/Models/Autoencoder.h>// the autoencoder to train unsupervised
#include <shark/Models/ImpulseNoiseModel.h>// model adding noise to the inputs
#include <shark/Models/ConcatenatedModel.h>// to concatenate Autoencoder with noise adding model

//training the  model
#include <shark/ObjectiveFunctions/ErrorFunction.h>//the error function performing the regularisation of the hidden neurons
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h> // squared loss used for unsupervised pre-training
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h> // loss used for supervised training
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> // loss used for evaluation of performance
#include <shark/ObjectiveFunctions/Regularizer.h> //L1 and L2 regularisation
#include <shark/Algorithms/GradientDescent/SteepestDescent.h> //optimizer: simple gradient descent.
#include <shark/Algorithms/GradientDescent/Rprop.h> //optimizer for autoencoders
//###end<includes>

using namespace std;
using namespace shark;

//our artificial problem
LabeledData<RealVector,unsigned int> createProblem(){
	std::vector<RealVector> data(320,RealVector(16));
	std::vector<unsigned int> label(320);
	RealVector line(4);
	for(std::size_t k = 0; k != 10; ++k){
		for(size_t x=0; x != 16; x++) {
			for(size_t j=0; j != 4; j++) {
				bool val = (x & (1<<j)) > 0;
				line(j) = val;
				if(Rng::coinToss(0.3))
					line(j) = !val;
			}

			for(int i=0; i != 4; i++) {
				subrange(data[x+k*16],i*4 ,i*4 + 4) = line;
			}
			for(int i=0; i != 4; i++) {
				for(int l=0; l<4; l++) {
					data[x+k*16+160](l*4 + i) = line(l);
				}
			}
			label[x+k*16] = 1; 
			label[x+k*16+160] = 0; 
		}
	}
	return createLabeledDataFromRange(data,label);
}

//training of an auto encoder with one hidden layer
//###begin<function>
template<class AutoencoderModel>
AutoencoderModel trainAutoencoderModel(
	UnlabeledData<RealVector> const& data,//the data to train with
	std::size_t numHidden,//number of features in the AutoencoderModel
	double regularisation,//strength of the regularisation
	double noiseStrength, // strength of the added noise
	std::size_t iterations //number of iterations to optimize
){
//###end<function>
//###begin<model>	
	//create the model
	std::size_t inputs = dataDimension(data);
	AutoencoderModel baseModel;
	baseModel.setStructure(inputs, numHidden);
	initRandomUniform(baseModel,-0.1*std::sqrt(1.0/inputs),0.1*std::sqrt(1.0/inputs));
	ImpulseNoiseModel noise(noiseStrength,0.0);//set an input pixel with probability p to 0
	ConcatenatedModel<RealVector,RealVector> model = noise>> baseModel;
//###end<model>	
//###begin<objective>		
	//create the objective function
	LabeledData<RealVector,RealVector> trainSet(data,data);//labels identical to inputs
	SquaredLoss<RealVector> loss;
	ErrorFunction<RealVector,RealVector> error(trainSet, &model, &loss);
	TwoNormRegularizer regularizer(error.numberOfVariables());
	error.setRegularizer(regularisation,&regularizer);
//###end<objective>	
	//set up optimizer
//###begin<optimizer>
	IRpropPlusFull optimizer;
	optimizer.init(error);
	std::cout<<"Optimizing model: "+model.name()<<std::endl;
	for(std::size_t i = 0; i != iterations; ++i){
		optimizer.step(error);
		std::cout<<i<<" "<<optimizer.solution().value<<std::endl;
	}
//###end<optimizer>
	model.setParameterVector(optimizer.solution().point);
	return baseModel;
}

//###begin<network_types>
typedef Autoencoder<RectifierNeuron,LinearNeuron> AutoencoderModel;//type of autoencoder
typedef FFNet<RectifierNeuron,LinearNeuron> Network;//final supervised trained structure
//###end<network_types>

//unsupervised pre training of a network with two hidden layers
//###begin<pretraining_autoencoder>
Network unsupervisedPreTraining(
	UnlabeledData<RealVector> const& data,
	std::size_t numHidden1,std::size_t numHidden2, std::size_t numOutputs,
	double regularisation, double noiseStrength, std::size_t iterations
){
	//train the first hidden layer
	std::cout<<"training first layer"<<std::endl;
	AutoencoderModel layer =  trainAutoencoderModel<AutoencoderModel>(
		data,numHidden1,
		regularisation, noiseStrength,
		iterations
	);
	//compute the mapping onto the features of the first hidden layer
	UnlabeledData<RealVector> intermediateData = layer.evalLayer(0,data);
	
	//train the next layer
	std::cout<<"training second layer"<<std::endl;
	AutoencoderModel layer2 =  trainAutoencoderModel<AutoencoderModel>(
		intermediateData,numHidden2,
		regularisation, noiseStrength,
		iterations
	);
//###end<pretraining_autoencoder>
//###begin<pretraining_creation>
	//create the final network
	Network network;
	network.setStructure(dataDimension(data),numHidden1,numHidden2, numOutputs);
	initRandomNormal(network,0.1);
	network.setLayer(0,layer.encoderMatrix(),layer.hiddenBias());
	network.setLayer(1,layer2.encoderMatrix(),layer2.hiddenBias());
	
	return network;
//###end<pretraining_creation>
}

int main()
{
//###begin<supervised_training>
	//model parameters
	std::size_t numHidden1 = 8;
	std::size_t numHidden2 = 8;
	//unsupervised hyper parameters
	double unsupRegularisation = 0.001;
	double noiseStrength = 0.3;
	std::size_t unsupIterations = 100;
	//supervised hyper parameters
	double regularisation = 0.0001;
	std::size_t iterations = 200;
	
	//load data and split into training and test
	LabeledData<RealVector,unsigned int> data = createProblem();
	data.shuffle();
	LabeledData<RealVector,unsigned int> test = splitAtElement(data,static_cast<std::size_t>(0.5*data.numberOfElements()));
	
	//unsupervised pre training
	Network network = unsupervisedPreTraining(
		data.inputs(),numHidden1, numHidden2,numberOfClasses(data),
		unsupRegularisation, noiseStrength, unsupIterations
	);
	
	//create the supervised problem. Cross Entropy loss with one norm regularisation
	CrossEntropy loss;
	ErrorFunction<RealVector,unsigned int> error(data, &network, &loss);
	OneNormRegularizer regularizer(error.numberOfVariables());
	error.setRegularizer(regularisation,&regularizer);
	
	//optimize the model
	std::cout<<"training supervised model"<<std::endl;
	IRpropPlusFull optimizer;
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
	cout << "classification error,train: " << loss01.eval(data.labels(), predictionTrain) << endl;
	
	Data<RealVector> prediction = network(test.inputs());
	cout << "classification error,test: " << loss01.eval(test.labels(), prediction) << endl;
	
}
