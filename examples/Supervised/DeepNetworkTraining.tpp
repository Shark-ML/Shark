//###begin<includes>
//noisy autoencoder model and deep network
#include <shark/Models/FFNet.h>// neural network as autoencoder
#include <shark/Models/GaussianNoiseModel.h>// model adding noise to the inputs
#include <shark/Models/ConcatenatedModel.h>// to concatenate autoencoder with noise adding model

//training the  model
#include <shark/ObjectiveFunctions/ErrorFunction.h>//the error function performing the regularisation of the hidden neurons
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h> // squared loss used for unsupervised pre-training
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h> // loss used for supervised training
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> // loss used for evaluation of performance
#include <shark/ObjectiveFunctions/Regularizer.h> //L1 and L2 regularisation
#include <shark/Algorithms/StoppingCriteria/TrainingError.h> //stopping criterion for learning
#include <shark/Algorithms/GradientDescent/SteepestDescent.h> //optimizer: simple gradient descent.
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

//###begin<autoencoder_signature>
//the type of neural network architecture we are going to use
typedef FFNet<RectifierNeuron,LinearNeuron> Network;

//training of an auto encoder with one hidden layer
Network trainDenoisingAutoencoder(
	UnlabeledData<RealVector> const& data,//the data to train with
	std::size_t numHidden,//number of features in the autoncoder
	double regularisation, double noiseVariance,//L2 regularisation and noise strength
	double learningRate, double momentum//parameters for steepest descent
){
//###end<autoencoder_signature>
	std::size_t inputs = dataDimension(data);
	
	//create the model
//###begin<autoencoder_model>
	Network model;
	model.setStructure(inputs, numHidden, inputs,FFNetStructures::Normal, true);
	GaussianNoiseModel noise(inputs,noiseVariance);
	ConcatenatedModel<RealVector,RealVector> denoisingAutoencoder = noise >> model;
	initRandomNormal(denoisingAutoencoder,0.1);
//###end<autoencoder_model>
	
	//create the objective function
//###begin<autoencoder_error>
	LabeledData<RealVector,RealVector> trainSet(data,data);//labels identical to inputs
	SquaredLoss<RealVector> loss;
	ErrorFunction<RealVector,RealVector> error(trainSet, &denoisingAutoencoder, &loss);
	TwoNormRegularizer regularizer(error.numberOfVariables());
	error.setRegularizer(regularisation,&regularizer);
//###end<autoencoder_error>
	
	//set up optimizer
//###begin<autoencoder_optimization>
	SteepestDescent optimizer;
	optimizer.setLearningRate(learningRate);
	optimizer.setMomentum(momentum);
	
	//optimize the model
	optimizer.init(error);
	TrainingError<> stoppingCriterion(1000,0.1);//stop if the relative improvement of the error is smaller than 1.e-5 ina  strip of100 iterations
	do{
		optimizer.step(error);
//###end<autoencoder_optimization>
		if(error.evaluationCounter() % 200 == 0){
			std::cout<<error.evaluationCounter()<<" "<<optimizer.solution().value<<std::endl;
		}
//###begin<autoencoder_optimization>
	}while(!stoppingCriterion.stop(optimizer.solution()));
	denoisingAutoencoder.setParameterVector(optimizer.solution().point);
	return model;
//###end<autoencoder_optimization>
	
}

//unsupervised pre training of a network with two hidden layers
//###begin<pretraining_autoencoder>
Network unsupervisedPreTraining(
	UnlabeledData<RealVector> const& data,
	std::size_t numHidden1,std::size_t numHidden2, std::size_t numOutputs,
	double regularisation, double noiseVariance, double learningRate, double momentum
){
	//train the first hidden layer
	std::cout<<"training first layer"<<std::endl;
	Network layer =  trainDenoisingAutoencoder(
		data,numHidden1,
		regularisation, noiseVariance,
		learningRate, momentum
	);
	//compute the mapping onto the features of the first hidden layer
	UnlabeledData<RealVector> intermediateData = layer.evalLayer(0,data);
	
	//train the next layer
	std::cout<<"training second layer"<<std::endl;
	Network layer2 =  trainDenoisingAutoencoder(
		intermediateData,numHidden2,
		regularisation, noiseVariance,
		learningRate, momentum
	);
//###end<pretraining_autoencoder>
//###begin<pretraining_creation>
	//create the final network
	Network network;
	network.setStructure(dataDimension(data),numHidden1,numHidden2, numOutputs);
	initRandomNormal(network,0.1);
	network.setLayer(0,layer.layerMatrix(0),layer.bias(0));
	network.setLayer(1,layer2.layerMatrix(0),layer2.bias(0));
	
	return network;
//###end<pretraining_creation>
}

int main()
{
//###begin<supervised_training>
	//model parameters
	std::size_t numHidden1 = 4;
	std::size_t numHidden2 = 4;
	//unsupervised hyper parameters
	double unsupRegularisation = 0.001;
	double noiseVariance = 0.2;
	double unsupMomentum = 0.3;
	double unsupLearningRate = 0.1*(1-unsupMomentum);
	//supervised hyper parameters
	double regularisation = 0.0001;
	double momentum = 0.3;
	double learningRate = 0.1*(1-unsupMomentum);
	
	//load data and split into training and test
	LabeledData<RealVector,unsigned int> data = createProblem();
	data.shuffle();
	LabeledData<RealVector,unsigned int> test = splitAtElement(data,static_cast<std::size_t>(0.5*data.numberOfElements()));
	
	//unsupervised pre training
	Network network = unsupervisedPreTraining(
		data.inputs(),numHidden1, numHidden2,numberOfClasses(data),
		unsupRegularisation, noiseVariance, unsupLearningRate, unsupMomentum
	);
	
	//create the supervised problem. Cross Entropy loss with one norm regularisation
	CrossEntropy loss;
	ErrorFunction<RealVector,unsigned int> error(data, &network, &loss);
	OneNormRegularizer regularizer(error.numberOfVariables());
	error.setRegularizer(regularisation,&regularizer);
	
	//set up optimizer
	SteepestDescent optimizer;
	optimizer.setLearningRate(learningRate);
	optimizer.setMomentum(momentum);
	
	//optimize the model
	std::cout<<"training supervised model"<<std::endl;
	optimizer.init(error);
	TrainingError<> stoppingCriterion(1000,0.001);//stop if the relative improvement of the error is smaller than 1.e-5 ina  strip of100 iterations
	do{
		optimizer.step(error);
		if(error.evaluationCounter() % 200 == 0){
			std::cout<<error.evaluationCounter()<<" "<<optimizer.solution().value<<std::endl;
		}
	}while(!stoppingCriterion.stop(optimizer.solution()));
	network.setParameterVector(optimizer.solution().point);
//###end<supervised_training>
	
	//evaluation
	ZeroOneLoss<unsigned int,RealVector> loss01;
	Data<RealVector> predictionTrain = network(data.inputs());
	cout << "classification error,features: " << loss01.eval(data.labels(), predictionTrain) << endl;
	
	Data<RealVector> prediction = network(test.inputs());
	cout << "classification error,features: " << loss01.eval(test.labels(), prediction) << endl;
	
}
