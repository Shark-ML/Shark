//###begin<includes>
//noisy AutoencoderModel model and deep network
#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers and concatenating noise model
#include <shark/Unsupervised/RBM/BinaryRBM.h> // model for unsupervised pre-training

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
				if(random::coinToss(random::globalRng, 0.3))
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

//training of an RBM
BinaryRBM trainRBM(
	UnlabeledData<RealVector> const& data,//the data to train with
	std::size_t numHidden,//number of features in the AutoencoderModel
	std::size_t iterations, //number of iterations to optimize
	double regularisation,//strength of the regularisation
	double learningRate // learning rate of steepest descent
){
	//create rbm with simple binary units using the global random number generator
	std::size_t inputs = dataDimension(data);
	BinaryRBM rbm(random::globalRng);
	rbm.setStructure(inputs,numHidden);
	initRandomUniform(rbm,-0.1*std::sqrt(1.0/inputs),0.1*std::sqrt(1.0/inputs));//initialize weights uniformly
	
	//create derivative to optimize the rbm
	//we want a simple vanilla CD-1.
	BinaryCD estimator(&rbm);
	TwoNormRegularizer<> regularizer;
	//0.0 is the regularization strength. 0.0 means no regularization. choose as >= 0.0
	estimator.setRegularizer(regularisation,&regularizer);
	estimator.setK(1);//number of sampling steps
	estimator.setData(data);//the data used for optimization

	//create and configure optimizer
	SteepestDescent<> optimizer;
	optimizer.setLearningRate(learningRate);//learning rate of the algorithm
	
	//now we train the rbm and evaluate the mean negative log-likelihood at the end
	unsigned int numIterations = iterations;//iterations for training
	estimator.init();
	optimizer.init(estimator);
	for(unsigned int iteration = 0; iteration != numIterations; ++iteration) {
		optimizer.step(estimator);
	}
	rbm.setParameterVector(optimizer.solution().point);
	return rbm;
}


int main()
{
	//model parameters
	std::size_t numHidden1 = 8;
	std::size_t numHidden2 = 8;
	//unsupervised hyper parameters
	double unsupRegularisation = 0.001;
	double unsupLearningRate = 0.1;
	std::size_t unsupIterations = 10000;
	//supervised hyper parameters
	double regularisation = 0.0001;
	std::size_t iterations = 200;
	
	//load data and split into training and test
	LabeledData<RealVector,unsigned int> data = createProblem();
	data.shuffle();
	LabeledData<RealVector,unsigned int> test = splitAtElement(data,static_cast<std::size_t>(0.5*data.numberOfElements()));
	
//###begin<pretraining_rbm>
	//train the first hidden layer
	std::cout<<"pre-training first layer"<<std::endl;
	BinaryRBM rbm1 =  trainRBM(
		data.inputs(),numHidden1,
		unsupRegularisation,unsupIterations, unsupLearningRate
	);
	
	//compute the mapping onto features of the first hidden layer
	rbm1.evaluationType(true,true);//we compute the direction visible->hidden and want the features and no samples
	UnlabeledData<RealVector> intermediateData=rbm1(data.inputs());
	
	//train the next layer
	std::cout<<"pre-training second layer"<<std::endl;
	BinaryRBM rbm2 =  trainRBM(
		intermediateData,numHidden2,
		unsupRegularisation,unsupIterations, unsupLearningRate
	);
	
//###end<pretraining_rbm>

//###begin<pretraining_creation>
	//build three layer neural network from the re-trained RBMs
	LinearModel<RealVector, LogisticNeuron> layer1(rbm1.weightMatrix(),rbm1.hiddenNeurons().bias());
	LinearModel<RealVector, LogisticNeuron> layer2(rbm2.weightMatrix(),rbm2.hiddenNeurons().bias());
	LinearModel<RealVector> output(layer2.outputShape(),numberOfClasses(data));
	initRandomNormal(output,0.01);
	auto network = layer1 >> layer2 >> output;
//###end<pretraining_creation>
	
//###begin<supervised_training>
	//create the supervised problem. Cross Entropy loss with two norm regularisation
	CrossEntropy<unsigned int, RealVector> loss;
	ErrorFunction<> error(data, &network, &loss);
	TwoNormRegularizer<> regularizer(error.numberOfVariables());
	error.setRegularizer(regularisation,&regularizer);
	
	//optimize the model
	std::cout<<"training supervised model"<<std::endl;
	Rprop<> optimizer;
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
	cout << "classification error,train: " << loss01.eval(data.labels(), predictionTrain) << endl;
	
	Data<RealVector> prediction = network(test.inputs());
	cout << "classification error,test: " << loss01.eval(test.labels(), prediction) << endl;
	
}
