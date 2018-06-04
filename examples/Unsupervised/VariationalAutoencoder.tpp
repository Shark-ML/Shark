//###begin<includes>
#include <shark/Data/SparseData.h>//for reading in the images as sparseData/Libsvm format
#include <shark/Data/Pgm.h>//for printing out reconstructions
#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers
#include <shark/Algorithms/GradientDescent/Adam.h>// The Adam optimization algorithm
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h> //squared loss function (can also be cross-entropy for greyscale images)
#include <shark/ObjectiveFunctions/VariationalAutoencoderError.h> //variational autoencoder error function
using namespace shark;
//###end<includes>

int main(int argc, char **argv)
{
	if(argc < 2) {
		std::cerr << "usage: " << argv[0] << " path/to/mnist_subset.libsvm" << std::endl;
		return 1;
	}
	
	//Step1: load data
//###begin<data>
	LabeledData<FloatVector,unsigned int> data;
	importSparseData( data, argv[1] , 784 , 50);
//###end<data>
	
	//Step 2: define model
//###begin<model_creation>
	//build encoder network
	//note that the output layer must be linear and must twice the number of outputs than the decoder inptus
	//as we have to model mean and variance for each decoder-input.
	LinearModel<FloatVector, RectifierNeuron> encoder1(data.inputShape(),500, true);
	LinearModel<FloatVector, LinearNeuron> encoder2(encoder1.outputShape(),2 * 300, true);
	auto encoder = encoder1 >> encoder2;
	
	//build decoder network
	//mnist is scaled between 0 and 1 soa  sigmoid output makes prediciton compelte black and complete white pixels easier
	LinearModel<FloatVector, RectifierNeuron> decoder1(300, 500, true);
	LinearModel<FloatVector, LogisticNeuron> decoder2(decoder1.outputShape(), data.inputShape(), true);
	auto decoder = decoder1 >> decoder2;
//###end<model_creation>
	
//###begin<objfunct>	
	SquaredLoss<FloatVector> loss;
	double lambda = 1.0;
	VariationalAutoencoderError<FloatVector> error(data.inputs(), &encoder, &decoder,&loss, lambda);
//###end<objfunct>	
	
	//Step 4 set up optimizer and run optimization
//###begin<training>	
	std::size_t iterations = 20000;
	Adam<FloatVector> optimizer;
	optimizer.setEta(0.001);
	initRandomNormal(encoder,0.0001);
	initRandomNormal(decoder,0.0001);
	error.init();
	optimizer.init(error);
	std::cout<<"Optimizing model "<<std::endl;
	for(std::size_t i = 0; i <= iterations; ++i){
		optimizer.step(error);
		if(i % 100 == 0){
			//create some reconstructions for evaluation
			auto const& batch = data.batch(0).input;
			RealMatrix reconstructed = decoder(error.sampleZ(optimizer.solution().point, batch));
			
			std::cout<<i<<" "<<optimizer.solution().value<<" "<<loss(batch, reconstructed)/batch.size1()<<std::endl;
			//store reconstructions
			exportFiltersToPGMGrid("reconstructed"+std::to_string(i), reconstructed,28,28);
		}
	}
//###end<training>
}
