#include <shark/Data/Csv.h> //load the csv file
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h> //normalize

#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h> //normalize

#include<shark/Models/RBFLayer.h>
#include<shark/Models/ConvexCombination.h>
#include<shark/Models/ConcatenatedModel.h>

#include<shark/ObjectiveFunctions/NegativeLogLikelihood.h>
#include<shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/Algorithms/KMeans.h>

using namespace shark;
using namespace std;


int main(int argc, char **argv) {
	if(argc < 2) {
		cerr << "usage: " << argv[0] << " (filename)" << endl;
		return 1;
	}
	UnlabeledData<RealVector> data;
	try {
		importCSV(data, argv[1], ' ');
		Normalizer<> normalizer;
		NormalizeComponentsUnitVariance<> normalizingTrainer(true); // zero-mean
		normalizingTrainer.train(normalizer, data);
		data = normalizer(data);
	} 
	catch (...) {
		cerr << "unable to read data from file " <<  argv[1] << endl;
		return 1;
	}
	
	std::size_t inputs=dataDimension(data);
	std::size_t hiddens = 2;
	unsigned numberOfSteps = 100;

	//create mixture of gaussian mixture and initialize weights
	RBFLayer gaussians;
	gaussians.setStructure(inputs,hiddens);
	ConvexCombination combination;
	combination.setStructure(hiddens);
	ConcatenatedModel<RealVector,RealVector> mixture= gaussians >> combination;
	initRandomUniform(mixture,-0.1,0.1);
	//kMeans(data, gaussians);
	
	//create error function
	NegativeLogLikelihood error(&mixture,data);
	//initialize Rprop
	IRpropPlus optimizer;
	optimizer.init(error);
	
	RealMatrix initCenters = gaussians.centers();
	RealVector initWidth = gaussians.gamma();
	
	std::cout<<0<<" "<<optimizer.solution().value<<std::endl;;
	for(unsigned step = 0; step != numberOfSteps; ++step){
		optimizer.step(error);
		if(step % 10 == 9)
			std::cout<<step<<" "<<optimizer.solution().value<<std::endl;;
	}
	
	std::cout<<"center solutions"<<std::endl;
	std::cout<<"init:"<<initCenters<<std::endl;
	std::cout<<"after optimization:"<<gaussians.centers()<<std::endl;
	std::cout<<"width solutions"<<std::endl;
	std::cout<<"init:"<<initWidth<<std::endl;
	std::cout<<"after optimization:"<<gaussians.gamma()<<std::endl;
}
