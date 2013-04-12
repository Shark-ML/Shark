#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/Algorithms/DirectSearch/CMA.h>
#include <shark/ObjectiveFunctions/DenoisingAutoencoderError.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Models/LinearModel.h>
#include <shark/Rng/GlobalRng.h>

#define BOOST_TEST_MODULE ObjectiveFunction_DenoisingAutoencoderError
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;
using namespace std;


double resultParams[]=
{0.5,0.25,0.166667,
 1  ,0.5 ,0.333333,
 1.5,0.75,0.5};


BOOST_AUTO_TEST_CASE( ObjectiveFunction_DenoisedAutoencoderError_NoNoise )
{
	//create regression data from the testfunction
	std::vector<RealVector> data;
	RealVector input(3);

	for (size_t i=0; i<1000; i++)
	{
		input(0)=Rng::uni(-1,1);
		input(1)=2*input(0);
		input(2)=3*input(0);
		data.push_back(input);
	}

	UnlabeledData<RealVector> dataset = createDataFromRange(data);
	IRpropPlus optimizer;
	SquaredLoss<> loss;
	LinearModel<> model(3,3);
	initRandomNormal(model,0.1);
	// batchsize 1 corresponds to stochastic gradient descent
	DenoisingAutoencoderError<> mse(&model,&loss);
	mse.setData(dataset);
	optimizer.init(mse);
	// train the cmac
	double error = 0.0;
	for (size_t iteration=0; iteration<301; ++iteration){
		optimizer.step(mse);
		if (iteration % 100 == 0){
			error = optimizer.solution().value;
			RealVector best = optimizer.solution().point;
			std::cout << iteration << " error:" << error << " parameter:" << best << std::endl;
		}
	}
	cout << "Optimization done for k=0. Error:" << error << std::endl;
	BOOST_CHECK_SMALL(error, 1.e-15);
}

BOOST_AUTO_TEST_CASE( ObjectiveFunction_DenoisedAutoencoderError_Noise_CMA )
{
	//create regression data from the testfunction
	std::vector<RealVector> data;
	RealVector input(3);

	for (size_t i=0; i<1000; i++)
	{
		input(0)=Rng::uni(-1,1);
		input(1)=2*input(0);
		input(2)=3*input(0);
		data.push_back(input);
	}

	UnlabeledData<RealVector> dataset = createDataFromRange(data);
	CMA optimizer;
	SquaredLoss<> loss;
	LinearModel<> model(3,3);
	initRandomNormal(model,0.1);
	// batchsize 1 corresponds to stochastic gradient descent
	DenoisingAutoencoderError<> mse(&model,&loss,1);
	mse.setData(dataset);
	optimizer.init(mse);
	// train the cmac
	double error = 0.0;
	for (size_t iteration=0; iteration<301; ++iteration){
		optimizer.step(mse);
		if (iteration % 100 == 0){
			error = optimizer.solution().value;
			RealVector best = optimizer.solution().point;
			std::cout << iteration << " error:" << error << " parameter:" << best << std::endl;
		}
	}
	cout << "Optimization done for k=1. Error:" << error << std::endl;
	BOOST_CHECK_SMALL(error, 1.e-15);
	for(std::size_t i = 0; i != 9; ++i){
		BOOST_CHECK_SMALL(optimizer.solution().point(i)-resultParams[i],1.e-6);
	}
}
BOOST_AUTO_TEST_CASE( ObjectiveFunction_DenoisedAutoencoderError_Noise_IRprop )
{
	//create regression data from the testfunction
	std::vector<RealVector> data;
	RealVector input(3);

	for (size_t i=0; i<1000; i++)
	{
		input(0)=Rng::uni(-1,1);
		input(1)=2*input(0);
		input(2)=3*input(0);
		data.push_back(input);
	}

	UnlabeledData<RealVector> dataset = createDataFromRange(data);
	IRpropPlus optimizer;
	SquaredLoss<> loss;
	LinearModel<> model(3,3);
	initRandomNormal(model,0.1);
	// batchsize 1 corresponds to stochastic gradient descent
	DenoisingAutoencoderError<> mse(&model,&loss,1);
	mse.setData(dataset);
	optimizer.init(mse);
	// train the cmac
	double error = 0.0;
	for (size_t iteration=0; iteration<301; ++iteration){
		optimizer.step(mse);
		if (iteration % 100 == 0){
			error = optimizer.solution().value;
			RealVector best = optimizer.solution().point;
			std::cout << iteration << " error:" << error << " parameter:" << best << std::endl;
		}
	}
	cout << "Optimization done for k=1. Error:" << error << std::endl;
	BOOST_CHECK_SMALL(error, 1.e-15);
	for(std::size_t i = 0; i != 9; ++i){
		BOOST_CHECK_SMALL(optimizer.solution().point(i)-resultParams[i],1.e-6);
	}
}
