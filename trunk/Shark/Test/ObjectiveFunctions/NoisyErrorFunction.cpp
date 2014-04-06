#include <shark/Algorithms/GradientDescent/SteepestDescent.h>
#include <shark/ObjectiveFunctions/NoisyErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Models/LinearModel.h>
#include <shark/Rng/GlobalRng.h>

#define BOOST_TEST_MODULE ML_NoisyErrorFunction
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;
using namespace std;

struct TestFunction : public SingleObjectiveFunction
{
	typedef SingleObjectiveFunction Base;

	std::string name() const
	{ return "TestFunction"; }

	RealVector weights;
	TestFunction():weights(3){
		weights(0)=1;
		weights(1)=2;
		weights(2)=-1;
	}
	std::size_t numberOfVariables()const{
		return 3;
	}
	// adds just a value c on the input
	virtual double eval(RealVector const& pattern)const
	{
		return inner_prod(weights,pattern);
	}
};


BOOST_AUTO_TEST_CASE( ML_NoisyErrorFunction )
{
	//create regression data from the testfunction
	TestFunction function;
	std::vector<RealVector> data;
	std::vector<RealVector> target;
	RealVector input(3);
	RealVector output(1);

	for (size_t i=0; i<1000; i++)
	{
		for(size_t j=0;j!=3;++j)
		{
			input(j)=Rng::uni(-1,1);
		}
		data.push_back(input);
		output(0)=function.eval(input);
		target.push_back(output);
	}

	RegressionDataset dataset = createLabeledDataFromRange(data,target);

	//startingPoint
	RealVector point(3);
	point(0) = 0;
	point(1) = 0;
	point(2) = 0;
	SteepestDescent optimizer;
	SquaredLoss<> loss;
	LinearModel<> model(3);
	// batchsize 1 corresponds to stochastic gradient descent
	NoisyErrorFunction<> mse(&model,&loss,1);
	mse.setDataset(dataset);
	optimizer.init(mse, point);
	// train the cmac
	double error = 0.0;
	for (size_t iteration=0; iteration<501; ++iteration){
		optimizer.step(mse);
		if (iteration % 100 == 0){
			error = optimizer.solution().value;
			RealVector best = optimizer.solution().point;
			std::cout << iteration << " error:" << error << " parameter:" << best << std::endl;
		}
	}
	cout << "Optimization done. Error:" << error << std::endl;
	BOOST_CHECK_SMALL(error, 1.e-15);
}
