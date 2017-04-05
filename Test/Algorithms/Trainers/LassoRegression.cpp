#define BOOST_TEST_MODULE TRAINERS_LASSOREGRESSION
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/LassoRegression.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Core/Random.h>
#include <iostream>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_LassoRegression)

// The optimal linear regression weight vector for this problem is
// w = (0,0,0,0,0,1/2,0,0,0,0)
class TestProblem : public LabeledDataDistribution<RealVector, RealVector>
{
public:
	void draw(RealVector& input, RealVector& label) const
	{
		input.resize(10);
		label.resize(1);
		double g = random::gauss(random::globalRng);
		for (size_t i=0; i<10; i++) input(i) = random::gauss(random::globalRng);
		input(5) += g;
		label(0) = g;
	}
};

BOOST_AUTO_TEST_CASE(LassoRegression_TEST)
{
	// create a highly artificial test problem
	TestProblem problem;
	std::size_t n = 10000;
	LabeledData<RealVector, RealVector> data = problem.generateDataset(n);

	// train the model with high regularization to drive all
	// non-informative weights to exact zero
	double lambda = 1e3;
	LinearModel<RealVector> model(10);
	LassoRegression<RealVector> trainer(lambda);
	trainer.train(model, data);

	// check zero and non-zero coefficients
	RealMatrix m = model.matrix();
	double target = 0.5 * (1.0 - lambda / (double)n);   // correct for regularization
	for (size_t i=0; i<10; i++)
	{
		double value = m(0, i);
		if (i == 5)
		{
			BOOST_CHECK_SMALL(value - target, 1e-2);
		}
		else
		{
			BOOST_CHECK_EQUAL(value, 0.0);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
