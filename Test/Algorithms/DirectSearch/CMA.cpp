#define BOOST_TEST_MODULE DirectSearch_CMA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/CMA.h>
#include <shark/Algorithms/DirectSearch/ElitistCMA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Cigar.h>
#include <shark/ObjectiveFunctions/Benchmarks/Discus.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>
#include <shark/ObjectiveFunctions/Benchmarks/Sphere.h>

#include "../testFunction.h"

using namespace shark;
using namespace shark::benchmarks;


struct MultiplicativeNoisySphere : public SingleObjectiveFunction {
	
	MultiplicativeNoisySphere(std::size_t numberOfVariables, double sigma):m_numberOfVariables(numberOfVariables), m_sigma(sigma) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= IS_NOISY;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "MultiplicativeNoisySphere"; }

	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}

	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
	}

	SearchPointType proposeStartingPoint() const {
		RealVector x(numberOfVariables());

		for (std::size_t i = 0; i < x.size(); i++) {
			x(i) = random::gauss(random::globalRng,0,1);
		}
		return x;
	}

	double eval(SearchPointType const& x) const {
		SIZE_CHECK(x.size() == numberOfVariables());
		m_evaluationCounter++;
		double f = norm_sqr(x);
		double noise = (1+random::gauss(random::globalRng,0,m_sigma));
		return f * noise;
	}
private:
	std::size_t m_numberOfVariables;
	double m_sigma;
};

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_CMA)

BOOST_AUTO_TEST_CASE( CMA_Cigar )
{
	Cigar function(3);
	CMA optimizer;
	ElitistCMA elitistCMA;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	testFunction( optimizer, function, 10, 1000, 1E-10 );
}

BOOST_AUTO_TEST_CASE( CMA_Discus )
{
	Discus function(3);
	CMA optimizer;

	std::cout<<"\nTesting: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	testFunction( optimizer, function, 10, 1000, 1E-10 );
}

BOOST_AUTO_TEST_CASE( CMA_Ellipsoid )
{
	Ellipsoid function(5);
	CMA optimizer;

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	testFunction( optimizer, function, 10, 1000, 1E-10 );
}

BOOST_AUTO_TEST_CASE( CMA_Rosenbrock )
{
	Rosenbrock function( 3 );
	CMA optimizer;

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	testFunction( optimizer, function, 10, 1000, 1E-10 );
}

BOOST_AUTO_TEST_CASE( CMA_Ellipsoid_Niko )
{
	const unsigned N = 10;
	RealVector x0(10, 0.1);
	Ellipsoid elli(N, 1E6);
	elli.init();
	CMA cma;
	cma.setInitialSigma(0.1);
	cma.init(elli, x0);
	BOOST_REQUIRE(cma.sigma() == 0.1);

	for(unsigned i=0; i<6000; i++) 	cma.step( elli );
	BOOST_CHECK(cma.solution().value < 1E-8);
	BOOST_CHECK(cma.condition() > 1E5);
}

BOOST_AUTO_TEST_CASE( CMA_Sphere_Niko )
{
	random::globalRng.seed(43);
	const unsigned N = 10;
	RealVector x0(10, 0.1);
	Sphere sphere(N);
	sphere.init();
	CMA cma(random::globalRng);
	cma.setInitialSigma(1.e-4);
	cma.init(sphere, x0);
	BOOST_REQUIRE(cma.sigma() == 1.e-4);

	bool sigmaHigh = false;
	bool condHigh = false;
	for(unsigned i=0; i<1500; i++) {
		cma.step( sphere );
		if(i % 50 == 0){
			std::cout<<i <<"\t" << cma.sigma() <<"\t" << cma.condition() <<std::endl;
		}
		if(cma.sigma() > 0.01) sigmaHigh = true;
		if(cma.condition() > 40) condHigh = true;
	}
	BOOST_CHECK(cma.solution().value < 1E-9);
	BOOST_CHECK(sigmaHigh);
	BOOST_CHECK(!condHigh);
}

BOOST_AUTO_TEST_CASE( CMA_Multiplicative_Noisy_Sphere)
{
	std::cout<<"start"<<std::endl;
	random::globalRng.seed(42);
	const unsigned N = 10;
	RealVector x0(10, 0.1);
	MultiplicativeNoisySphere sphere(N,10.0);
	sphere.init();
	CMA cma(random::globalRng);
	cma.init(sphere, x0);

	double start = log(norm_sqr(cma.solution().point));
	for(unsigned i=0; i<1501; i++) {
		cma.step( sphere );
		if(i%50 == 0)
			std::cout<<i<<"\t"<<norm_sqr(cma.solution().point)<<"\t"<<(std::log(norm_sqr(cma.solution().point))-start)/sphere.evaluationCounter()<<"\t"<<cma.numberOfEvaluations()<<std::endl;
	}
	BOOST_CHECK(cma.solution().value < 1E-13);
}

BOOST_AUTO_TEST_SUITE_END()
