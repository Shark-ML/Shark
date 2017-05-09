#define BOOST_TEST_MODULE GradDesc_Adam
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>
#include <shark/Algorithms/GradientDescent/Adam.h>

using namespace shark;

struct NoisyEllipsoid : public SingleObjectiveFunction {
	NoisyEllipsoid(size_t numberOfVariables, double alpha, double noise) : m_alpha(alpha) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= HAS_FIRST_DERIVATIVE;
		m_features |= IS_NOISY;
		m_numberOfVariables = numberOfVariables;
		m_epsilon = noise;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NoisyEllipsoid"; }
	
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
			x(i) = random::uni(*mep_rng, 0,1);
		}
		return x;
	}

	double eval( const SearchPointType & p ) const {
		m_evaluationCounter++;
		double sum = 0;
		double sizeMinusOne = p.size() - 1.;
		for( std::size_t i = 0; i < p.size(); i++ ){
			sum += ::pow( m_alpha, i / sizeMinusOne ) * sqr(p( i ) - random::gauss(random::globalRng,0,m_epsilon*m_epsilon));
		}

		return sum;
	}

	double evalDerivative( const SearchPointType & p, FirstOrderDerivative & derivative ) const {
		double sizeMinusOne=p.size() - 1.;
		derivative.resize(p.size());
		double sum = 0.0;
		for (std::size_t i = 0; i < p.size(); i++) {
			double c = ::pow(m_alpha, i / sizeMinusOne);
			double noise = random::gauss(random::globalRng,0,m_epsilon*m_epsilon);
			derivative(i) = 2 * c * (p(i) - noise);
			sum += c * sqr(p(i) - noise);
		}
		return sum;
	}
private:
	std::size_t m_numberOfVariables;
	double m_alpha;
	double m_epsilon;
};


BOOST_AUTO_TEST_SUITE (GradDesc_Adam)

BOOST_AUTO_TEST_CASE( Basic_Test )
{
	NoisyEllipsoid function(10,1.e-3,1.e-3);
	shark::Adam optimizer;
	function.init();
	optimizer.init(function);


	// train the model
	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	for(size_t iteration=0;iteration<10000;++iteration)
	{
		optimizer.step(function);
	}
	Ellipsoid test;
	BOOST_CHECK_SMALL(test(optimizer.solution().point),1.e-5);
}

BOOST_AUTO_TEST_SUITE_END()
