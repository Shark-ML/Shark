#define BOOST_TEST_MODULE GradDesc_Adam
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>
#include <shark/Algorithms/GradientDescent/Adam.h>

using namespace shark;
using namespace shark::benchmarks;

template<class VectorType>
struct NoisyEllipsoid : public AbstractObjectiveFunction<VectorType,double> {
	NoisyEllipsoid(size_t numberOfVariables, double alpha, double noise) : m_alpha(alpha) {
		this-> m_features |= this-> CAN_PROPOSE_STARTING_POINT;
		this-> m_features |= this-> HAS_FIRST_DERIVATIVE;
		this-> m_features |= this-> IS_NOISY;
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

	VectorType proposeStartingPoint() const {
		VectorType x(numberOfVariables());

		for (std::size_t i = 0; i < x.size(); i++) {
			x(i) = random::uni(*this->mep_rng, 0,1);
		}
		return x;
	}

	double eval( VectorType const& p ) const {
		this->m_evaluationCounter++;
		double sum = 0;
		double sizeMinusOne = p.size() - 1.;
		for( std::size_t i = 0; i < p.size(); i++ ){
			sum += ::pow( m_alpha, i / sizeMinusOne ) * sqr(p( i ) - random::gauss(random::globalRng,0,m_epsilon*m_epsilon));
		}

		return sum;
	}

	double evalDerivative( VectorType const& p, VectorType & derivative ) const {
		this->m_evaluationCounter++;
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

typedef boost::mpl::list<RealVector, FloatVector > VectorTypes;

BOOST_AUTO_TEST_CASE_TEMPLATE(Basic_Test, VectorType,VectorTypes){
	NoisyEllipsoid<VectorType> function(10,1.e-3,1.e-3);
	shark::Adam<VectorType> optimizer;
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
