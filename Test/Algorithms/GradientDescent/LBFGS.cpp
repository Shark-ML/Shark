#define BOOST_TEST_MODULE GradDesc_LBFGS<>
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/GradientDescent/LBFGS.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

#include "../testFunction.h"

using namespace shark;
using namespace shark::benchmarks;

BOOST_AUTO_TEST_SUITE (Algorithms_GradientDescent_LBFGS)

BOOST_AUTO_TEST_CASE( LBFGS_dlinmin )
{
	Ellipsoid function(20);
	LBFGS<> optimizer;
	optimizer.setHistCount(10);
	optimizer.lineSearch().lineSearchType()=LineSearchType::Dlinmin;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and dlinmin"<<std::endl;
	testFunction(optimizer,function,100,100);
}
BOOST_AUTO_TEST_CASE( LBFGS_wolfe )
{
	Ellipsoid function(20);
	LBFGS<> optimizer;
	optimizer.setHistCount(10);
	optimizer.lineSearch().lineSearchType()=LineSearchType::WolfeCubic;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and wolfe line search"<<std::endl;
	testFunction(optimizer,function,100,100,1.e-8);
}
BOOST_AUTO_TEST_CASE( LBFGS_Dlinmin_Rosenbrock )
{
	Rosenbrock function(3);
	LBFGS<> optimizer;
	optimizer.setHistCount(3);
	optimizer.lineSearch().lineSearchType()=LineSearchType::Dlinmin;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and dlinmin"<<std::endl;
	testFunction(optimizer,function,100,100);
}
BOOST_AUTO_TEST_CASE( LBFGS_wolfe_Rosenbrock )
{
	Rosenbrock function(3);
	LBFGS<> optimizer;
	optimizer.setHistCount(3);
	optimizer.lineSearch().lineSearchType()=LineSearchType::WolfeCubic;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and wolfe line search"<<std::endl;
	testFunction(optimizer,function,100,100);
}




//with 2*n variables, variables 0...n-1 are constrained tbe in [-1,2] and the others
//lie in [1,2]
struct ConstrainedEllipsoid : public SingleObjectiveFunction {
	ConstrainedEllipsoid(size_t numPairs)
	: m_alpha(1.e-3)
	, m_handler(RealVector(numPairs,-1)|RealVector(numPairs,1), RealVector(2*numPairs,2)){
		announceConstraintHandler(&m_handler);
		m_features |= HAS_FIRST_DERIVATIVE;
		m_numberOfVariables = 2*numPairs;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ConstrainedEllipsoid"; }
	
	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}

	double eval( const SearchPointType & p ) const {
		BOOST_CHECK(isFeasible(p));
		m_evaluationCounter++;
		double sum = 0;
		double sizeMinusOne = p.size() - 1.;
		for( std::size_t i = 0; i < p.size(); i++ ){
			sum += ::pow( m_alpha, i / sizeMinusOne ) * sqr(p( i ) );
		}
		return sum;
	}

	double evalDerivative( const SearchPointType & p, FirstOrderDerivative & derivative ) const {
		BOOST_CHECK(isFeasible(p));
		double sizeMinusOne=p.size() - 1.;
		derivative.resize(p.size());
		for (std::size_t i = 0; i < p.size(); i++) {
			derivative(i) = 2 * ::pow(m_alpha, i / sizeMinusOne) * p(i);
		}
		return eval(p);
	}
private:
	std::size_t m_numberOfVariables;
	double m_alpha;
public:
	BoxConstraintHandler<SearchPointType> m_handler;
};

BOOST_AUTO_TEST_CASE( LBFGS_Constrained ){
	ConstrainedEllipsoid function(10);
	LBFGS<> optimizer;
	optimizer.setHistCount(3);

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and wolfe line search"<<std::endl;
	
	
	std::size_t trials = 100;
	std::size_t iterations = 100;
	double eps = 1.e-13;
	boost::progress_display pd( trials * iterations );
	for( size_t trial =0;trial != static_cast<size_t>(trials);++trial ){
		function.init();
		optimizer.init(function);
		for( unsigned int iteration = 0; iteration < iterations; ++iteration ) {
			optimizer.step( function );
			BOOST_CHECK(function.isFeasible(optimizer.solution().point));
			++pd;
		}
		//check solution
		RealVector derivative = optimizer.derivative();
		RealVector point = optimizer.solution().point;
		RealVector l=function.m_handler.lower();
		RealVector u=function.m_handler.upper();
		
		//~ std::cout<<derivative<<std::endl;
		//~ std::cout<<point<<std::endl;
		
		for(std::size_t i = 0; i != l.size(); ++i){
			if(point(i) - eps < l(i) )
				BOOST_CHECK_GT(derivative(i),-1.e-5);
			else if(point(i) + eps > u(i) )
				BOOST_CHECK_LT(derivative(i),1.e-5);
			else 
				BOOST_CHECK_SMALL(derivative(i), 1.e-5);
		}
	}
}



BOOST_AUTO_TEST_SUITE_END()
