#define BOOST_TEST_MODULE ML_CG
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/GradientDescent/CG.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>
#include <shark/ObjectiveFunctions/Benchmarks/Sphere.h>

#include "../testFunction.h"

using namespace shark;
using namespace shark::benchmarks;


BOOST_AUTO_TEST_SUITE (Algorithms_GradientDescent_CG)

BOOST_AUTO_TEST_CASE( CG_dlinmin )
{
	Ellipsoid function(5);
	CG<> optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearchType::Dlinmin;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and dlinmin"<<std::endl;
	testFunction(optimizer,function,100,100);
}
BOOST_AUTO_TEST_CASE( CG_WolfeCubic )
{
	Ellipsoid function(5);
	CG<> optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearchType::WolfeCubic;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and WolfeCubic"<<std::endl;
	testFunction(optimizer,function,100,5000,1.e-10);
}
BOOST_AUTO_TEST_CASE( CG_Dlinmin_Rosenbrock )
{
	Rosenbrock function(3);
	CG<> optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearchType::Dlinmin;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and dlinmin"<<std::endl;
	testFunction(optimizer,function,100,3000,1.e-14);
}
BOOST_AUTO_TEST_CASE( CG_WolfeCubic_Rosenbrock )
{
	Rosenbrock function(3);
	CG<> optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearchType::WolfeCubic;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and WolfeCubic"<<std::endl;
	testFunction(optimizer,function,100,2000);
}
#ifdef SHARK_USE_OPENCL
struct GPUBenchmarkWrapper : public AbstractObjectiveFunction<RealGPUVector, double> {
	GPUBenchmarkWrapper(SingleObjectiveFunction& f) : m_f(&f) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= HAS_FIRST_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return m_f->name(); }
	
	std::size_t numberOfVariables()const{
		return m_f->numberOfVariables();
	}

	SearchPointType proposeStartingPoint() const {
		return blas::copy_to_gpu(m_f->proposeStartingPoint());
	}

	double eval( const SearchPointType & p ) const {
		return m_f->eval(blas::copy_to_cpu(p));
	}

	double evalDerivative( const SearchPointType & p, FirstOrderDerivative & derivative ) const {
		RealVector derivative_cpu;
		double val = m_f->evalDerivative(blas::copy_to_cpu(p), derivative_cpu);
		derivative.resize(numberOfVariables());
		noalias(derivative) = blas::copy_to_gpu(derivative_cpu);
		return val;
	}
private:
	SingleObjectiveFunction* m_f;
};
BOOST_AUTO_TEST_CASE( GPU_CG_dlinmin )
{
	Ellipsoid function_cpu(5);
	GPUBenchmarkWrapper function(function_cpu);
	CG<RealGPUVector> optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearchType::Dlinmin;

	std::cout<<"Testing on GPU: "<<optimizer.name()<<" with "<<function.name()<<" and dlinmin"<<std::endl;
	testFunction(optimizer,function,100,100, 1.e-5);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
