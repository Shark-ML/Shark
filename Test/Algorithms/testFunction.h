#ifndef TEST_ALGORITHMS_TESTFUNCTION_H
#define TEST_ALGORITHMS_TESTFUNCTION_H

#include <shark/Core/utility/functional.h>

#include <boost/parameter.hpp>
#include <boost/progress.hpp>

#include <fstream>

namespace shark {
BOOST_PARAMETER_NAME(optimizer)    // Note: no semicolon
BOOST_PARAMETER_NAME(function)
BOOST_PARAMETER_NAME(trials)
BOOST_PARAMETER_NAME(iterations)
BOOST_PARAMETER_NAME(epsilon)
BOOST_PARAMETER_NAME(fstop)

BOOST_PARAMETER_FUNCTION(
	(void),
	test_function,
	tag,
	(required ( in_out(optimizer), * ) ( in_out(function), * ) )
	(optional
	(trials, *, 10)
	(iterations, *, 1000)
	(epsilon, *, 1E-15)
	(fstop, *, 1E-10)
)) {
	boost::progress_display pd( trials * iterations );
		
	std::vector<double> stats;

	for( size_t trial =0;trial != static_cast<size_t>(trials);++trial ){
		optimizer.init(function);
				
		double error=0;

		for( size_t iteration = 0; iteration < static_cast<size_t>(iterations); ++iteration ) {
			optimizer.step( function );
			error=optimizer.solution().value;

			++pd;

			if( fstop > error )
				break;
		}
		stats.push_back(error);
	}
	std::cout<<std::endl;
	double median =*shark::median_element(stats);
	BOOST_CHECK_SMALL( median, epsilon );
}
template<class Optimizer,class Function>
void testFunction(Optimizer& optimizer,Function& function,unsigned int trials,unsigned int iterations, double epsilon = 1.e-15){
	boost::progress_display pd( trials * iterations );
		
	std::vector<double> stats;

	for( size_t trial =0;trial != static_cast<size_t>(trials);++trial ){
		optimizer.init(function);
				
		double error=0;

		for( size_t iteration = 0; iteration < static_cast<size_t>(iterations); ++iteration ) {
			optimizer.step( function );
			error=optimizer.solution().value;

			++pd;

			if( epsilon > error )
				break;
		}
		stats.push_back(error);
	}
	std::cout<<std::endl;
	double median =*shark::median_element(stats);
	BOOST_CHECK_SMALL( median, epsilon );
}
}
#endif
