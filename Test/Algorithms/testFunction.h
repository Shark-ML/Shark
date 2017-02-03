#ifndef TEST_ALGORITHMS_TESTFUNCTION_H
#define TEST_ALGORITHMS_TESTFUNCTION_H

#include <shark/Core/utility/functional.h>
#include <boost/progress.hpp>

#include <fstream>

namespace shark {
template<class Optimizer,class Function>
void testFunction(Optimizer& optimizer,Function& function,unsigned int trials,unsigned int iterations, double epsilon = 1.e-15){
	boost::progress_display pd( trials * iterations );
		
	std::vector<double> stats;

	for( size_t trial =0;trial != static_cast<size_t>(trials);++trial ){
		optimizer.init(function);
				
		double error=0;

		for( size_t iteration = 0; iteration < static_cast<size_t>(iterations); ++iteration ) {
			//~ std::cout<<iteration<<" "<<iterations<<std::endl;
			optimizer.step( function );
			error=optimizer.solution().value;

			if( epsilon > error ){
				pd+=iterations-iteration;
				break;
			}
			
			++pd;
		}
		stats.push_back(error);
	}
	std::cout<<std::endl;
	double median =*shark::median_element(stats);
	BOOST_CHECK_SMALL( median, epsilon );
}
}
#endif
