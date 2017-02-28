#ifndef TEST_ALGORITHMS_TESTFUNCTION_H
#define TEST_ALGORITHMS_TESTFUNCTION_H

#include <shark/Core/utility/functional.h>
#include <boost/progress.hpp>
#include <boost/range/adaptor/transformed.hpp>

#include <fstream>
#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>
namespace shark {
template<class Point, class Function>
void testFunction(AbstractSingleObjectiveOptimizer<Point>& optimizer,Function& function,unsigned int trials,unsigned int iterations, double epsilon = 1.e-15){
	boost::progress_display pd( trials * iterations );
		
	std::vector<double> stats;
	function.init();
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

struct PointExtractor{

	template<class T>
	RealVector const& operator()(T const& arg)const{
		return arg.value;
	}
};

template<class Point, class Function>
void testFunction(
	AbstractMultiObjectiveOptimizer<Point>& optimizer,Function& function,
	RealVector const& reference, double targetVolume,
 	unsigned int trials,unsigned int iterations,
	double epsilon = 1.e-15
){
	boost::progress_display pd( trials * iterations );
		
	RealVector stats;
	function.init();
	for( size_t trial =0;trial != static_cast<size_t>(trials);++trial ){
		optimizer.init(function);
				
		for( size_t iteration = 0; iteration < static_cast<size_t>(iterations); ++iteration ) {
			optimizer.step( function );
			++pd;
		}
		HypervolumeCalculator hyp;
		double volume = hyp(boost::adaptors::transform(optimizer.solution(),PointExtractor()),reference);
		stats.push_back(volume);
	}
	std::cout<<std::endl;
	BOOST_CHECK_SMALL( targetVolume - max(stats), epsilon );
}


}
#endif
