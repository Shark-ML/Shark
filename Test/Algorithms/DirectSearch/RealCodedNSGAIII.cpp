#define BOOST_TEST_MODULE DirectSearch_RealCodedNSGAIII
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/RealCodedNSGAIII.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>
#include "../testFunction.h"
using namespace shark;
using namespace shark::benchmarks;

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_RealCodedNSGAIII)

BOOST_AUTO_TEST_CASE( TEST_NSGAIII) {
	RealVector reference(2);
	reference(0) = 11;
	reference(1) = 11;
	std::size_t mu = 12;
	std::cout<<"Optimizing NSGAIII"<<std::endl;
	{
		DTLZ2 function(5);
		double volume = 120.178966;
		RealCodedNSGAIII optimizer;
		optimizer.mu() = 12;
		testFunction(optimizer, function, reference, volume,10, 1000,5.e-3);
	}
	{
		ZDT1 function(5);
		double volume = 120.613761;
		RealCodedNSGAIII optimizer;
		optimizer.mu() = 11;
		testFunction(optimizer, function, reference, volume,10, 1000,5.e-3);
	}
	{
		ZDT2 function(5);
		double volume = 120.286820;
		RealCodedNSGAIII optimizer;
		optimizer.mu() = mu;
		testFunction(optimizer, function, reference, volume,10, 1000,5.e-3);
	}
	//~ {
		//~ ZDT3 function(5);
		//~ double volume = 128.748470;
		//~ RealCodedNSGAIII optimizer;
		//~ optimizer.mu() = mu;
		//~ testFunction(optimizer, function, reference, volume,10, 2000,5.e-3);
	//~ }
	{
		ZDT6 function(5);
		double volume = 117.483246;
		RealCodedNSGAIII optimizer;
		optimizer.mu() = 11;
		testFunction(optimizer, function, reference, volume,10, 1000,5.e-3);
	}
}


BOOST_AUTO_TEST_SUITE_END()
