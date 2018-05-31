#define BOOST_TEST_MODULE DirectSearch_SMSEMOA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/SMS-EMOA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>
#include "../testFunction.h"

using namespace shark;
using namespace shark::benchmarks;

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_SMSEMOA)

BOOST_AUTO_TEST_CASE( HYPERVOLUME_Functions ) {
	RealVector reference(2);
	reference(0) = 11;
	reference(1) = 11;
	std::size_t mu = 10;
	{
		DTLZ2 function(5);
		double volume = 120.178966;
		SMSEMOA optimizer;
		optimizer.mu() = mu;
		optimizer.indicator().setReference(reference);
		testFunction(optimizer, function, reference, volume,10, 10000,5.e-3);
	}
	{
		DTLZ4 function(5);
		double volume = 120.178966;
		SMSEMOA optimizer;
		optimizer.mu() = mu;
		optimizer.indicator().setReference(reference);
		testFunction(optimizer, function, reference, volume,10, 10000,5.e-3);
	}
	{
		ZDT1 function(5);
		double volume = 120.613761;
		SMSEMOA optimizer;
		optimizer.mu() = mu;
		optimizer.indicator().setReference(reference);
		testFunction(optimizer, function, reference, volume,10, 10000,5.e-3);
	}
	{
		ZDT2 function(5);
		double volume = 120.286820;
		SMSEMOA optimizer;
		optimizer.mu() = mu;
		optimizer.indicator().setReference(reference);
		testFunction(optimizer, function, reference, volume,10, 10000,5.e-3);
	}
	{
		ZDT3 function(5);
		double volume = 128.748470;
		SMSEMOA optimizer;
		optimizer.mu() = mu;
		optimizer.indicator().setReference(reference);
		testFunction(optimizer, function, reference, volume,10, 10000,5.e-3);
	}
	{
		ZDT6 function(5);
		double volume = 117.483246;
		SMSEMOA optimizer;
		optimizer.mu() = mu;
		optimizer.indicator().setReference(reference);
		testFunction(optimizer, function, reference, volume,10, 10000,5.e-3);
	}
}


BOOST_AUTO_TEST_SUITE_END()
