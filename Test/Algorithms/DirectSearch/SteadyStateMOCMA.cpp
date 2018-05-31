#define BOOST_TEST_MODULE DirectSearch_STEADYSTATEMOCMA
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/SteadyStateMOCMA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>
#include "../testFunction.h"
using namespace shark;
using namespace shark::benchmarks;


BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_SteadyStateMOCMA)

BOOST_AUTO_TEST_CASE( HYPERVOLUME_Functions ) {
	RealVector reference(2);
	reference(0) = 11;
	reference(1) = 11;
	std::size_t mu = 10;
	{
		DTLZ2 function(5);
		double volume = 120.178966;
		SteadyStateMOCMA mocma;
		mocma.mu() = mu;
		mocma.indicator().setReference(reference);
		testFunction(mocma, function, reference, volume,1, 10000,5.e-3);
	}
	{
		DTLZ4 function(5);
		double volume = 120.178966;
		SteadyStateMOCMA mocma;
		mocma.mu() = mu;
		mocma.indicator().setReference(reference);
		testFunction(mocma, function, reference, volume,1, 10000,5.e-3);
	}
	{
		ZDT1 function(5);
		double volume = 120.613761;
		SteadyStateMOCMA mocma;
		mocma.mu() = mu;
		mocma.indicator().setReference(reference);
		testFunction(mocma, function, reference, volume,1, 10000,5.e-3);
	}
	{
		ZDT2 function(5);
		double volume = 120.286820;
		SteadyStateMOCMA mocma;
		mocma.mu() = mu;
		mocma.indicator().setReference(reference);
		testFunction(mocma, function, reference, volume,1, 10000,5.e-3);
	}
	{
		ZDT3 function(5);
		double volume = 128.748470;
		SteadyStateMOCMA mocma;
		mocma.mu() = mu;
		mocma.indicator().setReference(reference);
		testFunction(mocma, function, reference, volume,1, 10000,5.e-3);
	}
	{
		ZDT6 function(5);
		double volume = 117.483246;
		SteadyStateMOCMA mocma;
		mocma.mu() = mu;
		mocma.indicator().setReference(reference);
		testFunction(mocma, function, reference, volume,1, 10000,5.e-3);
	}
}

BOOST_AUTO_TEST_CASE( Algorithms_DirectSearch_SteadyStateMOCMA_Serialize ) {

	SteadyStateMOCMA ssMocma;
	DTLZ1 dtlz1;
	dtlz1.setNumberOfObjectives( 3 );
	dtlz1.setNumberOfVariables( 10 );
	BOOST_CHECK_NO_THROW( ssMocma.init( dtlz1 ) );
	BOOST_CHECK_NO_THROW( ssMocma.step( dtlz1 ) );
	
	{
		std::stringstream ss;
		TextOutArchive oa( ss );

		BOOST_CHECK_NO_THROW( (oa << ssMocma) );

		SteadyStateMOCMA ssMocma2;

		TextInArchive ia( ss );
		BOOST_CHECK_NO_THROW( (ia >> ssMocma2) );

		random::globalRng.seed( 0 );
		ssMocma.step( dtlz1 );
		SteadyStateMOCMA::SolutionType set1 =  ssMocma.solution();
		random::globalRng.seed( 0 );
		ssMocma2.step( dtlz1 );
		SteadyStateMOCMA::SolutionType set2 =  ssMocma2.solution();

		for( unsigned int i = 0; i < set1.size(); i++ ) {
			BOOST_CHECK_SMALL( norm_2( set1.at( i ).value - set2.at( i ).value ), 1E-20 );
			BOOST_CHECK_SMALL(norm_2( set1.at( i ).point - set2.at( i ).point ), 1E-20 );
		}

	}
}
BOOST_AUTO_TEST_SUITE_END()
