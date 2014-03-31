#define BOOST_TEST_MODULE DirectSearch_CMA
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/SteadyStateMOCMA.h>

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

using namespace shark;

BOOST_AUTO_TEST_CASE( ApproximatedHypSteadyStateMOCMA ) {

	PropertyTree node;
	SteadyStateMOCMA ssMocma;
	BOOST_CHECK_NO_THROW( ssMocma.configure( node ) );

	DTLZ1 dtlz1;
	dtlz1.setNumberOfObjectives( 3 );
	dtlz1.setNumberOfVariables( 10 );
	BOOST_CHECK_NO_THROW( ssMocma.init( dtlz1 ) );
	BOOST_CHECK_NO_THROW( ssMocma.step( dtlz1 ) );
	
	{
		std::stringstream ss;
		boost::archive::text_oarchive oa( ss );

		BOOST_CHECK_NO_THROW( (oa << ssMocma) );

		SteadyStateMOCMA ssMocma2;

		boost::archive::text_iarchive ia( ss );
		BOOST_CHECK_NO_THROW( (ia >> ssMocma2) );

		Rng::seed( 0 );
		ssMocma.step( dtlz1 );
		SteadyStateMOCMA::SolutionSetType set1 =  ssMocma.solution();
		Rng::seed( 0 );
		ssMocma2.step( dtlz1 );
		SteadyStateMOCMA::SolutionSetType set2 =  ssMocma2.solution();

		for( unsigned int i = 0; i < set1.size(); i++ ) {
			BOOST_CHECK_SMALL( norm_2( set1.at( i ).value - set2.at( i ).value ), 1E-20 );
			BOOST_CHECK_SMALL(norm_2( set1.at( i ).point - set2.at( i ).point ), 1E-20 );
		}

	}
}