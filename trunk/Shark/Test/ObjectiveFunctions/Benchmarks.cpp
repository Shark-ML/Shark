#define BOOST_TEST_MODULE ObjectiveFunctions_Benchmarks
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/progress.hpp>
#include <boost/serialization/vector.hpp>

#include <shark/Algorithms/DirectSearch/AGE.h>
#include <shark/Algorithms/DirectSearch/SteadyStateMOCMA.h>
#include <shark/Algorithms/DirectSearch/SMS-EMOA.h>
#include <shark/Algorithms/DirectSearch/RealCodedNSGAII.h>

#include <shark/Algorithms/DirectSearch/HypervolumeApproximator.h>
#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>

#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

#include <shark/Statistics/Statistics.h>

#include <fstream>

#include "TestObjectiveFunction.h"

BOOST_AUTO_TEST_CASE( MultiObjective_Benchmark_Functions ) {

	shark::moo::RealValuedObjectiveFunctionFactory::instance().print( std::cout );

	typedef shark::moo::RealValuedObjectiveFunctionFactory::class_type function_type;

	shark::moo::RealValuedObjectiveFunctionFactory::const_iterator it;
	for( it = shark::moo::RealValuedObjectiveFunctionFactory::instance().begin();
		it != shark::moo::RealValuedObjectiveFunctionFactory::instance().end();
		++it
		) {
			std::cout<< "Considering function: " << it->first <<std::endl;

			function_type * function = NULL;
			BOOST_CHECK( it->second != NULL );
			function = it->second->create();

			BOOST_CHECK( function != NULL );
			BOOST_CHECK( function->name() == it->first );
			BOOST_CHECK_NO_THROW( function->setNumberOfVariables( 10 ) );
			BOOST_CHECK( function->numberOfVariables() == 10 );
			BOOST_CHECK_NO_THROW( function->init() );

			function_type::SearchPointType sp;

			if( function->features() & function_type::CAN_PROPOSE_STARTING_POINT ) {
				
				BOOST_CHECK_NO_THROW(  function->proposeStartingPoint( sp ) );
				BOOST_CHECK( function->isFeasible( sp ) );
				BOOST_CHECK( sp.size() == function->numberOfVariables() );

				//~ function_type::SearchPointType sp2( sp );
				//~ if( function->features() & function_type::CAN_PROVIDE_CLOSEST_FEASIBLE ) {
					//~ BOOST_CHECK_NO_THROW( function->closestFeasible( sp2 ) );
					//~ //BOOST_CHECK( sp == sp2 );
				//~ } else
					//~ BOOST_CHECK_THROW( function->closestFeasible( sp2 ), shark::Exception );

				if( function->features() & function_type::HAS_VALUE ) {
					BOOST_CHECK_NO_THROW( function->eval( sp ) );
				} else
					BOOST_CHECK_THROW( function->eval( sp ), shark::Exception );

				function_type::FirstOrderDerivative fod;
				if( function->features() & function_type::HAS_FIRST_DERIVATIVE ) {
					BOOST_CHECK_NO_THROW( function->evalDerivative( sp, fod ) );
				} else
					BOOST_CHECK_THROW( function->evalDerivative( sp, fod ), shark::Exception );

				function_type::SecondOrderDerivative sod;
				if( function->features() & function_type::HAS_SECOND_DERIVATIVE ) {
					BOOST_CHECK_NO_THROW( function->evalDerivative( sp, sod ) );
				} else
					BOOST_CHECK_THROW( function->evalDerivative( sp, sod ), shark::Exception );
					
			} else {
				BOOST_CHECK_THROW( function->proposeStartingPoint( sp ), shark::Exception );
			}
		
	}
}

BOOST_AUTO_TEST_CASE( SingleObjective_Benchmark_Functions ) {

	shark::soo::RealValuedObjectiveFunctionFactory::instance().print( std::cout );

	typedef shark::soo::RealValuedObjectiveFunctionFactory::class_type function_type;

	shark::soo::RealValuedObjectiveFunctionFactory::const_iterator it;
	for( it = shark::soo::RealValuedObjectiveFunctionFactory::instance().begin();
		it != shark::soo::RealValuedObjectiveFunctionFactory::instance().end();
		++it
		) {
			std::cout<< "Considering function: " << it->first <<std::endl;

			function_type * function = NULL;
			BOOST_CHECK( it->second != NULL );
			function = it->second->create();

			BOOST_CHECK( function != NULL );
			BOOST_CHECK( function->name() == it->first );
			if(function->hasScalableDimensionality()){
				BOOST_CHECK_NO_THROW( function->setNumberOfVariables( 10 ) );
			}else{
				BOOST_CHECK_THROW( function->setNumberOfVariables( 10 ), shark::Exception );
			}
			BOOST_CHECK_NO_THROW( function->init() );

			function_type::SearchPointType sp;

			if( function->features() & function_type::CAN_PROPOSE_STARTING_POINT ) {

				BOOST_CHECK_NO_THROW(  function->proposeStartingPoint( sp ) );
				BOOST_CHECK( function->isFeasible( sp ) );
				BOOST_CHECK( sp.size() == function->numberOfVariables() );

				//~ function_type::SearchPointType sp2( sp );
				//~ if( function->features() & function_type::CAN_PROVIDE_CLOSEST_FEASIBLE ) {
					//~ BOOST_CHECK_NO_THROW( function->closestFeasible( sp2 ) );
					//~ //BOOST_CHECK( sp == sp2 );
				//~ } else
					//~ BOOST_CHECK_THROW( function->closestFeasible( sp2 ), shark::Exception );

				if( function->features() & function_type::HAS_VALUE ) {
					BOOST_CHECK_NO_THROW( function->eval( sp ) );
				} else
					BOOST_CHECK_THROW( function->eval( sp ), shark::Exception );

				function_type::FirstOrderDerivative fod;
				if( function->features() & function_type::HAS_FIRST_DERIVATIVE ) {
					BOOST_CHECK_NO_THROW( function->evalDerivative( sp, fod ) );
				} else
					BOOST_CHECK_THROW( function->evalDerivative( sp, fod ), shark::Exception );

				function_type::SecondOrderDerivative sod;
				if( function->features() & function_type::HAS_SECOND_DERIVATIVE ) {
					BOOST_CHECK_NO_THROW( function->evalDerivative( sp, sod ) );
				} else
					BOOST_CHECK_THROW( function->evalDerivative( sp, sod ), shark::Exception );

			} else {
				BOOST_CHECK_THROW( function->proposeStartingPoint( sp ), shark::Exception );
			}

	}
}

BOOST_AUTO_TEST_CASE( Himmelblau ) {
	shark::Himmelblau hb;

	//~ BOOST_CHECK_THROW( hb( shark::RealVector( 1 ) ), shark::Exception );
	//~ BOOST_CHECK_THROW( hb( shark::RealVector( 3 ) ), shark::Exception );

	shark::RealVector v( 2 );
	
	v( 0 ) = -0.270844;
	v( 1 ) = -0.923038;

	BOOST_CHECK_SMALL( hb( v ) - 181.616, 1E-3 );

	v( 0 ) = 3;
	v( 1 ) = 2;
	BOOST_CHECK_SMALL( hb( v ), 1E-10 );

	v( 0 ) = -2.805118;
	v( 1 ) = 3.131312;
	BOOST_CHECK_SMALL( hb( v ), 1E-10 );

	v( 0 ) = -3.779310;
	v( 1 ) = -3.283186;
	BOOST_CHECK_SMALL( hb( v ), 1E-10 );

	v( 0 ) = 3.584428;
	v( 1 ) = -1.848126;
	BOOST_CHECK_SMALL( hb( v ), 1E-10 );
}

BOOST_AUTO_TEST_CASE( Rosenbrock_Derivative )
{
	const std::size_t dimensions = 5;
	const unsigned trials = 10000;
	
	shark::Rosenbrock rosenbrock(dimensions);
	for(unsigned i = 0; i != trials; ++i)
	{
		shark::RealVector point(dimensions);
		rosenbrock.proposeStartingPoint(point);
		shark::testDerivative(rosenbrock, point,1.e-7,1.e-7);
	}
}
BOOST_AUTO_TEST_CASE( Ellipsoid_Derivative )
{
	const std::size_t dimensions = 5;
	const unsigned trials = 10000;
	
	shark::Ellipsoid ellipsoid(dimensions);
	for(unsigned i = 0; i != trials; ++i)
	{
		shark::RealVector point(dimensions);
		ellipsoid.proposeStartingPoint(point);
		shark::testDerivative(ellipsoid, point,1.e-5,1.e-9);
	}
}
