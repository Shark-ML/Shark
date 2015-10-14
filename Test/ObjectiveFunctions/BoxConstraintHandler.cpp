#define BOOST_TEST_MODULE OBJECTIVEFUNCTION_BOXCONSTRAINTHANDLER
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/LinAlg/Base.h>

using namespace shark;

struct Fixture {

	Fixture():handlerSame(10,0,1){
		RealVector lower(10);
		RealVector upper(10);
		for(std::size_t i = 0; i != 10; ++i){
			lower(i) = i-10.0;
			upper(i) = lower(i)+2*i;
		}
		handler = BoxConstraintHandler<RealVector>(lower,upper);
	}
	BoxConstraintHandler<RealVector> handler;
	BoxConstraintHandler<RealVector> handlerSame;
};

BOOST_FIXTURE_TEST_SUITE (ObjectiveFunctions_BoxConstraintHandler, Fixture)

//checks that the handler is constructed with the right bounds
BOOST_AUTO_TEST_CASE( BoxConstraintHandler_Bounds ) {
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_CLOSE(handler.lower()(i),i-10.0,1.e-10);
		BOOST_CHECK_CLOSE(handler.upper()(i),3*i-10.0,1.e-10);
		BOOST_CHECK_CLOSE(handlerSame.lower()(i),0,1.e-10);
		BOOST_CHECK_CLOSE(handlerSame.upper()(i),1,1.e-10);
	}
}


//checks that the handler geenerates feasible points in the region 
//and that isFeasible and clsoestFeasible do the right thing (i.e. return true and do nothing) 
BOOST_AUTO_TEST_CASE( BoxConstraintHandler_Generate ) {
	for(std::size_t i = 0; i != 1000; ++i){
		RealVector point;
		handler.generateRandomPoint(point);
		BOOST_REQUIRE_EQUAL(point.size(),10);
		//check that the point is feasible
		for(std::size_t i = 0; i != 10; ++i){
			BOOST_CHECK(point(i) >= handler.lower()(i));
			BOOST_CHECK(point(i) <= handler.upper()(i));
		}
		//must be feasible
		BOOST_CHECK(handler.isFeasible(point));
		//closest feasible must not change the point
		RealVector feasible=point;
		handler.closestFeasible(feasible);
		BOOST_CHECK_SMALL(norm_sqr(feasible-point),1.e-12);
	}
}

//given a random point, checks whether it is feasible and checks whether the correction is okay.
BOOST_AUTO_TEST_CASE( BoxConstraintHandler_Infeasible ) {
	for(std::size_t i = 0; i != 1000; ++i){
		RealVector point(10);
		RealVector corrected(10);
		bool feasible = true;
		for(std::size_t i = 0; i != 10; ++i){
			point(i) = Rng::uni(-30,40);
			corrected(i) = point(i);
			if(point(i) < handler.lower()(i)){
				corrected(i) = handler.lower()(i);
				feasible = false;
			}
			if(point(i) > handler.upper()(i)){
				corrected(i) = handler.upper()(i);
				feasible = false;
			}
		}
		
		//check feasibility of both points
		BOOST_CHECK(handler.isFeasible(point) == feasible);
		BOOST_CHECK(handler.isFeasible(corrected));
		
		//now check that the handler returns the same corrected point
		RealVector handlerCorrected = point;
		handler.closestFeasible(handlerCorrected);
		BOOST_REQUIRE_EQUAL(handlerCorrected.size(),10);
		for(std::size_t i = 0; i != 10; ++i){
			BOOST_CHECK_EQUAL(handlerCorrected(i),corrected(i));
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
