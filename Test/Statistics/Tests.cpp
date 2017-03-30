
#include <shark/Statistics/Tests.h>

#define BOOST_TEST_MODULE Statistics_Tests
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Statistical_Tests)

//all results taken from R
BOOST_AUTO_TEST_CASE( TTest_Test) {
	//mean is 49
	RealVector sample = { 46, 48, 51, 49, 46, 51, 52, 47, 49, 48, 48, 51, 49, 50, 52, 47 };
	
	{
		statistics::TTest test(51);
		BOOST_CHECK_CLOSE(test.statistics(sample),-4.0,1.e-10);
		BOOST_CHECK_CLOSE(p(test,sample,statistics::Tail::Left),0.0005796584,1.e-5);
		BOOST_CHECK_CLOSE(p(test,sample,statistics::Tail::Right),0.9994203,1.e-5);
		BOOST_CHECK_CLOSE(p(test,sample,statistics::Tail::TwoSided),0.001159317,1.e-4);
	}
}

BOOST_AUTO_TEST_CASE( TwoSample_TTest_Test) {
	RealVector sampleX = { 104, 111, 116, 103, 97, 99, 109, 112, 94 };
	RealVector sampleY = { 103, 110, 115, 102, 96, 98, 108, 111, 93, 104 };
	
	//test for unequal variance(default)
	{
		statistics::TwoSampleTTest test;
		BOOST_CHECK_CLOSE(test.statistics(sampleX, sampleY),0.2988072,1.e-4);
		BOOST_CHECK_CLOSE(p(test,sampleX, sampleY, statistics::Tail::Left),0.6155935,1.e-5);
		BOOST_CHECK_CLOSE(p(test,sampleX, sampleY, statistics::Tail::Right),0.3844065,1.e-4);
		BOOST_CHECK_CLOSE(p(test,sampleX, sampleY, statistics::Tail::TwoSided),0.768813,1.e-4);
	}
	
	//test for equal variance
	{
		statistics::TwoSampleTTest test(true);
		BOOST_CHECK_CLOSE(test.statistics(sampleX, sampleY),0.2997885,1.e-4);
		BOOST_CHECK_CLOSE(p(test,sampleX, sampleY, statistics::Tail::Left),0.6160134,1.e-5);
		BOOST_CHECK_CLOSE(p(test,sampleX, sampleY, statistics::Tail::Right),0.3839866,1.e-4);
		BOOST_CHECK_CLOSE(p(test,sampleX, sampleY, statistics::Tail::TwoSided),0.7679731,1.e-4);
	}
}

BOOST_AUTO_TEST_CASE( Paired_TTest_Test) {
	RealVector sampleX = { 223, 259, 248, 220, 287, 191, 229, 270, 245, 201 };
	RealVector sampleY = { 220, 244, 243, 211, 299 ,170, 210, 276, 252, 189 };
	
	{
		statistics::PairedTTest test;
		BOOST_CHECK_CLOSE(test.statistics(sampleX, sampleY),1.638538,1.e-4);
		BOOST_CHECK_CLOSE(p(test,sampleX, sampleY, statistics::Tail::Left),0.9321334,1.e-5);
		BOOST_CHECK_CLOSE(p(test,sampleX, sampleY, statistics::Tail::Right),0.06786656,1.e-4);
		BOOST_CHECK_CLOSE(p(test,sampleX, sampleY, statistics::Tail::TwoSided),0.1357331,1.e-4);
	}
}

BOOST_AUTO_TEST_SUITE_END()
