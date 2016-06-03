#define BOOST_TEST_MODULE ALGORITHMS_HYPERVOLUME_CONTRIBUTION
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeContribution2D.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeContribution3D.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeContributionMD.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeContributionApproximator.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>

using namespace shark;

std::vector<KeyValuePair<double,std::size_t> > contributionsNaive(std::vector<RealVector> const& set, RealVector const& reference){
	HypervolumeCalculator hv;
	double totalVolume = hv(set,reference);
	std::vector<KeyValuePair<double,std::size_t> > contributions(set.size());
	for(std::size_t i = 0; i != set.size(); ++i){
		auto subset = set;
		subset.erase(subset.begin()+i,subset.begin()+i+1);
		contributions[i].value = i;
		contributions[i].key = totalVolume - hv(subset,reference);
	}
	return contributions;
}

template<class Algorithm>
void testContribution(Algorithm algorithm, std::vector<RealVector> const& set, std::size_t k, RealVector const& reference){
	
	std::vector<KeyValuePair<double,std::size_t>> leastContributions = algorithm.smallest(set,k,reference);
	std::vector<KeyValuePair<double,std::size_t>> largestContributions = algorithm.largest(set,k,reference);
	
	auto naiveContributions = contributionsNaive(set, reference);
	auto naiveLeastContributions = naiveContributions;
	std::sort(naiveLeastContributions.begin(),naiveLeastContributions.end());

	
	BOOST_REQUIRE_EQUAL(leastContributions.size(),k);
	BOOST_REQUIRE_EQUAL(largestContributions.size(),k);
	
	
	//check that the contributions of the points are equal, this guards
	//against point with the same contribution. Due to numerical issues
	//we have to test for closeness.
	
	//check least contributors
	for(std::size_t i = 0; i != k; ++i){
		//ensure returned value is correct
		BOOST_CHECK_SMALL(naiveLeastContributions[i].key - leastContributions[i].key,1.e-9);
		//ensure contribution of returned index is the same
		double contribution = naiveContributions[leastContributions[i].value].key;
		BOOST_CHECK_SMALL(naiveLeastContributions[i].key - contribution,1.e-9);
	}
	
	//check largest contributors
	for(std::size_t i = 0; i != k; ++i){
		//ensure returned value is correct
		BOOST_CHECK_SMALL(naiveLeastContributions[set.size()-i-1].key - largestContributions[i].key,1.e-9);
		//ensure contribution of returned index is the same
		//we can not check returned indizes directly as several might have the same value
		double contribution = naiveContributions[largestContributions[i].value].key;
		BOOST_CHECK_SMALL(naiveLeastContributions[set.size()-i-1].key - contribution,1.e-9);
	}
}


template<class Algorithm>
void testContributionNoRef(Algorithm algorithm, std::vector<RealVector> const& set, std::size_t k){
	
	std::vector<KeyValuePair<double,std::size_t>> leastContributions = algorithm.smallest(set,k);
	std::vector<KeyValuePair<double,std::size_t>> largestContributions = algorithm.largest(set,k);
	
	//compute reference
	RealVector reference(set[0]);
	for(auto const& p: set){
		noalias(reference) = max(reference,p);
	}

	//compute contributions
	auto naiveLeastContributions = contributionsNaive(set, reference);
	std::sort(naiveLeastContributions.begin(),naiveLeastContributions.end());
	//remove extrema
	for(std::size_t j = 0; j < set[0].size(); ++j){
		auto min = std::min_element(set.begin(),set.end(),[=](RealVector const& a, RealVector const& b){return a(j) < b(j);});
		auto index = min - set.begin();
		for(std::size_t i = 0; i != naiveLeastContributions.size(); ++i){
			if(naiveLeastContributions[i].value == index){
				naiveLeastContributions.erase(naiveLeastContributions.begin()+i);
				break;
			}
		}
	}
	
	BOOST_REQUIRE_EQUAL(leastContributions.size(),k);
	BOOST_REQUIRE_EQUAL(largestContributions.size(),k);
	
	//todo it is non-trivial to check whether the returned indices are correct.
	
	//check least contributors
	for(std::size_t i = 0; i != k; ++i){
		//ensure returned value is correct
		BOOST_CHECK_SMALL(naiveLeastContributions[i].key - leastContributions[i].key,1.e-9);
	}
	
	for(std::size_t i = 0; i != k; ++i){
		//ensure returned value is correct
		BOOST_CHECK_SMALL(naiveLeastContributions.end()[-i-1].key - largestContributions[i].key,1.e-9);
	}
}

//creates points on a front defined by points x in [0,1]^3
// 1 is a linear front, 2 a convex front, 1/2 a concave front
//reference point is 1^d
std::vector<RealVector> createRandomFront(std::size_t numPoints, std::size_t numObj, double p){
	std::vector<RealVector> points(numPoints);
	for (std::size_t i = 0; i != numPoints; ++i) {
		points[i].resize(numObj);
		double norm = 0;
		double sum = 0;
		for(std::size_t j = 0; j != numObj; ++j){
			points[i](j) = 1- Rng::uni(0.0, 1.0-sum);
			sum += 1-points[i](j);
			norm += std::pow(points[i](j),p);
		}
		norm = std::pow(norm,1/p);
		points[i] /= norm;
	}
	return points;
}

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_Operators_HypervolumeContribution)
BOOST_AUTO_TEST_CASE( Algorithms_HypervolumeContribution2D ) {

	HypervolumeContribution2D hs;
	const unsigned int numTests = 100;
	const std::size_t numPoints = 10;
	
	RealVector reference(2,1.0);
	
	for(unsigned int t = 0; t != numTests; ++t){
		auto frontLinear = createRandomFront(numPoints,2,1);
		auto frontConvex = createRandomFront(numPoints,2,2);
		auto frontConcave = createRandomFront(numPoints,2,0.5);
		
		for(std::size_t k = 1; k <= numPoints; ++k){
			testContribution(hs,frontLinear,k,reference);
			testContribution(hs,frontConvex,k,reference);
			testContribution(hs,frontConcave,k,reference);
			if(k < numPoints - 2){
				testContributionNoRef(hs,frontLinear,k);
				testContributionNoRef(hs,frontConvex,k);
				testContributionNoRef(hs,frontConcave,k);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( Algorithms_HypervolumeContribution3D ) {

	HypervolumeContribution3D hs;
	const unsigned int numTests = 100;
	const std::size_t numPoints = 50;
	
	RealVector reference(3,1.0);
	Rng::seed(42);
	
	for(unsigned int t = 0; t != numTests; ++t){
		auto frontLinear = createRandomFront(numPoints,3,1);
		auto frontConvex = createRandomFront(numPoints,3,2);
		auto frontConcave = createRandomFront(numPoints,3,0.5);
		
		for(std::size_t k = 1; k <= 7; ++k){
			testContribution(hs,frontLinear,k,reference);
			testContribution(hs,frontConvex,k,reference);
			testContribution(hs,frontConcave,k,reference);
			testContributionNoRef(hs,frontLinear,k);
			testContributionNoRef(hs,frontConvex,k);
			testContributionNoRef(hs,frontConcave,k);
		}
		
		//all points
		testContribution(hs,frontLinear,numPoints,reference);
		testContribution(hs,frontConvex,numPoints,reference);
		testContribution(hs,frontConcave,numPoints,reference);
		
		//without a reference we can not get than numPoints-3 contributions
		//as three points are required to set up the reference frame
		testContributionNoRef(hs,frontLinear,numPoints-3);
		testContributionNoRef(hs,frontConvex,numPoints-3);
		testContributionNoRef(hs,frontConcave,numPoints-3);
	}
}

BOOST_AUTO_TEST_CASE( Algorithms_HypervolumeContributionMD_With_3D ) {

	HypervolumeContributionMD hs;
	const unsigned int numTests = 20;
	const std::size_t numPoints = 100;
	
	RealVector reference(3,1.0);
	Rng::seed(42);
	
	for(unsigned int t = 0; t != numTests; ++t){
		auto frontLinear = createRandomFront(numPoints,3,1);
		auto frontConvex = createRandomFront(numPoints,3,2);
		auto frontConcave = createRandomFront(numPoints,3,0.5);
		
		for(std::size_t k = 1; k <= 7; ++k){
			testContribution(hs,frontLinear,k,reference);
			testContribution(hs,frontConvex,k,reference);
			testContribution(hs,frontConcave,k,reference);
			testContributionNoRef(hs,frontLinear,k);
			testContributionNoRef(hs,frontConvex,k);
			testContributionNoRef(hs,frontConcave,k);
		}
		
		//all points
		testContribution(hs,frontLinear,numPoints,reference);
		testContribution(hs,frontConvex,numPoints,reference);
		testContribution(hs,frontConcave,numPoints,reference);
		
		//without a reference we can not get moe than numPoints-3 contributions
		//as three points are required to set up the reference frame
		testContributionNoRef(hs,frontLinear,numPoints-3);
		testContributionNoRef(hs,frontConvex,numPoints-3);
		testContributionNoRef(hs,frontConcave,numPoints-3);
	}
}


BOOST_AUTO_TEST_CASE( Algorithms_HypervolumeContributionApproximator ) {
	const unsigned int numTests = 20;
	const unsigned int numTrials = 100;
	const std::size_t numPoints = 20;
	
	RealVector reference(3,1.0);
	Rng::seed(42);
	
	for(unsigned int t = 0; t != numTests; ++t){
		auto set = createRandomFront(numPoints,3,2);
		
		auto contributionsTrue = contributionsNaive(set, reference);
		std::vector<double> contributions(set.size());
		for(std::size_t i = 0; i != contributionsTrue.size(); ++i){
			contributions[contributionsTrue[i].value]=contributionsTrue[i].key;
		}

		HypervolumeContributionApproximator algorithm;
		algorithm.epsilon() = 0.01;
		algorithm.delta() = 0.1;
		std::vector<double> approxContributions;
		for(std::size_t i = 0; i != numTrials; ++i){
			auto result = algorithm.smallest(set,1,reference)[0].value;
			approxContributions.push_back(contributions[result]);
		}
		std::sort(approxContributions.begin(),approxContributions.end());
		
		//check that we do not have too many errors, i.e. contributions with errors larger than 1+epsilon
		//we make on average 100*errorProbability=10 errors. we give 50% more slack
		BOOST_CHECK_LT(approxContributions[(1-1.5*algorithm.delta())*numTrials], (1+algorithm.epsilon())*contributionsTrue[0].key);
	}
}

BOOST_AUTO_TEST_SUITE_END()
