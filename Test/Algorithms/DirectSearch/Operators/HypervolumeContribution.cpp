#define BOOST_TEST_MODULE ALGORITHMS_HYPERVOLUME_CONTRIBUTION
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeContribution2D.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeContribution3D.h>
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
	
	std::vector<std::size_t> leastContributions = algorithm.least(set,k,reference);
	std::vector<std::size_t> largestContributions = algorithm.largest(set,k,reference);
	
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
		double contribution = naiveContributions[leastContributions[i]].key;
		
		BOOST_CHECK_SMALL(naiveLeastContributions[i].key-contribution,1.e-11);
	}
	
	//check largest contributors
	for(std::size_t i = 0; i != k; ++i){
		double contribution = naiveContributions[largestContributions[i]].key;
		
		BOOST_CHECK_SMALL(naiveLeastContributions[set.size()-i-1].key-contribution,1.e-11);
	}
}

//creates points on a front defined by points x in [0,1]^d
// 1 is a linear front, 2 a convex front, 1/2 a concave front
//reference point is 1^d
std::vector<RealVector> createRandomFront(std::size_t numPoints, std::size_t numObj, double p){
	std::vector<RealVector> points(numPoints);
	for (std::size_t i = 0; i != numPoints; ++i) {
		points[i].resize(numObj);
		for (std::size_t j = 0; j != numObj - 1; ++j) {
			points[i][j] = Rng::uni(0.0, 1.0);
		}
		if (numObj > 2 && Rng::coinToss())
		{
			// make sure that some objective values coincide
			std::size_t jj = Rng::discrete(0, numObj - 2);
			points[i][jj] = std::round(4.0 * points[i][jj]) / 4.0;
		}
		double sum = 0.0;
		for (std::size_t j = 0; j != numObj - 1; ++j) sum += points[i][j];
		points[i][numObj - 1] = pow(1.0 - sum / (numObj - 1.0), p);
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
		
		for(std::size_t k = 1; k <= 10; ++k){
			testContribution(hs,frontLinear,k,reference);
			testContribution(hs,frontConvex,k,reference);
			testContribution(hs,frontConcave,k,reference);
		}
		
		//all points
		testContribution(hs,frontLinear,numPoints,reference);
		testContribution(hs,frontConvex,numPoints,reference);
		testContribution(hs,frontConcave,numPoints,reference);
	}
}

BOOST_AUTO_TEST_SUITE_END()
