#define BOOST_TEST_MODULE ALGORITHMS_HYPERVOLUME_SSP
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeSubsetSelection2D.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>

#include <numeric>//accumulate
#include <utility>//move

using namespace shark;


//return all subset of a set with size k
template<class T>
std::vector<std::vector<T> > subsets(std::vector<T> set,std::size_t k){
	if(set.size() == k){
		return {set};
	}
	if(k== 1){
		std::vector<std::vector<T> > result;
		for(auto const& point: set){
			result.push_back({point});
		}
		return result;
	}
	T point = set.back();
	set.pop_back();
	auto km1subsets = subsets(set,k-1);
	auto result = subsets(set,k);
	for(auto & s: km1subsets){
		s.push_back(point);
		result.push_back(std::move(s));
	}
	return result;
}

double hypSSPNaive(std::vector<RealVector> const& set, std::size_t k, RealVector const& reference){
	auto kSets = subsets(set,k);
	double maxVolume = 0;
	for(auto const& points: kSets){
		HypervolumeCalculator hv;
		maxVolume = std::max(maxVolume,hv(points,reference));
	}
	return maxVolume;
}

template<class Algorithm>
void testHypSSP(Algorithm algorithm, std::vector<RealVector> const& set, std::size_t k, RealVector const& reference){
	
	//ground truth
	double maxVolume = hypSSPNaive(set,k,reference);//exponential runtime!
	
	//compute algorithm
	std::vector<bool> selection(set.size());
	algorithm(set,selection,k,reference);
	
	//count number of selected points
	std::size_t count = std::accumulate(selection.begin(),selection.end(),(std::size_t)0);
	BOOST_REQUIRE_EQUAL(count,k);
	
	//check volume of solution
	std::vector<RealVector> chosenPoints;
	for(std::size_t i = 0; i != set.size(); ++i){
		if(selection[i])
			chosenPoints.push_back(set[i]);
	}
	HypervolumeCalculator hv;
	double volume = hv(chosenPoints,reference);
	
	BOOST_CHECK_CLOSE(maxVolume,volume, 1.e-10);//numerical difficiulties might sometimes select different sets with nearly the same volume(e.g. when two points are extremely to each other)
}

template<class Algorithm>
void testHypSSPNoRef(Algorithm algorithm, std::vector<RealVector> const& set, std::size_t k){
	SIZE_CHECK(k > set[0].size());
	
	//choose the reference such that the edge points do NOT have any volume
	RealVector reference(set[0]);
	for(auto const& point: set)
		noalias(reference)=max(reference,point);
	
	//ground truth assuems that the extrema in all targets are kept, which are three points(assuming a nondegenerate front)
	double maxVolume = hypSSPNaive(set,k-reference.size(),reference);//exponential runtime!
	
	//compute algorithm with the precomputed reference and k-ref.size() points. should give ame result as maxVolume
	std::vector<bool> selection1(set.size());
	algorithm(set,selection1,k-reference.size(),reference);
	
	//compute algorithm without reference, this should give the same volume but also return the edge points
	std::vector<bool> selection2(set.size());
	algorithm(set,selection2,k);
	
	//count number of selected points
	std::size_t count1 = std::accumulate(selection1.begin(),selection1.end(),(std::size_t)0);
	BOOST_REQUIRE_EQUAL(count1,k-reference.size());
	
	//count number of selected points
	std::size_t count2 = std::accumulate(selection2.begin(),selection2.end(),(std::size_t)0);
	BOOST_REQUIRE_EQUAL(count2,k);
	
	//check volume of solution
	std::vector<RealVector> chosenPoints1;
	std::vector<RealVector> chosenPoints2;
	for(std::size_t i = 0; i != set.size(); ++i){
		if(selection1[i])
			chosenPoints1.push_back(set[i]);
		if(selection2[i])
			chosenPoints2.push_back(set[i]);
		
		//check that the only difference is that selection2 picked more points
		if(selection1[i] != selection2[i]){
			BOOST_CHECK_EQUAL(selection1[i],false);
		}
	}
	HypervolumeCalculator hv;
	double volume1 = hv(chosenPoints1,reference);
	double volume2 = hv(chosenPoints2,reference);
	
	BOOST_CHECK_CLOSE(maxVolume,volume1, 1.e-10);//numerical difficiulties might sometimes select different sets with nearly the same volume
	BOOST_CHECK_CLOSE(maxVolume,volume2, 1.e-10);
}

//creates points on a front defined by points x in [0,1]^d
// 1 is a linear front, 2 a convex front, 1/2 a concave front
//reference point is 1^d
std::vector<RealVector> createRandomFront(std::size_t numPoints, std::size_t numObj, double p){
	std::vector<RealVector> points(numPoints);
	for (std::size_t i = 0; i != numPoints; ++i) {
		points[i].resize(numObj);
		for (std::size_t j = 0; j != numObj - 1; ++j) {
			points[i][j] = random::uni(random::globalRng,0.0, 1.0);
		}
		if (numObj > 2 && random::coinToss(random::globalRng))
		{
			// make sure that some objective values coincide
			std::size_t jj = random::discrete(random::globalRng, std::size_t(0), numObj - 2);
			points[i][jj] = std::round(4.0 * points[i][jj]) / 4.0;
		}
		double sum = 0.0;
		for (std::size_t j = 0; j != numObj - 1; ++j) sum += points[i][j];
		points[i][numObj - 1] = pow(1.0 - sum / (numObj - 1.0), p);
	}
	return points;
}

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_Operators_HypervolumeSubsetSelection)
BOOST_AUTO_TEST_CASE( Algorithms_HypervolumeSubsetSelection2D ) {

	HypervolumeSubsetSelection2D hs;
	const unsigned int numTests = 100;
	const std::size_t numPoints = 10;
	
	RealVector reference(2,1.0);
	
	for(unsigned int t = 0; t != numTests; ++t){
		auto frontLinear = createRandomFront(10,2,1);
		auto frontConvex = createRandomFront(10,2,2);
		auto frontConcave = createRandomFront(10,2,0.5);
		
		for(std::size_t k = 1; k != numPoints; ++k){
			testHypSSP(hs,frontLinear,k,reference);
			testHypSSP(hs,frontConvex,k,reference);
			testHypSSP(hs,frontConcave,k,reference);
		}
	}
}

BOOST_AUTO_TEST_CASE( Algorithms_HypervolumeSubsetSelection2D_NoRef ) {

	HypervolumeSubsetSelection2D hs;
	const unsigned int numTests = 100;
	const std::size_t numPoints = 10;
	
	for(unsigned int t = 0; t != numTests; ++t){
		auto frontLinear = createRandomFront(10,2,1);
		auto frontConvex = createRandomFront(10,2,2);
		auto frontConcave = createRandomFront(10,2,0.5);
		
		for(std::size_t k = 3; k != numPoints; ++k){
			testHypSSPNoRef(hs,frontLinear,k);
			testHypSSPNoRef(hs,frontConvex,k);
			testHypSSPNoRef(hs,frontConcave,k);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
