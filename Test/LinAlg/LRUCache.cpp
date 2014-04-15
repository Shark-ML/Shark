#define BOOST_TEST_MODULE LINALG_LRUCACHE
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/LRUCache.h>
#include <shark/Rng/GlobalRng.h>
#include <algorithm>

using namespace shark;

void simulateCache(
	std::size_t maxIndex,std::size_t cacheSize,
	std::vector<std::size_t> const& accessIndices,
	std::vector<std::size_t> const& accessSizes,
	std::vector<std::pair<std::size_t,std::size_t > > const& flips
){
	BOOST_REQUIRE_EQUAL(accessIndices.size(), accessSizes.size());
	BOOST_REQUIRE_EQUAL(accessIndices.size(), flips.size());
	std::size_t simulationSteps = accessIndices.size();
	
	LRUCache<std::size_t> cache(maxIndex,cacheSize);
	std::size_t currentCacheSize = 0;
	std::vector<std::size_t> elemSizes(maxIndex,0);
	std::list<std::size_t> lruList;
	for(std::size_t t = 0; t != simulationSteps; ++t){
		std::size_t index = accessIndices[t];
		std::size_t size = accessSizes[t];
		//in the simulated cache we can just throw queried elements away if they would
		//ned to be resized and add them later on
		if(size > elemSizes[index] && elemSizes[index] != 0){
			//remove element from the simulated cache
			currentCacheSize -= elemSizes[index];
			elemSizes[index] = 0;
			lruList.erase(std::find(lruList.begin(),lruList.end(),index));
		}
		if(size <= elemSizes[index] && elemSizes[index] != 0){
			lruList.erase(std::find(lruList.begin(),lruList.end(),index));
			lruList.push_front(index);
		}else{
			while(cacheSize < currentCacheSize+size){
				std::size_t index2 = lruList.back();
				currentCacheSize -= elemSizes[index2];
				elemSizes[index2] = 0;
				lruList.pop_back();
			}
			//add element to the simulated cache
			currentCacheSize +=size;
			elemSizes[index] = size;
			lruList.push_front(index);
		}
		
		//access real cache
		std::size_t* line = cache.getCacheLine(index,size);
		for(std::size_t i = 0; i != size; ++i){
			line[i] = i+1;
		}
		
		//check whether the caching is correct
		for(std::size_t i = 0; i != maxIndex; ++i){
			BOOST_REQUIRE_EQUAL(cache.lineLength(i), elemSizes[i]);
			BOOST_CHECK(cache.cachedLines()<= cacheSize);
			BOOST_CHECK_EQUAL(cache.size(), currentCacheSize);
			//check that elements are the same
			for(std::size_t j = 0; j != elemSizes[i]; ++j){
				BOOST_CHECK_EQUAL(cache.getLinePointer(i)[j], j+1);
			}
		}
		
		//apply flipping
		std::pair<std::size_t,std::size_t > flip = flips[t];
		if(flip.first == flip.second)
			continue;
		
		//flip in simulated cache
		std::swap(elemSizes[flip.first],elemSizes[flip.second]);
		std::list<std::size_t>::iterator iter1 = std::find(lruList.begin(),lruList.end(),flip.first);
		std::list<std::size_t>::iterator iter2 = std::find(lruList.begin(),lruList.end(),flip.second);
		if(iter1 == lruList.end() && iter2 == lruList.end())
			continue;
		else if(iter1 == lruList.end())
			*iter2=flip.first;
		else if(iter2 == lruList.end())
			*iter1 = flip.second;
		else
			std::iter_swap(iter1,iter2);
		
		//flip in real cache
		cache.swapLineIndices(flip.first,flip.second);
		
	}
}

///\brief tests whether simple same length access-schemes work
BOOST_AUTO_TEST_CASE( LinAlg_LRUCache_Simple_Access ) {
	std::size_t cacheSize = 10;
	std::size_t maxIndex = 20;
	std::size_t simulationSteps = 10000;
	
	std::vector<std::size_t> accessIndices(simulationSteps);
	std::vector<std::size_t> accessSizes(simulationSteps,1);
	std::vector<std::pair<std::size_t,std::size_t > > flips(simulationSteps,std::pair<std::size_t,std::size_t >(0,0));
	for(std::size_t i = 0; i != simulationSteps; ++i){
		accessIndices[i] = Rng::discrete(0,maxIndex-1);
	}
	simulateCache(maxIndex, cacheSize,accessIndices,accessSizes,flips);
}
///\brief tests whether simple different  length access-schemes work
BOOST_AUTO_TEST_CASE( LinAlg_LRUCache_DifferentLength_Access ) {
	std::size_t cacheSize = 10;
	std::size_t maxIndex = 20;
	std::size_t simulationSteps = 10000;
	
	std::vector<std::size_t> accessIndices(simulationSteps);
	std::vector<std::size_t> accessSizes(simulationSteps);
	std::vector<std::pair<std::size_t,std::size_t > > flips(simulationSteps,std::pair<std::size_t,std::size_t >(0,0));
	for(std::size_t i = 0; i != simulationSteps; ++i){
		accessIndices[i] = Rng::discrete(0,maxIndex-1);
		accessSizes[i] = Rng::discrete(1,3);
	}
	simulateCache(maxIndex, cacheSize,accessIndices,accessSizes,flips);
}

///\brief tests whether simple same length access-schemes work
BOOST_AUTO_TEST_CASE( LinAlg_LRUCache_DifferentLength_Access_fliped ) {
	std::size_t cacheSize = 10;
	std::size_t maxIndex = 20;
	std::size_t simulationSteps = 10000;
	
	std::vector<std::size_t> accessIndices(simulationSteps);
	std::vector<std::size_t> accessSizes(simulationSteps);
	std::vector<std::pair<std::size_t,std::size_t > > flips(simulationSteps);

	for(std::size_t i = 0; i != simulationSteps; ++i){
		accessIndices[i] = Rng::discrete(0,maxIndex-1);
		accessSizes[i] = Rng::discrete(1,3);
		flips[i].first = Rng::discrete(0,maxIndex-1);
		flips[i].second = Rng::discrete(0,maxIndex-1);
	}
	simulateCache(maxIndex, cacheSize,accessIndices,accessSizes,flips);
}

