#define BOOST_TEST_MODULE CoreScopedHandleTestModule
#include <boost/bind.hpp>
#include <boost/test/unit_test.hpp>

#include <shark/Core/utility/Iterators.h>
#include <shark/Rng/GlobalRng.h>
using namespace shark;

BOOST_AUTO_TEST_CASE(MULTI_SEQUENCE_ITERATOR_TEST)
{
	std::vector<std::vector<std::size_t> > vecs(10);
	std::vector<std::size_t> values(1000);
	//~ for(std::size_t i = 0; i != 1000; ++i){
		//~ std::size_t bin = Rng::discrete(0,9);
		//~ vecs[bin].push_back(i);
	//~ }
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 100; ++j){
			vecs[i].push_back(i*100+j);
		}
	}
	{
		std::vector<std::size_t>::iterator valPos = values.begin();
		for(std::size_t i = 0; i != 10; ++i){
			std::copy(vecs[i].begin(),vecs[i].end(),valPos);
			valPos +=vecs[i].size();
		}
	}
	
	{//check op++
		MultiSequenceIterator<std::vector<std::vector<std::size_t> > > iter(
			vecs, vecs.begin(),vecs.begin()->begin(),0
		);
		
		MultiSequenceIterator<std::vector<std::vector<std::size_t> > > end(
			vecs, vecs.end(),std::vector<std::size_t>::iterator(),1000
		);
		
		for(std::size_t pos = 0; pos != 1000; ++pos,++iter){

			std::size_t value = values[pos];
			BOOST_CHECK_EQUAL(*iter,value);
			BOOST_CHECK_EQUAL(iter.index(),pos);
			BOOST_CHECK_EQUAL(end-iter,1000-pos);
		}
		BOOST_CHECK(iter == end);
	}
	
	{//check op+=
		MultiSequenceIterator<std::vector<std::vector<std::size_t> > > iter(
			vecs, vecs.begin(),vecs.begin()->begin(),0
		);
		
		MultiSequenceIterator<std::vector<std::vector<std::size_t> > > end(
			vecs, vecs.end(),std::vector<std::size_t>::iterator(),1000
		);
		
		std::ptrdiff_t pos = 0;
		for(std::size_t trial = 0; trial != 10000; ++trial){
			std::size_t value = values[pos];
			BOOST_CHECK_EQUAL(*iter,value);
			BOOST_CHECK_EQUAL(iter.index(),pos);
			BOOST_CHECK_EQUAL(end-iter,1000-pos);
			
			std::ptrdiff_t newPos = Rng::discrete(0,999);
			std::ptrdiff_t diff = newPos - pos;
			std::cout<<diff<<" "<<pos<<" "<<newPos<<std::endl;
			iter+=diff;
			pos =newPos;
		}
	}
	
}

