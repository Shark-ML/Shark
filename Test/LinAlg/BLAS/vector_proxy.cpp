#define BOOST_TEST_MODULE LinAlg_Vector_Proxy
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>

using namespace shark;

template<class V1, class V2>
void checkDenseVectorEqual(V1 const& v1, V2 const& v2){
	BOOST_REQUIRE_EQUAL(v1.size(),v2.size());
	//indexed access
	for(std::size_t i = 0; i != v2.size(); ++i){
		BOOST_CHECK_EQUAL(v1(i),v2(i));
	}
	//iterator accessranges
	typedef typename V1::const_iterator Iter;
	BOOST_REQUIRE_EQUAL(v1.end()-v1.begin(), v1.size());
	std::size_t k = 0;
	for(Iter it = v1.begin(); it != v1.end(); ++it,++k){
		BOOST_CHECK_EQUAL(k,it.index());
		BOOST_CHECK_EQUAL(*it,v2(k));
	}
	//test that the actual iterated length equals the number of elements
	BOOST_CHECK_EQUAL(k, v2.size());
}

template<class V1, class V2>
void checkDenseVectorAssignment(V1& v1, V2 const& v2){
	BOOST_REQUIRE_EQUAL(v1.size(),v2.size());
	//indexed access
	for(std::size_t i = 0; i != v2.size(); ++i){
		v1(i) = 0;
		BOOST_CHECK_EQUAL(v1(i),0);
		v1(i) = v2(i);
		BOOST_CHECK_EQUAL(v1(i),v2(i));
		v1(i) = 0;
		BOOST_CHECK_EQUAL(v1(i),0);
	}
	//iterator accessranges
	typedef typename V1::iterator Iter;
	BOOST_REQUIRE_EQUAL(v1.end()-v1.begin(), v1.size());
	std::size_t k = 0;
	for(Iter it = v1.begin(); it != v1.end(); ++it,++k){
		BOOST_CHECK_EQUAL(k,it.index());
		*it = 0;
		BOOST_CHECK_EQUAL(v1(k),0);
		*it = v2(k);
		BOOST_CHECK_EQUAL(v1(k),v2(k));
		*it = 0;
		BOOST_CHECK_EQUAL(v1(k),0);
	}
	//test that the actual iterated length equals the number of elements
	BOOST_CHECK_EQUAL(k, v2.size());
}

std::size_t Dimensions = 8;
struct VectorProxyFixture
{
	blas::vector<double> denseData;
	blas::compressed_vector<double> compressedData;
	
	VectorProxyFixture():denseData(Dimensions){
		for(std::size_t i = 0; i!= Dimensions;++i){
			denseData(i) = i+5;
		}
	}
};

BOOST_FIXTURE_TEST_SUITE (LinAlg_BLAS_vector_proxy, VectorProxyFixture);

BOOST_AUTO_TEST_CASE( LinAlg_Dense_Subrange ){
	//all possible combinations of ranges on the data vector
	for(std::size_t rangeEnd=0;rangeEnd!= Dimensions;++rangeEnd){
		for(std::size_t rangeBegin =0;rangeBegin <=rangeEnd;++rangeBegin){//<= for 0 range
			std::size_t size=rangeEnd-rangeBegin;
			blas::vector<double> vTest(size);
			for(std::size_t i = 0; i != size; ++i){
				vTest(i) = denseData(i+rangeBegin);
			}
			checkDenseVectorEqual(subrange(denseData,rangeBegin,rangeEnd),vTest);
			
			//assignment using op() and iterators
			{
				blas::vector<double> newData(Dimensions,1.0);
				blas::vector_range<blas::vector<double> > rangeTest = subrange(newData,rangeBegin,rangeEnd);
				checkDenseVectorAssignment(rangeTest,vTest);//cehcks op() and iterators for assignment
				
				//check that after assignment all elements outside the range are still intact
				for(std::size_t i = 0; i != rangeBegin; ++i){
					BOOST_CHECK_EQUAL(newData(i),1.0);
				}
				for(std::size_t i = rangeEnd; i != Dimensions; ++i){
					BOOST_CHECK_EQUAL(newData(i),1.0);
				}
			}
			
			//check clear
			{
			
				blas::vector<double> newData(Dimensions,1.0);
				blas::vector_range<blas::vector<double> > rangeTest = subrange(newData,rangeBegin,rangeEnd);
				
				rangeTest.clear();
				for(std::size_t i = 0; i != size; ++i){
					BOOST_CHECK_EQUAL(rangeTest(i),0);
					BOOST_CHECK_EQUAL(newData(i+rangeBegin),0);
				}
				
				//check that after clear all elements outside the range are still intact
				for(std::size_t i = 0; i != rangeBegin; ++i){
					BOOST_CHECK_EQUAL(newData(i),1.0);
				}
				for(std::size_t i = rangeEnd; i != Dimensions; ++i){
					BOOST_CHECK_EQUAL(newData(i),1.0);
				}
			}
		}
	}
}

BOOST_AUTO_TEST_SUITE_END();