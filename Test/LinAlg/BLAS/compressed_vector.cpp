#define BOOST_TEST_MODULE LinAlg_VectorSparse
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>

using namespace shark;
using namespace blas;



//this test tests push_back behavior of set_element and operator()
BOOST_AUTO_TEST_SUITE (LinAlg_BLAS_compressed_vector)

BOOST_AUTO_TEST_CASE( LinAlg_sparse_vector_insert_element_end){
	std::size_t dimensions = 20;
	
	compressed_vector<std::size_t> vector_set(dimensions);
	compressed_vector<std::size_t> vector_operator(dimensions);
	compressed_vector<std::size_t>::iterator iter = vector_set.begin();
	for(std::size_t i = 0; i != dimensions; ++i){
		//check set_element
		iter = vector_set.set_element(iter,i,3*i);
		BOOST_REQUIRE_EQUAL(iter.index(), i);
		BOOST_REQUIRE_EQUAL(iter-vector_set.begin(), i);
		BOOST_REQUIRE_EQUAL(*iter, 3*i);
		BOOST_REQUIRE_EQUAL(vector_set.nnz(),i+1);
		for(std::size_t k = 0; k <=i; ++k){
			BOOST_CHECK_EQUAL(vector_set.raw_storage().values[k], 3*k);
			BOOST_CHECK_EQUAL(vector_set.raw_storage().indices[k], k);
		}
		++iter;
		BOOST_REQUIRE_EQUAL(iter-vector_set.begin(), i+1);
	}
	
	for(std::size_t i = 0; i != dimensions; ++i){
		//check operator()
		vector_operator(i) = 3*i;
		BOOST_REQUIRE_EQUAL(vector_operator(i), 3*i);
		BOOST_REQUIRE_EQUAL(vector_operator.nnz(),i+1);
		for(std::size_t k = 0; k <=i; ++k){
			BOOST_CHECK_EQUAL(vector_operator.raw_storage().values[k], 3*k);
			BOOST_CHECK_EQUAL(vector_operator(k), 3*k);
			BOOST_CHECK_EQUAL(vector_operator.raw_storage().indices[k], k);
		}
	}
}


BOOST_AUTO_TEST_SUITE_END()
