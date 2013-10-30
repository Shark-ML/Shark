#define BOOST_TEST_MODULE BLAS_Vector_vector_iterators
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>

using namespace shark;
using namespace blas;

BOOST_AUTO_TEST_CASE( BLAS_Dense_Storage_Iterator)
{
	double values[]={0.1,0.2,0.3,0.4,0.5,0.6};
	
	//reading
	{
		dense_storage_iterator<const double> iter(values,1,2);
		dense_storage_iterator<const double> start=iter;
		dense_storage_iterator<const double> end(values,3,2);
		BOOST_REQUIRE_EQUAL(end-start, 2);
		BOOST_REQUIRE_EQUAL(start-iter, 0);
		BOOST_REQUIRE(start == iter);
		BOOST_REQUIRE(end != start);
		BOOST_REQUIRE(end == start+2);
		std::size_t k = 1;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),k);
			BOOST_CHECK_EQUAL(*iter,values[2*k]);
			BOOST_CHECK_EQUAL(start[k-1],values[2*k]);
			BOOST_CHECK_EQUAL(*(start+k-1),values[2*k]);
			BOOST_CHECK(iter < end);
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(end-iter, 0);
		BOOST_REQUIRE_EQUAL(iter-start, 2);
		BOOST_REQUIRE_EQUAL(k, 3);
	}
	
	//writing
	{
		dense_storage_iterator<double> iter(values,0,2);
		dense_storage_iterator<double> end(values,3,2);
		std::size_t k = 0;
		while(iter != end){
			*iter = k;
			++k;
			++iter;
		}
		for(std::size_t i = 0; i != 6; ++i){
			if(i% 2 == 0)
				BOOST_CHECK_EQUAL(values[i],i/2);
			else
				BOOST_CHECK_CLOSE(values[i],i*0.1+0.1, 1.e-10);
		}
	}
}

BOOST_AUTO_TEST_CASE( BLAS_Compressed_Storage_Iterator)
{
	double values[]={0.1,0.2,0.3,0.4,0.5,0.6};
	std::size_t indizes[]={3,8,11,12,15,16};
	
	//reading
	{
		compressed_storage_iterator<const double,const std::size_t> iter(values,indizes,1,2);
		compressed_storage_iterator<const double,const std::size_t> start=iter;
		compressed_storage_iterator<const double,const std::size_t> end(values,indizes,5,2);
		BOOST_REQUIRE_EQUAL(start.row(), 2);
		BOOST_REQUIRE_EQUAL(start-iter, 0);
		BOOST_REQUIRE_EQUAL(end-start, 4);
		BOOST_REQUIRE(start == iter);
		BOOST_REQUIRE(end != start);
		std::size_t k = 1;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),indizes[k]);
			BOOST_CHECK_EQUAL(*iter,values[k]);
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(end-iter, 0);
		BOOST_REQUIRE_EQUAL(iter-start, 4);
		BOOST_REQUIRE_EQUAL(k, 5);
	}
	//writing
	{
		compressed_storage_iterator<double,const std::size_t> iter(values,indizes,1,2);
		compressed_storage_iterator<double,const std::size_t> end(values,indizes,5,2);
		std::size_t k = 1;
		while(iter != end){
			*iter = 2*k;
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(k, 5);
		for(std::size_t i = 1;  i !=5; ++i){
			BOOST_CHECK_EQUAL(values[i],2*i); 
		}
	}
}

struct IndexedMocup{
	typedef double value_type;
	typedef double& reference;
	typedef double const& const_reference;
	
	IndexedMocup(double* array,std::size_t size):m_array(array),m_size(size){}
	
	std::size_t size()const{
		return m_size;
	}
	reference operator()(std::size_t i)const{
		return m_array[i];
	}
	
	bool same_closure(IndexedMocup const& other)const{
		return m_array == other.m_array;
	}
	
	double* m_array;
	std::size_t m_size;
};

struct ConstIndexedMocup{
	typedef double value_type;
	typedef double& reference;
	typedef double const& const_reference;
	
	ConstIndexedMocup(double* array,std::size_t size):m_array(array),m_size(size){}
	
	std::size_t size()const{
		return m_size;
	}
	const_reference operator()(std::size_t i)const{
		return m_array[i];
	}
	
	bool same_closure(ConstIndexedMocup const& other)const{
		return m_array == other.m_array;
	}
	
	double* m_array;
	std::size_t m_size;
};




BOOST_AUTO_TEST_CASE( BLAS_Indexed_Iterator)
{
	double values[]={0.1,0.2,0.3,0.4,0.5,0.6};
	
	//reading
	{
		ConstIndexedMocup mocup(values,6);
		indexed_iterator<const ConstIndexedMocup> iter(mocup,1);
		indexed_iterator<const ConstIndexedMocup> start=iter;
		indexed_iterator<const ConstIndexedMocup> end(mocup,5);
		BOOST_REQUIRE_EQUAL(end-start, 4);
		BOOST_REQUIRE_EQUAL(start-iter, 0);
		BOOST_REQUIRE(start == iter);
		BOOST_REQUIRE(end != start);
		BOOST_REQUIRE(start < end);
		BOOST_REQUIRE(end == start+4);
		std::size_t k = 1;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),k);
			BOOST_CHECK_EQUAL(*iter,values[k]);
			BOOST_CHECK_EQUAL(start[k-1],values[k]);
			BOOST_CHECK_EQUAL(*(start+k-1),values[k]);
			BOOST_CHECK(iter < end);
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(end-iter, 0);
		BOOST_REQUIRE_EQUAL(iter-start, 4);
		BOOST_REQUIRE_EQUAL(k, 5);
	}
	
	//writing
	{
		IndexedMocup mocup(values,6);
		indexed_iterator<IndexedMocup> iter(mocup,1);
		indexed_iterator<IndexedMocup> end(mocup,5);
		std::size_t k = 1;
		while(iter != end){
			*iter = 2*k;
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(k, 5);
		for(std::size_t i = 1;  i !=5; ++i){
			BOOST_CHECK_EQUAL(values[i],2*i); 
		}
	}
	
}

BOOST_AUTO_TEST_CASE( BLAS_Constant_Iterator)
{
	constant_iterator<double> iter(3,4.0);
	constant_iterator<double> start =iter;
	constant_iterator<double> end(10,4.0);
	BOOST_REQUIRE_EQUAL(start-iter, 0);
	BOOST_REQUIRE_EQUAL(end-start, 7);
	BOOST_REQUIRE(start == iter);
	BOOST_REQUIRE(end != start);
	BOOST_REQUIRE(start < end);
	std::size_t k = 3;
	while(iter != end){
		BOOST_CHECK_EQUAL(iter.index(),k);
		BOOST_CHECK_EQUAL(*iter,4.0);
		++iter;
		++k;
	}
	BOOST_REQUIRE_EQUAL(end-iter, 0);
	BOOST_REQUIRE_EQUAL(iter-start, 7);
	BOOST_REQUIRE_EQUAL(k, 10);
}

BOOST_AUTO_TEST_CASE( BLAS_Subrange_Iterator_Dense)
{
	double values[]={0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
	//reading
	{
		typedef dense_storage_iterator<const double> iterator;
		iterator inner_iter(values,0);
		iterator inner_end(values,10);
		subrange_iterator<iterator> iter(inner_iter,inner_end,2,2);
		subrange_iterator<iterator> start = iter;
		subrange_iterator<iterator> end(inner_iter,inner_end,8,2);
		BOOST_REQUIRE(start == iter);
		BOOST_REQUIRE(end != start);
		BOOST_REQUIRE(start < end);
		std::size_t k = 2;
		inner_iter+=2;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),k-2);
			BOOST_CHECK(iter.inner() == inner_iter);
			BOOST_CHECK_EQUAL(*iter,values[k]);
			++iter;
			++inner_iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(end-iter, 0);
		BOOST_REQUIRE_EQUAL(iter-start, 6);
		BOOST_REQUIRE_EQUAL(k, 8);
	}
	
	//reading full range
	{
		typedef dense_storage_iterator<double> iterator;
		iterator inner_iter(values,0);
		iterator inner_end(values,10);
		subrange_iterator<iterator> iter(inner_iter,inner_end,0,0);
		subrange_iterator<iterator> end(inner_end,0);
		std::size_t k = 0;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),k);
			BOOST_CHECK(iter.inner() == inner_iter);
			BOOST_CHECK_EQUAL(*iter,values[k]);
			++iter;
			++inner_iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(end-iter, 0);
		BOOST_REQUIRE_EQUAL(k, 10);
	}
	
	//writing
	{
		typedef dense_storage_iterator<double> iterator;
		iterator inner_iter(values,0);
		iterator inner_end(values,10);
		subrange_iterator<iterator> iter(inner_iter,inner_end,2,2);
		subrange_iterator<iterator> end(inner_iter,inner_end,8,2);
		std::size_t k = 2;
		while(iter != end){
			*iter = 2*k;
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(k, 8);
		for(std::size_t i = 2;  i !=8; ++i){
			BOOST_CHECK_EQUAL(values[i],2*i); 
		}
	}
}

BOOST_AUTO_TEST_CASE( BLAS_Subrange_Iterator_Compressed)
{
	double values[]={0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
	std::size_t indizes[]={3,8,11,12,15,16,18,20,23,30};
	//reading full range (but starting from different startIndex)
	{
		typedef compressed_storage_iterator<const double,const std::size_t> iterator;
		iterator inner_iter(values,indizes,0);
		iterator inner_end(values,indizes,10);
		subrange_iterator<iterator> iter(inner_iter,3);
		subrange_iterator<iterator> end(inner_end,3);
		std::size_t k = 0;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),indizes[k]-3);
			BOOST_CHECK(iter.inner() == inner_iter);
			BOOST_CHECK_EQUAL(*iter,values[k]);
			++iter;
			++inner_iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(k, 10);
	}
	
	//reading full range (but starting from different startIndex)
	{
		typedef compressed_storage_iterator<const double,const std::size_t> iterator;
		iterator inner_iter(values,indizes,0);
		iterator inner_end(values,indizes,10);
		iterator inner_iter_test(values,indizes,4);
		subrange_iterator<iterator> iter(inner_iter,inner_end,13,10);
		subrange_iterator<iterator> end(inner_iter,inner_end,23,10);
		//check that we got to the correct array positions
		BOOST_REQUIRE_EQUAL(iter.inner().index(),15);
		BOOST_REQUIRE_EQUAL(end.inner().index(),23);
		
		std::size_t k = 4;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),indizes[k]-10);
			BOOST_CHECK(iter.inner() == inner_iter_test);
			BOOST_CHECK_EQUAL(*iter,values[k]);
			++iter;
			++inner_iter_test;
			++k;
		}
		BOOST_REQUIRE_EQUAL(k, 8);
	}
	
	//writing
	{
		typedef compressed_storage_iterator<double,const std::size_t> iterator;
		iterator inner_iter(values,indizes,0);
		iterator inner_end(values,indizes,10);
		iterator inner_iter_test(values,indizes,4);
		subrange_iterator<iterator> iter(inner_iter,inner_end,13,10);
		subrange_iterator<iterator> end(inner_iter,inner_end,23,10);
		//check that we got to the correct array positions
		BOOST_REQUIRE_EQUAL(iter.inner().index(),15);
		BOOST_REQUIRE_EQUAL(end.inner().index(),23);
		std::size_t k = 4;
		while(iter != end){
			*iter = 2*k;
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(k, 8);
		for(std::size_t i = 4;  i !=8; ++i){
			BOOST_CHECK_EQUAL(values[i],2*i); 
		}
	}
	
}

BOOST_AUTO_TEST_CASE( BLAS_Transform_Iterator_Dense)
{
	double values[]={0.1,0.2,0.3,0.4,0.5,0.6};

	typedef dense_storage_iterator<const double> iterator;
	iterator dense_iter(values,0);
	iterator dense_end(values,6);
	transform_iterator<iterator,scalar_sqr<double> > iter(dense_iter,scalar_sqr<double>());
	transform_iterator<iterator,scalar_sqr<double> > start = iter;
	transform_iterator<iterator,scalar_sqr<double> > end(dense_end,scalar_sqr<double>());
	
	BOOST_REQUIRE_EQUAL(end-start, 6);
	BOOST_REQUIRE_EQUAL(start-iter, 0);
	BOOST_REQUIRE(start == iter);
	BOOST_REQUIRE(start != end);
	BOOST_REQUIRE(start < end);
	BOOST_REQUIRE(end == start+6);
	std::size_t k = 0;
	while(iter != end){
		BOOST_CHECK_EQUAL(iter.index(),k);
		BOOST_CHECK_EQUAL(*iter,values[k]*values[k]);
		BOOST_CHECK_EQUAL(start[k],values[k]*values[k]);
		BOOST_CHECK_EQUAL(*(start+k),values[k]*values[k]);
		BOOST_CHECK(iter < end);
		++iter;
		++k;
	}
	BOOST_REQUIRE_EQUAL(k, 6);
	BOOST_REQUIRE_EQUAL(end-iter, 0);
	BOOST_REQUIRE_EQUAL(iter-start, 6);
}

BOOST_AUTO_TEST_CASE( BLAS_Transform_Iterator_Compressed)
{
	double values[]={0.1,0.2,0.3,0.4,0.5,0.6};
	std::size_t indizes[]={3,8,11,12,15,16};
	
	typedef compressed_storage_iterator<const double,const std::size_t> iterator;
	iterator compressed_iter(values,indizes,0);
	iterator compressed_end(values,indizes,6);
	transform_iterator<iterator,scalar_sqr<double> > iter(compressed_iter,scalar_sqr<double>());
	transform_iterator<iterator,scalar_sqr<double> > start = iter;
	transform_iterator<iterator,scalar_sqr<double> > end(compressed_end,scalar_sqr<double>());
	
	BOOST_REQUIRE_EQUAL(end-start, 6);
	BOOST_REQUIRE_EQUAL(start-iter, 0);
	BOOST_REQUIRE(start == iter);
	BOOST_REQUIRE(start != end);
	std::size_t k = 0;
	while(iter != end){
		BOOST_CHECK_EQUAL(iter.index(),indizes[k]);
		BOOST_CHECK_EQUAL(*iter,values[k]*values[k]);
		++iter;
		++k;
	}
	BOOST_REQUIRE_EQUAL(k, 6);
	BOOST_REQUIRE_EQUAL(end-iter, 0);
	BOOST_REQUIRE_EQUAL(iter-start, 6);
}

BOOST_AUTO_TEST_CASE( BLAS_Binary_Transform_Iterator_Dense)
{
	double values1[]={0.1,0.2,0.3,0.4,0.5,0.6};
	double values2[]={0.3,0.5,0.7,0.9,1.1,1.3};
	

	typedef dense_storage_iterator<const double> iterator;
	iterator dense_iter1(values1,0);
	iterator dense_end1(values1,6);
	iterator dense_iter2(values2,0);
	iterator dense_end2(values2,6);
	
	typedef binary_transform_iterator<iterator,iterator,scalar_binary_plus<double,double> > transform_iterator;
	
	transform_iterator iter(scalar_binary_plus<double,double>(),dense_iter1,dense_end1,dense_iter2,dense_end2);
	transform_iterator start = iter;
	transform_iterator end(scalar_binary_plus<double,double>(),dense_end1,dense_end1,dense_end2,dense_end2);
	
	BOOST_REQUIRE_EQUAL(end-start, 6);
	BOOST_REQUIRE_EQUAL(start-iter, 0);
	BOOST_REQUIRE(start == iter);
	BOOST_REQUIRE(start != end);
	BOOST_REQUIRE(start < end);
	BOOST_REQUIRE(end == start+6);
	std::size_t k = 0;
	while(iter != end){
		double value = values1[k]+values2[k];
		BOOST_CHECK_EQUAL(iter.index(),k);
		BOOST_CHECK_EQUAL(*iter,value);
		BOOST_CHECK_EQUAL(start[k],value);
		BOOST_CHECK_EQUAL(*(start+k),value);
		BOOST_CHECK(iter < end);
		++iter;
		++k;
	}
	BOOST_REQUIRE_EQUAL(k, 6);
	BOOST_REQUIRE_EQUAL(end-iter, 0);
	BOOST_REQUIRE_EQUAL(iter-start, 6);
}

BOOST_AUTO_TEST_CASE( BLAS_Binary_Transform_Iterator_Compressed)
{
	double values1[]={0.1,0.2,0.3,0.4,0.5,0.6};
	double values2[]={0.3,0.5,0.7,0.9};
	double valuesResult[]={0.1,0.3,0.7,1.0,0.4,0.9,0.5,0.6};
	
	std::size_t indizes1[]={3,8,11,12,17,18};
	std::size_t indizes2[]={5,8,11,14};
	std::size_t indizesResult[]={3,5,8,11,12,14,17,18};
	

	typedef compressed_storage_iterator<const double,const std::size_t> iterator;
	typedef binary_transform_iterator<iterator,iterator,scalar_binary_plus<double,double> > transform_iterator;
	
	//a+b
	{
		iterator iter1(values1,indizes1,0);
		iterator end1(values1,indizes1,6);
		iterator iter2(values2,indizes2,0);
		iterator end2(values2,indizes2,4);
		transform_iterator iter(scalar_binary_plus<double,double>(),iter1,end1,iter2,end2);
		transform_iterator start = iter;
		transform_iterator end(scalar_binary_plus<double,double>(),end1,end1,end2,end2);
		
		BOOST_REQUIRE(start == iter);
		BOOST_REQUIRE(start != end);
		std::size_t k = 0;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),indizesResult[k]);
			BOOST_CHECK_EQUAL(*iter,valuesResult[k]);
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(k, 8);
	}
	
	//b+a
	{
		iterator iter2(values1,indizes1,0);
		iterator end2(values1,indizes1,6);
		iterator iter1(values2,indizes2,0);
		iterator end1(values2,indizes2,4);
		transform_iterator iter(scalar_binary_plus<double,double>(),iter1,end1,iter2,end2);
		transform_iterator start = iter;
		transform_iterator end(scalar_binary_plus<double,double>(),end1,end1,end2,end2);
		
		BOOST_REQUIRE(start == iter);
		BOOST_REQUIRE(start != end);
		std::size_t k = 0;
		while(iter != end){
			BOOST_CHECK_EQUAL(iter.index(),indizesResult[k]);
			BOOST_CHECK_EQUAL(*iter,valuesResult[k]);
			++iter;
			++k;
		}
		BOOST_REQUIRE_EQUAL(k, 8);
	}
}