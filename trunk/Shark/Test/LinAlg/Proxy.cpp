#define BOOST_TEST_MODULE LinAlg_Proxy
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/Base.h>

using namespace shark;
using namespace std;

BOOST_AUTO_TEST_CASE( LinAlg_FixedDenseVectorProxy )
{
	double mem[]={0,4,1,4,2,4,3,4};
	RealVector x(3); 
	RealMatrix xm(3,3);
	RealVector result(3);
	RealVector resultProxy(3);
	
	for(std::size_t i = 0; i != 3; ++i){
		x(i) = i;
		for(std::size_t j = 0; j != 3; ++j){
			xm(i,j) = j+3*i;
		}
	}
	
	//create some proxies
	blas::FixedDenseVectorProxy<double> proxy(x);
	blas::FixedDenseVectorProxy<double> proxym(mem,4,2);
	
	//check internal variables
	BOOST_REQUIRE_EQUAL(proxy.size() ,3);
	BOOST_REQUIRE_EQUAL(proxy.stride() ,1);
	BOOST_REQUIRE_EQUAL(proxy.data() ,&x(0));
	BOOST_REQUIRE_EQUAL(proxym.size() ,4);
	BOOST_REQUIRE_EQUAL(proxym.stride() ,2);
	BOOST_REQUIRE_EQUAL(proxym.data() ,mem);
	
	//check traits
	BOOST_REQUIRE_EQUAL(blas::traits::vector_stride(proxy) ,1);
	BOOST_REQUIRE_EQUAL(blas::traits::vector_storage(proxy) ,&x(0));
	BOOST_REQUIRE_EQUAL(blas::traits::vector_stride(proxym) ,2);
	BOOST_REQUIRE_EQUAL(blas::traits::vector_storage(proxym) ,mem);
	
	//check values
	for(std::size_t i = 0; i != 3; ++i){
		BOOST_REQUIRE_EQUAL(x(i) ,proxy(i));
		BOOST_REQUIRE_EQUAL(x(i) ,const_cast<blas::FixedDenseVectorProxy<double> const&>(proxy)(i));
		BOOST_REQUIRE_EQUAL(x(i) ,proxy[i]);
		BOOST_REQUIRE_EQUAL(x[i] ,const_cast<blas::FixedDenseVectorProxy<double> const&>(proxy)[i]);
	}
	for(std::size_t i = 0; i != 4; ++i){
		BOOST_REQUIRE_EQUAL(mem[2*i] ,proxym(i));
	}
	
	fast_prod(xm,x,result);
	fast_prod(xm,proxy,resultProxy);
	
	for(std::size_t i = 0; i != 3; ++i){
		BOOST_REQUIRE_EQUAL(result(i) ,resultProxy(i));
	}
}

template<class Proxy, class Vector>
void checkProxyBase(Vector& vec, std::size_t nnz){
	Proxy proxy(vec);
	//check basic properties
	BOOST_REQUIRE_EQUAL(proxy.nnz(),nnz);
	BOOST_REQUIRE_EQUAL(proxy.size(),vec.size());
	
	for(std::size_t i = 0; i != vec.size(); ++i){
		BOOST_CHECK_EQUAL(proxy(i), vec(i));
		BOOST_CHECK_EQUAL(proxy[i], vec[i]);
	}
	//check iterators
	typename boost::range_iterator<Proxy>::type proxyiter = proxy.begin();
	typename boost::range_iterator<Vector>::type vectoriter = vec.begin();
	
	for(;vectoriter != vec.end(); ++vectoriter, ++ proxyiter){
		BOOST_REQUIRE(proxyiter != proxy.end());
		BOOST_CHECK_EQUAL(vectoriter.index(), proxyiter.index());
		BOOST_CHECK_EQUAL(*vectoriter, *proxyiter);
	}
	BOOST_CHECK(proxyiter  == proxy.end());
}
template<class Vector>
void checkProxy(Vector const& vec, std::size_t nnz){
	checkProxyBase<blas::FixedSparseVectorProxy<double,std::size_t> >(vec,nnz);
	checkProxyBase<blas::FixedSparseVectorProxy<const double,std::size_t> >(vec,nnz);
	checkProxyBase<const blas::FixedSparseVectorProxy<const double,std::size_t> >(vec,nnz);
	checkProxyBase<const blas::FixedSparseVectorProxy<const double,std::size_t> >(vec,nnz);
	
	//check proxy conversion
	const blas::FixedSparseVectorProxy<const double,std::size_t> proxy(vec);
	checkProxyBase<blas::FixedSparseVectorProxy<double,std::size_t> >(proxy,nnz);
	checkProxyBase<blas::FixedSparseVectorProxy<const double,std::size_t> >(proxy,nnz);
	checkProxyBase<const blas::FixedSparseVectorProxy<const double,std::size_t> >(proxy,nnz);
	checkProxyBase<const blas::FixedSparseVectorProxy<const double,std::size_t> >(proxy,nnz);
	
	blas::FixedSparseVectorProxy<const double,std::size_t> cproxy(vec);
	checkProxyBase<blas::FixedSparseVectorProxy<double,std::size_t> >(cproxy,nnz);
	checkProxyBase<blas::FixedSparseVectorProxy<const double,std::size_t> >(cproxy,nnz);
	checkProxyBase<const blas::FixedSparseVectorProxy<const double,std::size_t> >(cproxy,nnz);
	checkProxyBase<const blas::FixedSparseVectorProxy<const double,std::size_t> >(cproxy,nnz);
}


BOOST_AUTO_TEST_CASE( LinAlg_FixedSparseVectorProxy )
{
	//som vectors to test
	CompressedRealVector x1(6);
	x1(3) = 1.5;
	CompressedRealVector x2(6);
	x2(3) = 2.0;
	x2(5) = 1.0;
	
	CompressedRealVector x3(6);//empty
	CompressedRealVector x4(6);//full
	for(std::size_t i = 0; i != 6; ++i)
		x4(i) = i;
	CompressedRealVector x5(10);
	x5(0) = 1.5;//only first element
	CompressedRealVector x6(10);
	x6(9) = 1.5;//only last element
	
	checkProxy(x1,1);
	checkProxy(x2,2);
	checkProxy(x3,0);
	checkProxy(x4,6);
	checkProxy(x5,1);
	checkProxy(x6,1);
}

BOOST_AUTO_TEST_CASE( LinAlg_FixedSparseVectorProxy_MatrixRow )
{
	//som vectors to test
	CompressedRealMatrix x1(5,6); //(last line is also empty)
	x1(0,3) = 1.5;
	x1(1,3) = 2.0;
	x1(1,5) = 1.0;
	for(std::size_t i = 0; i != 6; ++i)
		x1(3,i) = i;
	
	CompressedRealMatrix x2(2,10);
	x2(0,0) = 1.5;//only first element
	x2(1,9) = 1.5;//only last element
	
	checkProxy(row(x1,0),1);
	checkProxy(row(x1,1),2);
	checkProxy(row(x1,2),0);
	checkProxy(row(x1,3),6);
	checkProxy(row(x1,4),0);
	checkProxy(row(x2,0),1);
	checkProxy(row(x2,1),1);
}

