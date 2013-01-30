#define BOOST_TEST_MODULE LinAlg_Proxy
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/Base.h>

using namespace shark;
using namespace std;

BOOST_AUTO_TEST_CASE( LinAlg_FixedVectorProxy )
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
	FixedDenseVectorProxy<double> proxy(x);
	FixedDenseVectorProxy<double> proxym(mem,4,2);
	
	//check internal variables
	BOOST_REQUIRE_EQUAL(proxy.size() ,3);
	BOOST_REQUIRE_EQUAL(proxy.stride() ,1);
	BOOST_REQUIRE_EQUAL(proxy.data() ,&x(0));
	BOOST_REQUIRE_EQUAL(proxym.size() ,4);
	BOOST_REQUIRE_EQUAL(proxym.stride() ,2);
	BOOST_REQUIRE_EQUAL(proxym.data() ,mem);
	
	//check traits
	BOOST_REQUIRE_EQUAL(traits::vector_stride(proxy) ,1);
	BOOST_REQUIRE_EQUAL(traits::vector_storage(proxy) ,&x(0));
	BOOST_REQUIRE_EQUAL(traits::vector_stride(proxym) ,2);
	BOOST_REQUIRE_EQUAL(traits::vector_storage(proxym) ,mem);
	
	//check values
	for(std::size_t i = 0; i != 3; ++i){
		BOOST_REQUIRE_EQUAL(x(i) ,proxy(i));
		BOOST_REQUIRE_EQUAL(x(i) ,const_cast<FixedDenseVectorProxy<double> const&>(proxy)(i));
		BOOST_REQUIRE_EQUAL(x(i) ,proxy[i]);
		BOOST_REQUIRE_EQUAL(x[i] ,const_cast<FixedDenseVectorProxy<double> const&>(proxy)[i]);
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

