
#define BOOST_TEST_MODULE ML_ProductKernel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "KernelDerivativeTestHelper.h"

#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/MonomialKernel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/ProductKernel.h>

using namespace shark;

// This unit test checks whether the product
// kernel class correctly computes the product
// of its component kernels.
BOOST_AUTO_TEST_SUITE (Models_Kernels_ProductKernel)

BOOST_AUTO_TEST_CASE( ProductKernel_Test )
{

	// define three basis kernels
	LinearKernel<RealVector> k1;
	MonomialKernel<RealVector> k2(3);
	GaussianRbfKernel<RealVector> k3(1.0);

	// define a product kernel
	std::vector<AbstractKernelFunction<RealVector>* > kernels;
	kernels.push_back(&k1); kernels.push_back(&k2); kernels.push_back(&k3);
	ProductKernel<RealVector> kp(kernels);
	
	RealMatrix batchX1(10,3);
	RealMatrix batchX2(12,3);
	for(std::size_t i = 0; i != 10; ++i){
		batchX1(i,0) = Rng::uni(-1,1); 
		batchX1(i,1) = Rng::uni(-1,1); 
		batchX1(i,2) = Rng::uni(-1,1);
	}
	for(std::size_t i = 0; i != 12; ++i){
		batchX2(i,0) = Rng::uni(-1,1); 
		batchX2(i,1) = Rng::uni(-1,1); 
		batchX2(i,2) = Rng::uni(-1,1);
	}
	
	//Evaluate the kernel matrices by hand
	RealMatrix matK1 = k1(batchX1, batchX2);
	RealMatrix matK2 = k2(batchX1, batchX2);
	RealMatrix matK3 = k3(batchX1, batchX2);
	RealMatrix matTest = element_prod(matK1,element_prod(matK2,matK3));
	
	// test the product kernel
	RealMatrix matKp = kp(batchX1,batchX2);
	BOOST_REQUIRE_EQUAL(matKp.size1(),10);
	BOOST_REQUIRE_EQUAL(matKp.size2(),12);
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 12; ++j)
			BOOST_CHECK_SMALL(matTest(i,j)-matKp(i,j),1.e-13);
	}
	testEval(kp,batchX1,batchX2);
}

BOOST_AUTO_TEST_SUITE_END()
