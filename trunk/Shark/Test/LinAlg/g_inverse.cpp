#include <shark/LinAlg/Inverse.h>
#include <shark/LinAlg/rotations.h>

#define BOOST_TEST_MODULE LinAlg_g_inverse
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
using namespace shark;

template<class Vec,class Mat>
RealMatrix createSymmetricMatrix(Vec const& lambda,Mat const& rotation){
	RealMatrix intermediate=rotation;
	RealMatrix result(rotation.size1(),rotation.size2());
	zero(result);
	for(std::size_t i = 0; i != intermediate.size1(); ++i){
		row(intermediate,i) *= lambda(i);
	}
	fast_prod(trans(rotation),intermediate,result);
	return result;
}

BOOST_AUTO_TEST_CASE( LinAlg_decomposedGeneralInverse_fullRank ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 50;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable symmetric matrix A as well as its inverse
		RealVector lambda(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i) = 0.7*i+1;
		}
		RealMatrix R = blas::randomRotationMatrix(Dimensions);
		RealMatrix A = createSymmetricMatrix(lambda,R);
		
		RealMatrix U;
		decomposedGeneralInverse(A,U);
		
		//calculate inverse;
		RealMatrix AInv(Dimensions,Dimensions);
		fast_prod(U,trans(U),AInv);
		
		//should be left and right inverse
		RealMatrix resultLeft(Dimensions,Dimensions);
		RealMatrix resultRight(Dimensions,Dimensions);
		fast_prod(AInv,A,resultLeft);
		fast_prod(A,AInv,resultRight);
		
		double errorSame = norm_inf(resultLeft-resultRight);
		double errorLeft = norm_inf(resultLeft - RealIdentityMatrix(Dimensions));
		double errorRight = norm_inf(resultRight - RealIdentityMatrix(Dimensions));
		BOOST_CHECK_SMALL(errorSame,1.e-13);
		BOOST_CHECK_SMALL(errorLeft,1.e-10);
		BOOST_CHECK_SMALL(errorRight,1.e-10);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_decomposedGeneralInverse_RankK ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 50;
	std::size_t Rank = 30;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable symmetric matrix A as well as its inverse
		RealVector lambda(Dimensions,0.0);
		for(std::size_t i = 0; i != Rank; ++i){
			lambda(i) = 0.1*i+1;
		}
		RealMatrix R = blas::randomRotationMatrix(Dimensions);
		RealMatrix A = createSymmetricMatrix(lambda,R);
		
		RealMatrix U;
		decomposedGeneralInverse(A,U);
		
		//calculate inverse;
		RealMatrix AInv(Dimensions,Dimensions);
		
		fast_prod(U,trans(U),AInv);
		
		//Test whether A*AInv*A = A
		RealMatrix resultIntermediate(Dimensions,Dimensions);
		RealMatrix result(Dimensions,Dimensions);
		fast_prod(AInv,A,resultIntermediate);
		fast_prod(A,resultIntermediate,result);
		
		//std::cout<<result<<std::endl;
		
		double error = norm_inf(result - A);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

//first test that we can actually invert invertible matrices
BOOST_AUTO_TEST_CASE( LinAlg_g_inverse_Simple ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 50;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable symmetric matrix A as well as its inverse
		RealVector lambda(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i) = 0.7*i+1;
		}
		RealMatrix R = blas::randomRotationMatrix(Dimensions);
		RealMatrix A = createSymmetricMatrix(lambda,R);
		
		//analytic inverse
		RealVector lambdaInv(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambdaInv(i) = 1.0/lambda(i);
		}
		RealMatrix AInv = createSymmetricMatrix(lambdaInv,R);
		

		//calculate numeric inverse
		RealMatrix ATestInv = g_inverse(A);
		//if everything is correct, the matrix should be the same as our computed inverse
		//...hopefully
		double error = norm_inf(ATestInv - AInv);
		BOOST_CHECK_SMALL(error,1.e-11);
	}
}

//rank = m < n 
BOOST_AUTO_TEST_CASE( LinAlg_g_inverse_RankM ){
	std::size_t NumTests = 100;
	std::size_t M = 31;
	std::size_t N = 50;
	for(std::size_t test = 0; test != NumTests; ++test){
		//generate test input
		RealMatrix QTest = blas::randomRotationMatrix(N);
		RealMatrix RTest(M,N);
		RTest.clear();
		for(std::size_t i = 0; i != M; ++i){
			for(std::size_t j = 0; j != i; ++j){
				RTest(i,j) = Rng::gauss(0,1);
			}
			RTest(i,i) = 0.7*i+0.1;
		}
		RealMatrix ATest(M,N);
		fast_prod(RTest,QTest,ATest);
		

		//calculate numeric inverse
		RealMatrix ATestInv = g_inverse(ATest);
		//we can't test for matrix equality. but we can test pseudo inverse property
		RealMatrix result(M,M);
		result.clear();
		fast_prod(ATest,ATestInv,result,0);
		double error = norm_inf(result-RealIdentityMatrix(M));
		BOOST_CHECK_SMALL(error,1.e-11);
	}
}

RealMatrix croppedIdentity(std::size_t size, std::size_t rank){
	RealMatrix m(size,size);
	m.clear();
	for(std::size_t i = 0; i != rank; ++i){
			m(i,i) = 1;
	}
	return m;
}
//rank = n < m 
BOOST_AUTO_TEST_CASE( LinAlg_g_inverse_QR_RankN ){
	std::size_t NumTests = 100;
	std::size_t M = 50;
	std::size_t N = 31;
	for(std::size_t test = 0; test != NumTests; ++test){
		//generate test input
		RealMatrix QTest = blas::randomRotationMatrix(N);
		RealMatrix RTest(M,N);
		RTest.clear();
		for(std::size_t i = 0; i != N; ++i){
			for(std::size_t j = 0; j != std::min(i,N); ++j){
				RTest(i,j) = Rng::gauss(0,1);
			}
			if(i<N)
				RTest(i,i) = 0.7*(N-i);
		}
		
		RealMatrix ATest(M,N);
		fast_prod(RTest,QTest,ATest);
		

		//calculate numeric inverse
		RealMatrix ATestInv = g_inverse(ATest);
		//we can't test for matrix equality. but we can test pseudo inverse property
		RealMatrix result(M,M);
		result.clear();
		fast_prod(ATest,ATestInv,result,0);
		double error = norm_inf(result-croppedIdentity(M,N));
		BOOST_CHECK_SMALL(error,1.e-10);
	}
}

