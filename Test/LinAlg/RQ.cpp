#include <shark/LinAlg/RQ.h>
#include <shark/Rng/GlobalRng.h>
#define BOOST_TEST_MODULE LinAlg_RQ
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

RealMatrix createRandomTriangularMatrix(RealMatrix const& lambda,std::size_t Dimensions){
	RealMatrix R = lambda;
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j != i; ++j){
			R(i,j) = Rng::gauss(0,1);
		}
	}
	return R;
}

RealMatrix createRandomMatrix(RealMatrix const& lambda,std::size_t Dimensions){
	RealMatrix R = blas::randomRotationMatrix(Dimensions);
	RealMatrix Atemp(Dimensions,Dimensions);
	RealMatrix A(Dimensions,Dimensions);
	axpy_prod(R,lambda,Atemp);
	axpy_prod(Atemp,trans(R),A);
	return A;
}

//special case first: the matrix is triangular
BOOST_AUTO_TEST_SUITE (LinAlg_RQ)

BOOST_AUTO_TEST_CASE( LinAlg_PivotingRQ_Triangular_FullRank ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 50;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable triangular matrix A 
		//with diagonal elements lambda
		RealMatrix lambda(Dimensions,Dimensions);
		lambda.clear();
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i,i) = 0.7*i-10*Rng::uni(1,2);
		}
		RealMatrix A = createRandomTriangularMatrix(lambda,Dimensions);
		
		//calculate RQ
		RealMatrix R(Dimensions,Dimensions);
		RealMatrix Q(Dimensions,Dimensions);
		blas::permutation_matrix P(Dimensions);
		std::size_t rank = pivotingRQ(A,R,Q,P);
		//test whether result is full rank
		BOOST_CHECK_EQUAL(rank,Dimensions);

		//create reconstruction of A
		RealMatrix ATest(Dimensions,Dimensions);
		axpy_prod(R,Q,ATest);
		//test reconstruction error after pivoting
		swap_rows(P,A);
		double errorA = norm_inf(A-ATest);
		BOOST_CHECK_SMALL(errorA,1.e-12);
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(ATest)));//test for nans
		
		//test determinant of R
		double logDetA = trace(log(abs(lambda)));
		double logDetR = trace(log(abs(R)));
		BOOST_CHECK_SMALL(std::abs(logDetA)-std::abs(logDetR),1.e-10);
		
		//test orthonormal property of Q
		RealMatrix testIdentity(Dimensions,Dimensions);
		axpy_prod(Q,trans(Q),testIdentity);
		RealMatrix testIdentity2(Dimensions,Dimensions);
		axpy_prod(trans(Q),Q,testIdentity2);
		double errorID1 = norm_inf(testIdentity-RealIdentityMatrix(Dimensions));
		double errorID2 = norm_inf(testIdentity2-RealIdentityMatrix(Dimensions));
		BOOST_CHECK(!(boost::math::isnan)(errorID1));
		BOOST_CHECK(!(boost::math::isnan)(errorID2));
		BOOST_CHECK_SMALL(errorID1,1.e-12);
		BOOST_CHECK_SMALL(errorID2,1.e-12);
	}
}

//second special case: the matrix is a rotation matrix
BOOST_AUTO_TEST_CASE( LinAlg_PivotingRQ_Rotation_FullRank ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 51;
	for(std::size_t test = 0; test != NumTests; ++test){
		RealMatrix A = blas::randomRotationMatrix(Dimensions);
		//calculate RQ
		RealMatrix R(Dimensions,Dimensions);
		RealMatrix Q(Dimensions,Dimensions);
		blas::permutation_matrix P(Dimensions);
		std::size_t rank = pivotingRQ(A,R,Q,P);
		//test whether result is full rank
		BOOST_CHECK_EQUAL(rank,Dimensions);

		//create reconstruction of A
		RealMatrix ATest(Dimensions,Dimensions);
		axpy_prod(R,Q,ATest);
		//test reconstruction error after pivoting
		swap_rows(P,A);
		double errorA = norm_inf(A-ATest);
		BOOST_CHECK_SMALL(errorA,1.e-12);
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(ATest)));//test for nans
		
		//test determinant of R
		double logDetA = 0;
		double logDetR = trace(log(abs(R)));
		BOOST_CHECK_SMALL(std::abs(logDetA)-std::abs(logDetR),1.e-10);
		
		//test orthonormal property of Q
		RealMatrix testIdentity(Dimensions,Dimensions);
		axpy_prod(Q,trans(Q),testIdentity);
		RealMatrix testIdentity2(Dimensions,Dimensions);
		axpy_prod(trans(Q),Q,testIdentity2);
		double errorID1 = norm_inf(testIdentity-RealIdentityMatrix(Dimensions));
		double errorID2 = norm_inf(testIdentity2-RealIdentityMatrix(Dimensions));
		BOOST_CHECK(!(boost::math::isnan)(errorID1));
		BOOST_CHECK(!(boost::math::isnan)(errorID2));
		BOOST_CHECK_SMALL(errorID1,1.e-12);
		BOOST_CHECK_SMALL(errorID2,1.e-12);
	}
}

//the input matrix is square and full rank
BOOST_AUTO_TEST_CASE( LinAlg_PivotingRQ_Square_FullRank ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 48;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable eigenvalue problem matrix A
		RealMatrix lambda(Dimensions,Dimensions);
		lambda.clear();
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i,i) = -10+i*0.7;
		}
		RealMatrix A = createRandomMatrix(lambda,Dimensions);
		//calculate RQ
		RealMatrix R(Dimensions,Dimensions);
		RealMatrix Q(Dimensions,Dimensions);
		blas::permutation_matrix P(Dimensions);
		std::size_t rank = pivotingRQ(A,R,Q,P);
		//test whether result is full rank
		BOOST_CHECK_EQUAL(rank,Dimensions);

		//create reconstruction of A
		RealMatrix ATest(Dimensions,Dimensions);
		axpy_prod(R,Q,ATest);
		//test reconstruction error after pivoting
		swap_rows(P,A);
		double errorA = norm_inf(A-ATest);

		BOOST_CHECK_SMALL(errorA,1.e-12);
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(ATest)));//test for nans
		
		//test determinant of R
		double logDetA = trace(log(abs(lambda)));
		double logDetR = trace(log(abs(R)));
		BOOST_CHECK_SMALL(std::abs(logDetA)-std::abs(logDetR),1.e-12);
		
		//test orthonormal property of Q
		RealMatrix testIdentity(Dimensions,Dimensions);
		axpy_prod(Q,trans(Q),testIdentity);
		RealMatrix testIdentity2(Dimensions,Dimensions);
		axpy_prod(trans(Q),Q,testIdentity2);
		double errorID1 = norm_inf(testIdentity-RealIdentityMatrix(Dimensions));
		double errorID2 = norm_inf(testIdentity2-RealIdentityMatrix(Dimensions));
		BOOST_CHECK(!(boost::math::isnan)(errorID1));
		BOOST_CHECK(!(boost::math::isnan)(errorID2));
		BOOST_CHECK_SMALL(errorID1,1.e-12);
		BOOST_CHECK_SMALL(errorID2,1.e-12);
	}
}

//Square and not full rank
BOOST_AUTO_TEST_CASE( LinAlg_PivotingRQ_Square ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 53;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable eigenvalue problem matrix A
		RealMatrix lambda(Dimensions,Dimensions);
		lambda.clear();
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i,i) = double(i)-10;
			lambda(3,3)=0;
			lambda(1,1)=0;
		}
		RealMatrix A = createRandomMatrix(lambda,Dimensions);
		
		//calculate RQ
		RealMatrix R(Dimensions,Dimensions);
		RealMatrix Q(Dimensions,Dimensions);
		blas::permutation_matrix P(Dimensions);
		std::size_t rank = pivotingRQ(A,R,Q,P);
		//test whether rank is correct
		BOOST_CHECK_EQUAL(rank,Dimensions-3);

		//create reconstruction of A
		RealMatrix ATest(Dimensions,Dimensions);
		axpy_prod(R,Q,ATest);
		//test reconstruction error after pivoting
		swap_rows(P,A);
		double errorA= norm_inf(A-ATest);
		BOOST_CHECK_SMALL(errorA,1.e-12);
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(ATest)));//test for nans

		//test orthonormal property of Q
		RealMatrix testIdentity(Dimensions,Dimensions);
		axpy_prod(Q,trans(Q),testIdentity);
		RealMatrix testIdentity2(Dimensions,Dimensions);
		axpy_prod(trans(Q),Q,testIdentity2);
		double errorID1 = norm_inf(testIdentity-RealIdentityMatrix(Dimensions));
		double errorID2 = norm_inf(testIdentity2-RealIdentityMatrix(Dimensions));
		BOOST_CHECK_SMALL(errorID1,1.e-12);
		BOOST_CHECK_SMALL(errorID2,1.e-12);
	}
}

//non square m > n but rank is n input is generated from precreated RQ
//with R being diagonal
BOOST_AUTO_TEST_CASE( LinAlg_PivotingRQ_DiagonalR_RankN ){
	std::size_t NumTests = 100;
	std::size_t M = 47;
	std::size_t N = 31;
	for(std::size_t test = 0; test != NumTests; ++test){
		//generate test input
		RealMatrix QTest = blas::randomRotationMatrix(N);
		RealMatrix RTest(M,N);
		RTest.clear();
		for(std::size_t i = 0; i != N; ++i){
			for(std::size_t j = 0; j != i; ++j){
				RTest(i,j) = Rng::gauss(0,1);
			}
			//make sure that the matrix has rank M. zero on diagonal is quite bad for this...
			while(std::abs(RTest(i,i)) < 0.1)
				RTest(i,i) = 0.7*i-10;
		}
		
		RealMatrix ATest(M,N);
		axpy_prod(RTest,QTest,ATest);
		
		//calculate RQ decomposition from the input
		RealMatrix R(M,N);
		RealMatrix Q(N,N);
		blas::permutation_matrix P(M);
		std::size_t rank = pivotingRQ(ATest,R,Q,P);
		
		//test whether rank is correct
		BOOST_CHECK_EQUAL(rank,N);

		//create reconstruction of A
		RealMatrix A(M,N);
		axpy_prod(R,Q,A);
		
		//test reconstruction error after pivoting
		swap_rows(P,ATest);
		double errorA = norm_inf(A-ATest);
		BOOST_CHECK_SMALL(errorA,1.e-12);
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(A)));//test for nans

		//test orthonormal property of Q
		RealMatrix testIdentity(N,N);
		axpy_prod(Q,trans(Q),testIdentity);
		RealMatrix testIdentity2(N,N);
		axpy_prod(trans(Q),Q,testIdentity2);
		double errorID1 = norm_inf(testIdentity-RealIdentityMatrix(N));
		double errorID2 = norm_inf(testIdentity2-RealIdentityMatrix(N));
		BOOST_CHECK_SMALL(errorID1,1.e-12);
		BOOST_CHECK_SMALL(errorID2,1.e-12);
	}
}

//non square m < n but rank is m input is generated from precreated RQ
//with R being diagonal
BOOST_AUTO_TEST_CASE( LinAlg_PivotingRQ_DiagonalR_RankM ){
	std::size_t NumTests = 100;
	std::size_t M = 31;
	std::size_t N = 47;
	for(std::size_t test = 0; test != NumTests; ++test){
		//generate test input
		RealMatrix QTest = blas::randomRotationMatrix(N);
		RealMatrix RTest(M,N);
		RTest.clear();
		for(std::size_t i = 0; i != M; ++i){
			for(std::size_t j = 0; j != i; ++j){
				RTest(i,j) = Rng::gauss(0,1);
			}
			RTest(i,i) = 0.7*i-10;
		}
		RealMatrix ATest(M,N);
		axpy_prod(RTest,QTest,ATest);
		
		//calculate RQ decomposition from the input
		RealMatrix R(M,N);
		RealMatrix Q(N,N);
		blas::permutation_matrix P(M);
		std::size_t rank = pivotingRQ(ATest,R,Q,P);
		
		//test whether rank is correct
		BOOST_CHECK_EQUAL(rank,M);

		//create reconstruction of A
		RealMatrix A(M,N);
		axpy_prod(R,Q,A);
		
		//test reconstruction error after pivoting
		swap_rows(P,ATest);
		double errorA = norm_inf(A-ATest);
		BOOST_CHECK_SMALL(errorA,1.e-12);
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(A)));//test for nans

		//test orthonormal property of Q
		RealMatrix testIdentity(N,N);
		axpy_prod(Q,trans(Q),testIdentity);
		RealMatrix testIdentity2(N,N);
		axpy_prod(trans(Q),Q,testIdentity2);
		double errorID1 = norm_inf(testIdentity-RealIdentityMatrix(N));
		double errorID2 = norm_inf(testIdentity2-RealIdentityMatrix(N));
		BOOST_CHECK_SMALL(errorID1,1.e-12);
		BOOST_CHECK_SMALL(errorID2,1.e-12);
	}
}


//non square m > n > rank. input is generated from precreated RQ
//with R being diagonal
BOOST_AUTO_TEST_CASE( LinAlg_PivotingRQ_DiagonalR_RankLowerN ){
	std::size_t NumTests = 100;
	std::size_t M = 30;
	std::size_t N = 51;
	std::size_t Rank = 20;
	for(std::size_t test = 0; test != NumTests; ++test){
		//generate test input
		RealMatrix QTest = blas::randomRotationMatrix(N);
		RealMatrix RTest(M,N);
		RTest.clear();
		for(std::size_t i = 0; i != Rank; ++i){
			for(std::size_t j = 0; j != i; ++j){
				RTest(i,j) = Rng::gauss(0,1);
			}
			if(i< Rank)
				RTest(i,i) = 0.7*i-10;
		}
		
		RealMatrix ATest(M,N);
		axpy_prod(RTest,QTest,ATest);
		
		//calculate RQ decomposition from the input
		RealMatrix R(M,N);
		RealMatrix Q(N,N);
		blas::permutation_matrix P(M);
		std::size_t rank = pivotingRQ(ATest,R,Q,P);
		
		//test whether rank is correct
		BOOST_CHECK_EQUAL(rank,Rank);

		//create reconstruction of A
		RealMatrix A(M,N);
		axpy_prod(R,Q,A);
		
		//test reconstruction error after pivoting
		swap_rows(P,ATest);
		double errorA = norm_inf(A-ATest);
		BOOST_CHECK_SMALL(errorA,1.e-12);
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(A)));//test for nans

		//test orthonormal property of Q
		RealMatrix testIdentity(N,N);
		axpy_prod(Q,trans(Q),testIdentity);
		RealMatrix testIdentity2(N,N);
		axpy_prod(trans(Q),Q,testIdentity2);
		double errorID1 = norm_inf(testIdentity-RealIdentityMatrix(N));
		double errorID2 = norm_inf(testIdentity2-RealIdentityMatrix(N));
		BOOST_CHECK_SMALL(errorID1,1.e-12);
		BOOST_CHECK_SMALL(errorID2,1.e-12);
	}
}

//non square n > m > rank. input is generated from precreated RQ
//with R being diagonal
BOOST_AUTO_TEST_CASE( LinAlg_PivotingRQ_DiagonalR_RankLowerM ){
	std::size_t NumTests = 100;
	std::size_t M = 47;
	std::size_t N = 31;
	std::size_t Rank = 19;
	for(std::size_t test = 0; test != NumTests; ++test){
		//generate test input
		RealMatrix QTest = blas::randomRotationMatrix(N);
		RealMatrix RTest(M,N);
		RTest.clear();
		for(std::size_t i = 0; i != Rank; ++i){
			for(std::size_t j = 0; j != i; ++j){
				RTest(i,j) = Rng::gauss(0,1);
			}
			if(i< Rank)
				RTest(i,i) = 0.7*i-10;
		}
		
		RealMatrix ATest(M,N);
		axpy_prod(RTest,QTest,ATest);
		
		//calculate RQ decomposition from the input
		RealMatrix R(M,N);
		RealMatrix Q(N,N);
		blas::permutation_matrix P(M);
		std::size_t rank = pivotingRQ(ATest,R,Q,P);
		
		//test whether rank is correct
		BOOST_CHECK_EQUAL(rank,Rank);

		//create reconstruction of A
		RealMatrix A(M,N);
		axpy_prod(R,Q,A);
		
		//test reconstruction error after pivoting
		swap_rows(P,ATest);
		double errorA = norm_inf(A-ATest);
		BOOST_CHECK_SMALL(errorA,1.e-12);
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(A)));//test for nans

		//test orthonormal property of Q
		RealMatrix testIdentity(N,N);
		axpy_prod(Q,trans(Q),testIdentity);
		RealMatrix testIdentity2(N,N);
		axpy_prod(trans(Q),Q,testIdentity2);
		double errorID1 = norm_inf(testIdentity-RealIdentityMatrix(N));
		double errorID2 = norm_inf(testIdentity2-RealIdentityMatrix(N));
		BOOST_CHECK_SMALL(errorID1,1.e-12);
		BOOST_CHECK_SMALL(errorID2,1.e-12);
	}
}

BOOST_AUTO_TEST_SUITE_END()
