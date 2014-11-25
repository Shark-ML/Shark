#define BOOST_TEST_MODULE LinAlg_Solve
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/solveSystem.h>
#include <shark/LinAlg/rotations.h>
using namespace shark;

RealMatrix createRandomInvertibleMatrix(std::size_t Dimensions,double lambdaMin, double lambdaMax){
	RealMatrix R = blas::randomRotationMatrix(Dimensions);
	RealMatrix Atemp = trans(R);
	for(std::size_t i = 0; i != Dimensions; ++i){
		double lambda = 0;
		while(std::abs(lambda)<1.e-5)//prevent ill-conditioning
			lambda = Rng::uni(lambdaMin,lambdaMax);
		row(Atemp,i)*=lambda;
	}
	RealMatrix A(Dimensions,Dimensions);
	axpy_prod(R,Atemp,A);
	return A;
}
template<class Vec,class Mat>
RealMatrix createSymmetricMatrix(Vec const& lambda,Mat const& rotation){
	RealMatrix intermediate=rotation;
	RealMatrix result(rotation.size1(),rotation.size2());
	result.clear();
	for(std::size_t i = 0; i != intermediate.size1(); ++i){
		row(intermediate,i) *= lambda(i);
	}
	axpy_prod(trans(rotation),intermediate,result);
	return result;
}

//simple test which checks for all argument combinations whether they are correctly translated
BOOST_AUTO_TEST_SUITE (LinAlg_solve)

BOOST_AUTO_TEST_CASE( LinAlg_Solve_TriangularInPlace_Calls_Matrix ){
	
	//i actually had to come up with a simple hand calculated example here
	//this function is so basic that we can't use any other to prove it
	//(as it is used/might be used by these functions at some point...)
	RealMatrix A(2,2);
	A(0,0) = 3;
	A(0,1) = 0;
	A(1,0) = 2;
	A(1,1) = 4;
	RealMatrix Atrans = trans(A);
	
	RealMatrix AInv(2,2);
	AInv(0,0) = 1.0/3;
	AInv(0,1) = 0;
	AInv(1,0) = -1.0/6;
	AInv(1,1) = 0.25;
	
	RealMatrix unitAInv(2,2);
	unitAInv(0,0) = 1;
	unitAInv(0,1) = 0;
	unitAInv(1,0) = -2;
	unitAInv(1,1) = 1;
	
	RealMatrix input(2,10);
	for(std::size_t j = 0; j != 10; ++j){
		input(0,j) = Rng::uni(1,10);
		input(1,j) = Rng::uni(1,10);
	}
	RealMatrix transInput=trans(input);
	
	//we print out letters since ATLAS just crashes the program when an error occurs. 
	//very good if you need a backtrace...not!
	//now we test all combinations of systems
	//for non-blas::Unit-matrices
	std::cout<<"triangular matrix"<<std::endl;
	std::cout<<"a"<<std::endl;
	{
		RealMatrix testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::lower>(A,testResult);
		RealMatrix result = prod(AInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"b"<<std::endl;
	{
		RealMatrix testResult = subrange(input,0,1,0,2);
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::lower>(A,testResult);
		RealMatrix result = prod(subrange(input,0,1,0,2),AInv);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"c1"<<std::endl;
	{
		RealMatrix testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::upper>(Atrans,testResult);
		RealMatrix result = prod(trans(AInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"c2"<<std::endl;
	{
		RealMatrix testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::upper>(trans(A),testResult);
		RealMatrix result = prod(trans(AInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"d1"<<std::endl;
	{
		RealMatrix testResult = subrange(input,0,1,0,2);
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::upper>(Atrans,testResult);
		RealMatrix result = prod(subrange(input,0,1,0,2),trans(AInv));
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"d2"<<std::endl;
	{
		RealMatrix testResult = subrange(input,0,1,0,2);
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::upper>(trans(A),testResult);
		RealMatrix result = prod(subrange(input,0,1,0,2),trans(AInv));
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"e"<<std::endl;//(check for column major blas::lower arguments)
	{
		blas::matrix<double,blas::column_major> AcolMaj = A;
		blas::matrix<double,blas::column_major> testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::lower>(AcolMaj,testResult);
		RealMatrix result = prod(AInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"f"<<std::endl;//(check for row major blas::upper arguments)
	{
		RealMatrix Atrans = trans(A);
		RealMatrix testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::upper>(Atrans,testResult);
		RealMatrix result = prod(trans(AInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	
	//now we test all combinations of transpositions
	//for blas::Unit-matrices
	std::cout<<"a"<<std::endl;
	{
		RealMatrix testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::unit_lower>(A,testResult);
		RealMatrix result = prod(unitAInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"b"<<std::endl;
	{
		RealMatrix testResult = subrange(input,0,1,0,2);
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::unit_lower>(A,testResult);
		RealMatrix result = prod(subrange(input,0,1,0,2),unitAInv);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"c"<<std::endl;
	{
		RealMatrix testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::unit_upper>(trans(A),testResult);
		RealMatrix result = prod(trans(unitAInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"d"<<std::endl;
	{
		RealMatrix testResult = subrange(input,0,1,0,2);
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::unit_upper>(trans(A),testResult);
		RealMatrix result = prod(subrange(input,0,1,0,2),trans(unitAInv));
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"e"<<std::endl;//(check for column major blas::lower arguments)
	{
		blas::matrix<double,blas::column_major> AcolMaj = A;
		blas::matrix<double,blas::column_major> testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::unit_lower>(AcolMaj,testResult);
		RealMatrix result = prod(unitAInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"f"<<std::endl;//(check for row major blas::upper arguments)
	{
		RealMatrix Atrans = trans(A);
		RealMatrix testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::unit_upper>(Atrans,testResult);
		RealMatrix result = prod(trans(unitAInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
}

//simple test which checks for all argument combinations whether they are correctly translated
BOOST_AUTO_TEST_CASE( LinAlg_Solve_TriangularInPlace_Calls_Vector ){
	RealMatrix A(2,2);
	A(0,0) = 3;
	A(0,1) = 0;
	A(1,0) = 2;
	A(1,1) = 4;
	
	RealMatrix AInv(2,2);
	AInv(0,0) = 1.0/3;
	AInv(0,1) = 0;
	AInv(1,0) = -1.0/6;
	AInv(1,1) = 0.25;
	
	RealMatrix unitAInv(2,2);
	unitAInv(0,0) = 1;
	unitAInv(0,1) = 0;
	unitAInv(1,0) = -2;
	unitAInv(1,1) = 1;
	
	RealVector input(2);
	input(0) = Rng::uni(1,10);
	input(1) = Rng::uni(1,10);
	
	//we print out letters since ATLAS just crashes the program when an error occurs. 
	//very good if you need a backtrace...not!
	//now we test all combinations of transpositions
	//for non-blas::Unit-matrices
	std::cout<<"triangular vector"<<std::endl;
	std::cout<<"a"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::lower>(A,testResult);
		RealVector result = prod(AInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"b"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::upper>(trans(A),testResult);
		RealVector result = prod(trans(AInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"c"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::lower>(A,testResult);
		RealVector result = prod(input,AInv);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"d"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::upper>(trans(A),testResult);
		RealVector result = prod(input,trans(AInv));
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"e"<<std::endl;//(check for column major blas::lower arguments)
	{
		blas::matrix<double,blas::column_major> AcolMaj = A;
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::lower>(AcolMaj,testResult);
		RealVector result = prod(AInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"f"<<std::endl;//(check for row major blas::upper arguments)
	{
		RealMatrix Atrans = trans(A);
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::upper>(Atrans,testResult);
		RealVector result = prod(trans(AInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	
	//now we test all combinations of transpositions
	//for blas::Unit-matrices
	std::cout<<"a"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::unit_lower>(A,testResult);
		RealVector result = prod(unitAInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"b"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::unit_upper>(trans(A),testResult);
		RealVector result = prod(trans(unitAInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"c"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::unit_lower>(A,testResult);
		RealVector result = prod(input,unitAInv);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"d"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::unit_upper>(trans(A),testResult);
		RealVector result = prod(input,trans(unitAInv));
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"e"<<std::endl;//(check for column major blas::lower arguments)
	{
		blas::matrix<double,blas::column_major> AcolMaj = A;
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::unit_lower>(AcolMaj,testResult);
		RealVector result = prod(unitAInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"f"<<std::endl;//(check for row major blas::upper arguments)
	{
		RealMatrix Atrans = trans(A);
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::unit_upper>(Atrans,testResult);
		RealVector result = prod(trans(unitAInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
}

//for the remaining functions, we can use random systems and check, whether they are okay
BOOST_AUTO_TEST_CASE( LinAlg_Solve_Vector ){
	unsigned NumTests = 100;
	std::size_t Dimensions = 50;
	std::cout<<"blas::solveSystem vector"<<std::endl;
	for(unsigned testi = 0; testi != NumTests; ++testi){
		RealMatrix A = createRandomInvertibleMatrix(Dimensions,-2,2);
		RealVector b(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			b(i) = Rng::gauss(0,1);
		}
		
		RealVector x;
		blas::solveSystem(A,x,b);
		
		//calculate backwards
		RealVector test = prod(A,x);
		
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-11);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_Solve_Symmetric_Vector ){
	unsigned NumTests = 100;
	std::size_t Dimensions = 50;
	
	std::cout<<"blas::solve Symmetric vector"<<std::endl;
	for(unsigned testi = 0; testi != NumTests; ++testi){
		RealMatrix A = createRandomInvertibleMatrix(Dimensions,0.1,2);
		RealVector b(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			b(i) = Rng::gauss(0,1);
		}
		
		RealVector x;
		
		//first test AX=B
		blas::solveSymmSystem<blas::SolveAXB>(A,x,b);
		RealVector test = prod(A,x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
		
		//first test trans(A)X=B
		blas::solveSymmSystem<blas::SolveAXB>(trans(A),x,b);
		test = prod(trans(A),x);
		error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
		
		//now test XA=B
		blas::solveSymmSystem<blas::SolveXAB>(A,x,b);
		test = prod(x,A);
		error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
		
		//now test Xtrans(A)=B
		blas::solveSymmSystem<blas::SolveXAB>(trans(A),x,b);
		test = prod(x,trans(A));
		error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_Solve_Symmetric_Approximated_Vector ){
	unsigned NumTests = 100;
	std::size_t Dimensions = 50;
	
	std::cout<<"approximately blas::solve Symmetric vector"<<std::endl;
	for(unsigned testi = 0; testi != NumTests; ++testi){
		RealMatrix A = createRandomInvertibleMatrix(Dimensions,0.1,2);
		RealVector b(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			b(i) = Rng::gauss(0,1);
		}
		
		RealVector x;
		
		//first test AX=B
		approxSolveSymmSystem(A,x,b,1.e-13);
		RealVector test = prod(A,x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_Solve_Matrix ){
	unsigned NumTests = 100;
	std::size_t Dimensions = 50;
	std::size_t numRhs = 21;
	std::cout<<"blas::solve matrix"<<std::endl;
	for(unsigned testi = 0; testi != NumTests; ++testi){
		RealMatrix A = createRandomInvertibleMatrix(Dimensions,-2,2);
		RealMatrix B(Dimensions,numRhs);
		for(std::size_t i = 0; i != Dimensions; ++i){
			for(std::size_t j = 0; j != numRhs; ++j){
				B(i,j) = Rng::gauss(0,1);
			}
		}
		
		RealMatrix X;
		blas::solveSystem(A,X,B);
		
		//calculate backwards
		RealMatrix test(Dimensions,numRhs);
		axpy_prod(A,X,test);
		
		double error = norm_inf(test-B);
		BOOST_CHECK_SMALL(error,1.e-10);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_Solve_Symmetric_Matrix ){
	unsigned NumTests = 100;
	std::size_t Dimensions = 50;
	std::size_t numRhs = 21;
	std::cout<<"blas::solve symmetric matrix"<<std::endl;
	for(unsigned testi = 0; testi != NumTests; ++testi){
		RealMatrix A = createRandomInvertibleMatrix(Dimensions,0.1,2);
		RealMatrix ARight = createRandomInvertibleMatrix(numRhs,0.1,2);
		RealMatrix B(Dimensions,numRhs);
		for(std::size_t i = 0; i != Dimensions; ++i){
			for(std::size_t j = 0; j != numRhs; ++j){
				B(i,j) = Rng::gauss(0,1);
			}
		}
		
		RealMatrix X;
		blas::solveSymmSystem<blas::SolveAXB>(A,X,B);
		RealMatrix test = prod(A,X);
		double error = norm_inf(test-B);
		BOOST_CHECK_SMALL(error,1.e-12);
		
		//first test trans(A)X=B
		blas::solveSymmSystem<blas::SolveAXB>(trans(A),X,B);
		test = prod(trans(A),X);
		error = norm_inf(test-B);
		BOOST_CHECK_SMALL(error,1.e-12);
		
		//now test XA=B
		blas::solveSymmSystem<blas::SolveXAB>(ARight,X,B);
		test = prod(X,ARight);
		error = norm_inf(test-B);
		BOOST_CHECK_SMALL(error,1.e-12);
		
		//now test Xtrans(A)=B
		blas::solveSymmSystem<blas::SolveXAB>(trans(ARight),X,B);
		test = prod(X,trans(ARight));
		error = norm_inf(test-B);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

//this test tests, whether the semi-definite system solver can handle full rank matrices and arbitrary right-hand vectors
BOOST_AUTO_TEST_CASE( LinAlg_solveSymmSemiDefiniteSystemInPlace_fullRank_Vector){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 50;
	std::cout<<"blas::solve symmetric semi-definite matrix full rank vector LHS"<<std::endl;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable symmetric matrix A as well as its inverse
		RealVector lambda(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i) = 0.7*i+1;
		}
		//generate random left and right matrices as well as right hand side
		RealMatrix R = blas::randomRotationMatrix(Dimensions);
		RealMatrix A = createSymmetricMatrix(lambda,R);
		RealVector b(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			b(i) = Rng::gauss(0,1);
		}
		
		
		//System Ax=b
		{
			RealVector x=b;
			blas::solveSymmSemiDefiniteSystemInPlace<blas::SolveAXB>(A,x);
			RealVector test = prod(A,x);
			double error = norm_inf(test-b);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
		
		//System x^TA=b^T
		{
			RealVector x=b;
			blas::solveSymmSemiDefiniteSystemInPlace<blas::SolveXAB>(A,x);
			RealVector test = prod(x,A);
			double error = norm_inf(test-b);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_solveSymmSemiDefiniteSystemInPlace_RankK_Vector ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 50;
	std::size_t Rank = 10;
	std::cout<<"blas::solve symmetric semi-definite matrix general LHS"<<std::endl;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable symmetric matrix A as well as its inverse
		RealVector lambda(Dimensions,0.0);
		for(std::size_t i = 0; i != Rank; ++i){
			lambda(i) = 0.1*i+1;
		}
		RealVector lambdaInv(Dimensions,0.0);
		for(std::size_t i = 0; i != Rank; ++i){
			lambdaInv(i) = 1.0/lambda(i);
		}

		//generate random left and right matrices as well as right hand side
		RealMatrix R = blas::randomRotationMatrix(Dimensions);
		RealMatrix A = createSymmetricMatrix(lambda,R);
		RealMatrix AInv = createSymmetricMatrix(lambdaInv,R);
		RealVector b(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			b(i) = Rng::gauss(0,1);
		}	
		
		//System AX=B
		{
			RealVector x=b;
			blas::solveSymmSemiDefiniteSystemInPlace<blas::SolveAXB>(A,x);
			RealVector test = prod(AInv,b);
			double error = norm_inf(test-x);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
		
		//System XA=B
		{
			RealVector x=b;
			blas::solveSymmSemiDefiniteSystemInPlace<blas::SolveXAB>(A,x);
			RealVector test = prod(b,AInv);
			double error = norm_inf(test-x);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
	}
}


//this test tests, whether the semi-definite system solver can handle full rank matrices and generate proper inverses
BOOST_AUTO_TEST_CASE( LinAlg_solveSymmSemiDefiniteSystemInPlace_fullRank_Matrix_Inverse ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 50;
	std::cout<<"blas::solve symmetric semi-definite matrix full rank inverse"<<std::endl;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable symmetric matrix A as well as its inverse
		RealVector lambda(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i) = 0.7*i+1;
		}
		RealMatrix R = blas::randomRotationMatrix(Dimensions);
		RealMatrix A = createSymmetricMatrix(lambda,R);
		
		
		//System AX=I
		{
			RealMatrix AInv(Dimensions,Dimensions,0.0);
			for(std::size_t i = 0; i != Dimensions; ++i){
				AInv(i,i) = 1.0;
			}
			blas::solveSymmSemiDefiniteSystemInPlace<blas::SolveAXB>(A,AInv);
			
			//should be left and right inverse
			RealMatrix resultLeft(Dimensions,Dimensions);
			RealMatrix resultRight(Dimensions,Dimensions);
			axpy_prod(AInv,A,resultLeft);
			axpy_prod(A,AInv,resultRight);
			
			double errorSame = norm_inf(resultLeft-resultRight);
			double errorLeft = norm_inf(resultLeft - RealIdentityMatrix(Dimensions));
			double errorRight = norm_inf(resultRight - RealIdentityMatrix(Dimensions));
			BOOST_CHECK_SMALL(errorSame,1.e-13);
			BOOST_CHECK_SMALL(errorLeft,1.e-10);
			BOOST_CHECK_SMALL(errorRight,1.e-10);
		}
		//System XA=I
		{
			RealMatrix AInv(Dimensions,Dimensions,0.0);
			for(std::size_t i = 0; i != Dimensions; ++i){
				AInv(i,i) = 1.0;
			}
			blas::solveSymmSemiDefiniteSystemInPlace<blas::SolveXAB>(A,AInv);
			
			//should be left and right inverse
			RealMatrix resultLeft(Dimensions,Dimensions);
			RealMatrix resultRight(Dimensions,Dimensions);
			axpy_prod(AInv,A,resultLeft);
			axpy_prod(A,AInv,resultRight);
			
			double errorSame = norm_inf(resultLeft-resultRight);
			double errorLeft = norm_inf(resultLeft - RealIdentityMatrix(Dimensions));
			double errorRight = norm_inf(resultRight - RealIdentityMatrix(Dimensions));
			BOOST_CHECK_SMALL(errorSame,1.e-13);
			BOOST_CHECK_SMALL(errorLeft,1.e-10);
			BOOST_CHECK_SMALL(errorRight,1.e-10);
		}
	}
}

//this test tests, whether the semi-definite system solver can handle full rank matrices and arbitrary right-hand sides
BOOST_AUTO_TEST_CASE( LinAlg_solveSymmSemiDefiniteSystemInPlace_fullRank_Matrix ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 50;
	std::size_t NumRhs = 21;
	std::cout<<"blas::solve symmetric semi-definite matrix full rank general LHS"<<std::endl;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable symmetric matrix A as well as its inverse
		RealVector lambda(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i) = 0.7*i+1;
		}
		RealVector lambdaRight(NumRhs);
		for(std::size_t i = 0; i != NumRhs; ++i){
			lambdaRight(i) = 0.7*i+1;
		}
		//generate random left and right matrices as well as right hand side
		RealMatrix R = blas::randomRotationMatrix(Dimensions);
		RealMatrix RRight = blas::randomRotationMatrix(NumRhs);
		RealMatrix A = createSymmetricMatrix(lambda,R);
		RealMatrix ARight = createSymmetricMatrix(subrange(lambda,0,NumRhs),RRight);
		RealMatrix B(Dimensions,NumRhs);
		for(std::size_t i = 0; i != Dimensions; ++i){
			for(std::size_t j = 0; j != NumRhs; ++j){
				B(i,j) = Rng::gauss(0,1);
			}
		}
		
		
		//System AX=B
		{
			RealMatrix X=B;
			blas::solveSymmSemiDefiniteSystemInPlace<blas::SolveAXB>(A,X);
			RealMatrix test = prod(A,X);
			double error = norm_inf(test-B);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
		
		//System XA=B
		{
			RealMatrix X=B;
			blas::solveSymmSemiDefiniteSystemInPlace<blas::SolveXAB>(ARight,X);
			RealMatrix test = prod(X,ARight);
			double error = norm_inf(test-B);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_solveSymmSemiDefiniteSystemInPlace_RankK_Matrix ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 50;
	std::size_t Rank = 10;
	std::size_t NumRhs = 21;
	std::cout<<"blas::solve symmetric semi-definite matrix general LHS"<<std::endl;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable symmetric matrix A as well as its inverse
		RealVector lambda(Dimensions,0.0);
		for(std::size_t i = 0; i != Rank; ++i){
			lambda(i) = 0.1*i+1;
		}
		RealVector lambdaInv(Dimensions,0.0);
		for(std::size_t i = 0; i != Rank; ++i){
			lambdaInv(i) = 1.0/lambda(i);
		}

		//generate random left and right matrices as well as right hand side
		RealMatrix R = blas::randomRotationMatrix(Dimensions);
		RealMatrix RRight = blas::randomRotationMatrix(NumRhs);
		RealMatrix A = createSymmetricMatrix(lambda,R);
		RealMatrix AInv = createSymmetricMatrix(lambdaInv,R);
		RealMatrix ARight = createSymmetricMatrix(subrange(lambda,0,NumRhs),RRight);
		RealMatrix ARightInv = createSymmetricMatrix(subrange(lambdaInv,0,NumRhs),RRight);
		RealMatrix B(Dimensions,NumRhs);
		for(std::size_t i = 0; i != Dimensions; ++i){
			for(std::size_t j = 0; j != NumRhs; ++j){
				B(i,j) = Rng::gauss(0,1);
			}
		}		
		
		//System AX=B
		{
			RealMatrix X=B;
			blas::solveSymmSemiDefiniteSystemInPlace<blas::SolveAXB>(A,X);
			RealMatrix test = prod(AInv,B);
			double error = norm_inf(test-X);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
		
		//System XA=B
		{
			RealMatrix X=B;
			blas::solveSymmSemiDefiniteSystemInPlace<blas::SolveXAB>(ARight,X);
			RealMatrix test = prod(B,ARightInv);
			double error = norm_inf(test-X);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
	}
}


//first test that we can actually invert invertible matrices
BOOST_AUTO_TEST_CASE( LinAlg_generalSolveSystemInPlace_Invertible ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 50;
	std::cout<<"blas::solve general matrix full rank vector LHS"<<std::endl;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable symmetric matrix A as well as its inverse
		RealVector lambda(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i) = 0.7*i+1;
		}
		//generate random left and right matrices as well as right hand side
		RealMatrix R = blas::randomRotationMatrix(Dimensions);
		RealMatrix A = createSymmetricMatrix(lambda,R);
		RealVector b(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			b(i) = Rng::gauss(0,1);
		}
		
		
		//System Ax=b
		{
			RealVector x=b;
			blas::generalSolveSystemInPlace<blas::SolveAXB>(A,x);
			RealVector test = prod(A,x);
			double error = norm_inf(test-b);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
		
		//System x^TA=b^T
		{
			RealVector x=b;
			blas::generalSolveSystemInPlace<blas::SolveXAB>(A,x);
			RealVector test = prod(x,A);
			double error = norm_inf(test-b);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
	}
}

//now test square symmetric matrices without full rank
BOOST_AUTO_TEST_CASE( LinAlg_generalSolveSystemInPlace_RankK ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 50;
	std::size_t Rank = 20;
	std::cout<<"blas::solve general semi-definite matrix general LHS"<<std::endl;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable symmetric matrix A as well as its inverse
		RealVector lambda(Dimensions,0.0);
		for(std::size_t i = 0; i != Rank; ++i){
			lambda(i) = 0.1*i+1;
		}
		RealVector lambdaInv(Dimensions,0.0);
		for(std::size_t i = 0; i != Rank; ++i){
			lambdaInv(i) = 1.0/lambda(i);
		}

		//generate random left and right matrices as well as right hand side
		RealMatrix R = blas::randomRotationMatrix(Dimensions);
		RealMatrix A = createSymmetricMatrix(lambda,R);
		RealMatrix AInv = createSymmetricMatrix(lambdaInv,R);
		RealVector b(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			b(i) = Rng::gauss(0,1);
		}
		
		
		//System Ax=b
		{
			RealVector x=b;
			blas::generalSolveSystemInPlace<blas::SolveAXB>(A,x);
			RealVector test = prod(AInv,b);
			double error = norm_inf(test-x);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
		
		//System x^TA=b^T
		{
			RealVector x=b;
			blas::generalSolveSystemInPlace<blas::SolveXAB>(A,x);
			RealVector test = prod(b,AInv);
			double error = norm_inf(test-x);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
	}
}

//rectangular mxn matrix with rank k, m < n
BOOST_AUTO_TEST_CASE( LinAlg_generalSolveSystemInPlace_Rectangular1_RankK ){
	std::size_t NumTests = 100;
	std::size_t M = 31;
	std::size_t N = 50;
	std::size_t Rank = 20;
	std::cout<<"blas::solve rectangular matrix m<n general LHS"<<std::endl;
	for(std::size_t test = 0; test != NumTests; ++test){
		//generate test input
		RealMatrix QN = rows(blas::randomRotationMatrix(N),0,M);
		RealMatrix QM = blas::randomRotationMatrix(M);
		RealVector lambda(M,0.0);
		for(std::size_t i = 0; i != Rank; ++i){
			lambda(i) = 0.1*i+1;
		}
		
		RealVector lambdaInv(M,0.0);
		for(std::size_t i = 0; i != Rank; ++i){
			lambdaInv(i) = 1.0/lambda(i);
		}
		
		//create matrix and "inverse".
		RealMatrix A(M,N);
		{
			RealMatrix temp = QN;
			for(std::size_t i = 0; i != M; ++i){
				row(temp,i) *= lambda(i);
			}
			axpy_prod(QM,temp,A);
		}
		RealMatrix AInv(N,M);
		{
			RealMatrix temp = trans(QN);
			for(std::size_t i = 0; i != M; ++i){
				column(temp,i) *= lambdaInv(i);
			}
			axpy_prod(temp,trans(QM),AInv);
		}
		
		RealVector bM(M);
		for(std::size_t i = 0; i != M; ++i){
			bM(i) = Rng::gauss(0,1);
		}
		
		RealVector bN(N);
		for(std::size_t i = 0; i != N; ++i){
			bN(i) = Rng::gauss(0,1);
		}
		
		//System Ax=b
		{
			RealVector x=bM;
			blas::generalSolveSystemInPlace<blas::SolveAXB>(A,x);
			RealVector test = prod(AInv,bM);
			double error = norm_inf(test-x);
			BOOST_CHECK_SMALL(error,1.e-11);
		}
		
		//System x^TA=b^T
		{
			RealVector x=bN;
			blas::generalSolveSystemInPlace<blas::SolveXAB>(A,x);
			RealVector test = prod(bN,AInv);
			double error = norm_inf(test-x);
			BOOST_CHECK_SMALL(error,1.e-11);
		}
	}
}

//rectangular mxn matrix with rank k, m < n
BOOST_AUTO_TEST_CASE( LinAlg_generalSolveSystemInPlace_Rectangular2_RankK ){
	std::size_t NumTests = 100;
	std::size_t M = 50;
	std::size_t N = 31;
	std::size_t Rank = 20;
	std::cout<<"blas::solve rectangular matrix m>n general LHS"<<std::endl;
	for(std::size_t test = 0; test != NumTests; ++test){
		//generate test input
		RealMatrix QN = blas::randomRotationMatrix(N);
		RealMatrix QM = columns(blas::randomRotationMatrix(M),0,N);
		RealVector lambda(N,0.0);
		for(std::size_t i = 0; i != Rank; ++i){
			lambda(i) = 0.1*i+1;
		}
		
		RealVector lambdaInv(N,0.0);
		for(std::size_t i = 0; i != Rank; ++i){
			lambdaInv(i) = 1.0/lambda(i);
		}
		
		//create matrix and "inverse".
		RealMatrix A(M,N);
		{
			RealMatrix temp = QM;
			for(std::size_t i = 0; i != N; ++i){
				column(temp,i) *= lambda(i);
			}
			axpy_prod(temp,QN,A);
		}
		RealMatrix AInv(N,M);
		{
			RealMatrix temp = trans(QM);
			for(std::size_t i = 0; i != N; ++i){
				row(temp,i) *= lambdaInv(i);
			}
			axpy_prod(trans(QN),temp,AInv);
		}
		
		RealVector bM(M);
		for(std::size_t i = 0; i != M; ++i){
			bM(i) = Rng::gauss(0,1);
		}
		
		RealVector bN(N);
		for(std::size_t i = 0; i != N; ++i){
			bN(i) = Rng::gauss(0,1);
		}
		
		//System Ax=b
		{
			RealVector x=bM;
			blas::generalSolveSystemInPlace<blas::SolveAXB>(A,x);
			RealVector test = prod(AInv,bM);
			double error = norm_inf(test-x);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
		
		//System x^TA=b^T
		{
			RealVector x=bN;
			blas::generalSolveSystemInPlace<blas::SolveXAB>(A,x);
			RealVector test = prod(bN,AInv);
			double error = norm_inf(test-x);
			BOOST_CHECK_SMALL(error,1.e-12);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
