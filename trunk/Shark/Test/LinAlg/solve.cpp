#define BOOST_TEST_MODULE LinAlg_Solve
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/solveSystem.h>
#include <shark/LinAlg/rotations.h>
using namespace shark;

//simple test which checks for all argument combinations whether they are correctly translated
BOOST_AUTO_TEST_CASE( LinAlg_Solve_TriangularInPlace_Calls_Matrix ){
	
	//i actually had to come up with a simple hand calculated example here
	//this function is so basic that we can't use any other to prove it
	//(as it is used/might be used by these functions at some point...)
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
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::Lower>(A,testResult);
		RealMatrix result = prod(AInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"b"<<std::endl;
	{
		RealMatrix testResult = subrange(input,0,1,0,2);
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::Lower>(A,testResult);
		RealMatrix result = prod(subrange(input,0,1,0,2),AInv);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"c"<<std::endl;
	{
		RealMatrix testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::Upper>(trans(A),testResult);
		RealMatrix result = prod(trans(AInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"d"<<std::endl;
	{
		RealMatrix testResult = subrange(input,0,1,0,2);
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::Upper>(trans(A),testResult);
		RealMatrix result = prod(subrange(input,0,1,0,2),trans(AInv));
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"e"<<std::endl;//(check for column major blas::Lower arguments)
	{
		blas::matrix<double,blas::column_major> AcolMaj = A;
		blas::matrix<double,blas::column_major> testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::Lower>(AcolMaj,testResult);
		RealMatrix result = prod(AInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"f"<<std::endl;//(check for row major blas::Upper arguments)
	{
		RealMatrix Atrans = trans(A);
		RealMatrix testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::Upper>(Atrans,testResult);
		RealMatrix result = prod(trans(AInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	
	//now we test all combinations of transpositions
	//for blas::Unit-matrices
	std::cout<<"a"<<std::endl;
	{
		RealMatrix testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::UnitLower>(A,testResult);
		RealMatrix result = prod(unitAInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"b"<<std::endl;
	{
		RealMatrix testResult = subrange(input,0,1,0,2);
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::UnitLower>(A,testResult);
		RealMatrix result = prod(subrange(input,0,1,0,2),unitAInv);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"c"<<std::endl;
	{
		RealMatrix testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::UnitUpper>(trans(A),testResult);
		RealMatrix result = prod(trans(unitAInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"d"<<std::endl;
	{
		RealMatrix testResult = subrange(input,0,1,0,2);
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::UnitUpper>(trans(A),testResult);
		RealMatrix result = prod(subrange(input,0,1,0,2),trans(unitAInv));
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"e"<<std::endl;//(check for column major blas::Lower arguments)
	{
		blas::matrix<double,blas::column_major> AcolMaj = A;
		blas::matrix<double,blas::column_major> testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::UnitLower>(AcolMaj,testResult);
		RealMatrix result = prod(unitAInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"f"<<std::endl;//(check for row major blas::Upper arguments)
	{
		RealMatrix Atrans = trans(A);
		RealMatrix testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::UnitUpper>(Atrans,testResult);
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
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::Lower>(A,testResult);
		RealVector result = prod(AInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"b"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::Upper>(trans(A),testResult);
		RealVector result = prod(trans(AInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"c"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::Lower>(A,testResult);
		RealVector result = prod(input,AInv);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"d"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::Upper>(trans(A),testResult);
		RealVector result = prod(input,trans(AInv));
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"e"<<std::endl;//(check for column major blas::Lower arguments)
	{
		blas::matrix<double,blas::column_major> AcolMaj = A;
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::Lower>(AcolMaj,testResult);
		RealVector result = prod(AInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"f"<<std::endl;//(check for row major blas::Upper arguments)
	{
		RealMatrix Atrans = trans(A);
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::Upper>(Atrans,testResult);
		RealVector result = prod(trans(AInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	
	//now we test all combinations of transpositions
	//for blas::Unit-matrices
	std::cout<<"a"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::UnitLower>(A,testResult);
		RealVector result = prod(unitAInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"b"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::UnitUpper>(trans(A),testResult);
		RealVector result = prod(trans(unitAInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"c"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::UnitLower>(A,testResult);
		RealVector result = prod(input,unitAInv);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"d"<<std::endl;
	{
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveXAB,blas::UnitUpper>(trans(A),testResult);
		RealVector result = prod(input,trans(unitAInv));
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"e"<<std::endl;//(check for column major blas::Lower arguments)
	{
		blas::matrix<double,blas::column_major> AcolMaj = A;
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::UnitLower>(AcolMaj,testResult);
		RealVector result = prod(unitAInv,input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"f"<<std::endl;//(check for row major blas::Upper arguments)
	{
		RealMatrix Atrans = trans(A);
		RealVector testResult = input;
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::UnitUpper>(Atrans,testResult);
		RealVector result = prod(trans(unitAInv),input);
		double error = norm_inf(result-testResult);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
}

//for the remaining functions, we can use random systems and check, whether they are okay

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
	fast_prod(R,Atemp,A);
	return A;
}

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
		fast_prod(A,X,test);
		
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
