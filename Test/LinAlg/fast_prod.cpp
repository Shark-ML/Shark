#define BOOST_TEST_MODULE LinAlg_Fast_Prod
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/Base.h>
#include <shark/Core/Timer.h>
#include <shark/Rng/GlobalRng.h>

using namespace shark;

BOOST_AUTO_TEST_CASE( LinAlg_fast_prod_matrix_matrix ){
	RealMatrix A(10,10);
	RealMatrix B(10,10);
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 10; ++j){
			A(i,j) = Rng::uni(0,1);
			B(i,j) = Rng::uni(0,1);
		} 
	}
	RealMatrix C(10,10);
	RealMatrix testC(10,10);
	
	
	std::cout<<"a"<<std::endl;
	axpy_prod(A,B,C,true);
	fast_prod(A,B,testC);
	double error = norm_inf(C-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	testC.clear();
	
	std::cout<<"b"<<std::endl;
	axpy_prod(trans(A),B,C,true);
	fast_prod(trans(A),B,testC);
	error = norm_inf(C-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	testC.clear();
	
	std::cout<<"c"<<std::endl;
	axpy_prod(A,trans(B),C,true);
	fast_prod(A,trans(B),testC);
	error = norm_inf(C-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	testC.clear();
	
	std::cout<<"d"<<std::endl;
	axpy_prod(trans(A),trans(B),C,true);
	fast_prod(trans(A),trans(B),testC);
	error = norm_inf(C-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
}
BOOST_AUTO_TEST_CASE( LinAlg_fast_prod_matrix_matrix_nonsquare ){
	RealMatrix A(10,5);
	RealMatrix B(5,8);
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 5; ++j){
			A(i,j) = Rng::uni(0,1);
		} 
	}
	for(std::size_t i = 0; i != 5; ++i){
		for(std::size_t j = 0; j != 8; ++j){
			B(i,j) = Rng::uni(0,1);
		} 
	}
	RealMatrix C(10,8);
	RealMatrix testC(10,8);
	RealMatrix ATrans=trans(A);
	RealMatrix BTrans=trans(B);
	
	axpy_prod(A,B,C,true);
	
	std::cout<<"a"<<std::endl;
	fast_prod(A,B,testC);
	double error = norm_inf(C-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	testC.clear();
	
	std::cout<<"b"<<std::endl;
	fast_prod(trans(ATrans),B,testC);
	error = norm_inf(C-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	testC.clear();
	
	std::cout<<"c"<<std::endl;
	fast_prod(A,trans(BTrans),testC);
	error = norm_inf(C-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	testC.clear();
	
	std::cout<<"d"<<std::endl;
	fast_prod(trans(ATrans),trans(BTrans),testC);
	error = norm_inf(C-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
}
BOOST_AUTO_TEST_CASE( LinAlg_fast_prod_matrix_matrix_sparse ){
	CompressedRealMatrix A(10,10);
	CompressedRealMatrix B(10,10);
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 10; ++j){
			A(i,j) = Rng::uni(0,1);
			B(i,j) = Rng::uni(0,1);
		}
	}
	CompressedRealMatrix C(10,10);
	CompressedRealMatrix testC(10,10);
	
	axpy_prod(A,B,C,true);
	fast_prod(A,B,testC);
	double error = norm_inf(C-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
	testC.clear();
	axpy_prod(trans(A),B,C,true);
	fast_prod(trans(A),B,testC);
	error = norm_inf(C-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
	testC.clear();
	axpy_prod(A,trans(B),C,true);
	fast_prod(A,trans(B),testC);
	error = norm_inf(C-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
	testC.clear();
	axpy_prod(trans(A),trans(B),C,true);
	fast_prod(trans(A),trans(B),testC);
	error = norm_inf(C-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
}

BOOST_AUTO_TEST_CASE( LinAlg_fast_prod_matrix_vector ){
	RealMatrix A(10,10);
	RealVector b(10);
	
	for(std::size_t i = 0; i != 10; ++i){
		b(i) = Rng::uni(0,1);
		for(std::size_t j = 0; j != 10; ++j){
			A(i,j) = Rng::uni(0,1);
		}
	}
	RealVector c(10);
	RealVector testC(10);
	
	axpy_prod(A,b,c,true);
	
	std::cout<<"a"<<std::endl;
	fast_prod(A,b,testC);
	double error = norm_inf(c-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	testC.clear();
	
	std::cout<<"b"<<std::endl;
	axpy_prod(trans(A),b,c,true);
	fast_prod(trans(A),b,testC);
	error = norm_inf(c-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
	std::cout<<"c"<<std::endl;
	testC.clear();
	c.clear();
	RealSubMatrix Asub=subrange(A,0,5,0,5);
	RealSubVector bSub = subrange(b,0,5);
	RealSubVector cSub = subrange(c,0,5);
	RealMatrix Asubcopy = Asub;
	axpy_prod(Asub,bSub,cSub,true);
	fast_prod(Asubcopy,subrange(b,0,5),subrange(testC,0,5));
	error = norm_inf(c-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
	std::cout<<"d"<<std::endl;
	//subrange AND transpose
	testC.clear();
	c.clear();
	axpy_prod(trans(Asub),bSub,cSub,true);
	fast_prod(trans(Asubcopy),subrange(b,0,5),subrange(testC,0,5));
	error = norm_inf(c-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
}
BOOST_AUTO_TEST_CASE( LinAlg_fast_prod_matrix_vector_non_square ){
	RealMatrix A(5,10);
	RealVector b(10);
	
	for(std::size_t i = 0; i != 5; ++i){
		b(i) = Rng::uni(0,1);
		for(std::size_t j = 0; j != 10; ++j){
			A(i,j) = Rng::uni(0,1);
		}
	}
	RealVector c(5);
	RealVector testC(5);
	RealMatrix Atrans=trans(A);
	
	axpy_prod(A,b,c,true);
	
	std::cout<<"a"<<std::endl;
	fast_prod(A,b,testC);
	double error = norm_inf(c-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
	std::cout<<"b"<<std::endl;
	testC.clear();
	axpy_prod(trans(Atrans),b,c,true);
	fast_prod(trans(Atrans),b,testC);
	error = norm_inf(c-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
	
	std::cout<<"c"<<std::endl;
	testC.clear();
	c.clear();
	RealSubMatrix Asub=subrange(A,1,4,1,3);
	RealSubVector bSub = subrange(b,1,3);
	RealSubVector cSub = subrange(c,1,4);
	RealMatrix Asubcopy = Asub;
	axpy_prod(Asub,bSub,cSub,true);
	fast_prod(subrange(A,1,4,1,3),subrange(b,1,3),subrange(testC,1,4));
	error = norm_inf(c-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
	
	std::cout<<"d"<<std::endl;
	//subrange AND transpose
	testC.clear();
	c.clear();
	RealSubMatrix Atranssub=subrange(Atrans,1,3,1,4);
	axpy_prod(trans(Atranssub),bSub,cSub,true);
	fast_prod(trans(Atranssub),subrange(b,1,3),subrange(testC,1,4));
	error = norm_inf(c-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
}
BOOST_AUTO_TEST_CASE( LinAlg_fast_prod_matrix_vector_sparse ){
	CompressedRealMatrix A(10,10);
	CompressedRealVector b(10);
	
	for(std::size_t i = 0; i != 10; ++i){
		b(i) = Rng::uni(0,1);
		for(std::size_t j = 0; j != 10; ++j){
			A(i,j) = Rng::uni(0,1);
		}
	}
	CompressedRealVector c(10);
	CompressedRealVector testC(10);
	
	axpy_prod(A,b,c,true);
	fast_prod(A,b,testC);
	double error = norm_inf(c-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
	testC.clear();
	axpy_prod(trans(A),b,c,true);
	fast_prod(trans(A),b,testC);
	error = norm_inf(c-testC);
	BOOST_CHECK_SMALL(error,1.e-10);
	
	testC.clear();
	c.clear();
	CompressedRealMatrix Asub=subrange(A,0,5,0,5);
	CompressedRealVector bSub = subrange(b,0,5);
	CompressedRealVector cSub = subrange(c,0,5);
	axpy_prod(Asub,bSub,cSub,true);
	fast_prod(Asub,subrange(b,0,5),subrange(testC,0,5));
	error = norm_inf(cSub-subrange(testC,0,5));
	BOOST_CHECK_SMALL(error,1.e-10);
	
	//subrange AND transpose
	testC.clear();
	c.clear();
	axpy_prod(trans(Asub),bSub,cSub,true);
	fast_prod(trans(Asub),subrange(b,0,5),subrange(testC,0,5));
	error = norm_inf(cSub-subrange(testC,0,5));
	BOOST_CHECK_SMALL(error,1.e-10);
	
}
#ifdef NDEBUG
#ifdef SHARK_USE_ATLAS
BOOST_AUTO_TEST_CASE( LinAlg_fast_prod_vector_BENCHMARK ){
	const std::size_t matRows=28*28;
	const std::size_t matColumns = 28*28;
	std::size_t iterations= 1000;	
	
	RealVector testResult(matRows);
	
	RealVector argument(matColumns);
	blas::matrix<double,blas::row_major> matrix(matRows,matColumns);
	
	//initialize everything
	for(std::size_t j = 0; j != matRows; ++j){
		for(std::size_t i = 0; i != matColumns; ++i){
			matrix(j,i)=Rng::uni(-1,1);
		}
	}
	for(std::size_t i = 0; i != matColumns; ++i){
		argument(i)=Rng::uni(-1,1);
	}
	for(std::size_t j = 0; j != matRows; ++j){
		testResult(j)=Rng::uni(-1,1);
	}
	
	blas::matrix<double,blas::row_major>  matrix2=trans(matrix);
	std::cout<<"Benchmarking matrix vector prod"<<std::endl;
	double start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		fast_prod(matrix,argument,testResult,true);
	}
	double end=Timer::now();
	std::cout<<"fast_prod Ax: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){

		fast_prod(trans(matrix2),argument,testResult,true);
	}
	end=Timer::now();
	std::cout<<"fast_prod A^Tx: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		noalias(testResult)+=prod(matrix,argument);
	}
	end=Timer::now();
	std::cout<<"prod Ax: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		noalias(testResult)+=prod(trans(matrix2),argument);
	}
	end=Timer::now();
	std::cout<<"prod A^Tx: "<<end-start<<std::endl;
	
	double output= 0;
	output += inner_prod(testResult,testResult);
	std::cout<<"anti optimization output: "<<output<<std::endl;
	
}
BOOST_AUTO_TEST_CASE( LinAlg_fast_prod_matrix_BENCHMARK_MEDIUM ){
	const std::size_t matRows=512;
	const std::size_t matColumns = 512;
	std::size_t iterations= 5;	
	
	blas::matrix<double,blas::row_major> testResult(matRows,matRows);
	blas::matrix<double,blas::row_major> matrix(matRows,matColumns);
	blas::matrix<double,blas::row_major> matrix2(matColumns,matRows);
	testResult.clear();
	
	//initialize everything
	for(std::size_t j = 0; j != matRows; ++j){
		for(std::size_t i = 0; i != matColumns; ++i){
			matrix(i,j)=Rng::uni(-1,1);
			matrix2(j,i)=Rng::uni(-1,1);
		}
	}
	
	blas::matrix<double,blas::row_major>  matrix3=trans(matrix);
	blas::matrix<double,blas::row_major>  matrix4=trans(matrix2);
	std::cout<<"Benchmarking matrix matrix prod for medium sized matrices"<<std::endl;
	double start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		fast_prod(matrix,matrix2,testResult,true);
	}
	double end=Timer::now();
	std::cout<<"fast_prod AX: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){

		fast_prod(trans(matrix3),matrix2,testResult,true);
	}
	end=Timer::now();
	std::cout<<"fast_prod A^TX: "<<end-start<<std::endl;
	
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){

		fast_prod(matrix,trans(matrix4),testResult,1.0);
	}
	end=Timer::now();
	std::cout<<"fast_prod AX^T: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){

		fast_prod(trans(matrix3),trans(matrix4),testResult,1.0);
	}
	end=Timer::now();
	std::cout<<"fast_prod A^TX^T: "<<end-start<<std::endl;
	
	
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		axpy_prod(matrix,matrix2,testResult,false);
	}
	end=Timer::now();
	std::cout<<"axpy_prod AX: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		axpy_prod(trans(matrix3),matrix2,testResult,false);
	}
	end=Timer::now();
	std::cout<<"axpy_prod A^TX: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		axpy_prod(matrix,trans(matrix4),testResult,false);
	}
	end=Timer::now();
	std::cout<<"axpy_prod AX^T: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		axpy_prod(trans(matrix3),trans(matrix4),testResult,false);
	}
	end=Timer::now();
	std::cout<<"axpy_prod A^TX^T: "<<end-start<<std::endl;
	
	double output= 0;
	output += inner_prod(row(testResult,0),column(testResult,0));
	std::cout<<"anti optimization output: "<<output<<std::endl;
	
}
BOOST_AUTO_TEST_CASE( LinAlg_fast_prod_matrix_BENCHMARK_SMALL ){
	const std::size_t matRows=50;
	const std::size_t matColumns = 50;
	std::size_t iterations= 2000;	
	
	blas::matrix<double,blas::row_major> testResult(matRows,matRows);
	blas::matrix<double,blas::row_major> matrix(matRows,matColumns);
	blas::matrix<double,blas::row_major> matrix2(matColumns,matRows);
	testResult.clear();
	
	//initialize everything
	for(std::size_t j = 0; j != matRows; ++j){
		for(std::size_t i = 0; i != matColumns; ++i){
			matrix(i,j)=Rng::uni(-1,1);
			matrix2(j,i)=Rng::uni(-1,1);
		}
	}
	
	blas::matrix<double,blas::row_major>  matrix3=trans(matrix);
	blas::matrix<double,blas::row_major>  matrix4=trans(matrix2);
	std::cout<<"Benchmarking matrix matrix prod for small matrices"<<std::endl;
	double start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		fast_prod(matrix,matrix2,testResult,1.0);
	}
	double end=Timer::now();
	std::cout<<"fast_prod AX: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){

		fast_prod(trans(matrix3),matrix2,testResult,1.0);
	}
	end=Timer::now();
	std::cout<<"fast_prod A^TX: "<<end-start<<std::endl;
	
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){

		fast_prod(matrix,trans(matrix4),testResult,1.0);
	}
	end=Timer::now();
	std::cout<<"fast_prod AX^T: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){

		fast_prod(trans(matrix3),trans(matrix4),testResult,1.0);
	}
	end=Timer::now();
	std::cout<<"fast_prod A^TX^T: "<<end-start<<std::endl;
	
	
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		axpy_prod(matrix,matrix2,testResult,false);
	}
	end=Timer::now();
	std::cout<<"axpy_prod AX: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		axpy_prod(trans(matrix3),matrix2,testResult,false);
	}
	end=Timer::now();
	std::cout<<"axpy_prod A^TX: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		axpy_prod(matrix,trans(matrix4),testResult,false);
	}
	end=Timer::now();
	std::cout<<"axpy_prod AX^T: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		axpy_prod(trans(matrix3),trans(matrix4),testResult,false);
	}
	end=Timer::now();
	std::cout<<"axpy_prod A^TX^T: "<<end-start<<std::endl;
	
	double output= 0;
	output += inner_prod(row(testResult,0),column(testResult,0));
	std::cout<<"anti optimization output: "<<output<<std::endl;
}
#endif
#endif
