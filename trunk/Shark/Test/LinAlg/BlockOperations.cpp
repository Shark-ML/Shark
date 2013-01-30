#define BOOST_TEST_MODULE LinAlg_BlockOperations
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/BlockProducts.h>
#include <shark/LinAlg/Base.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Core/Timer.h>

#include<boost/preprocessor/repetition/repeat.hpp>
using namespace shark;

struct Product{
	template<class T>
	inline T operator()(T x, T y)const{
		return x*y;
		//double diff=x-y;
		//return diff*diff;
	}
};

BOOST_AUTO_TEST_CASE( LinAlg_detail_PanelBlockOperation){
	const std::size_t MatRows=20;
	const std::size_t MatColumns = 60;
	const std::size_t numVec = 4;
	
	blas::matrix<double,blas::row_major> result(numVec,MatRows);
	blas::matrix<double,blas::row_major> testResult(numVec,MatRows);
	
	blas::matrix<double,blas::row_major> argument(numVec,MatColumns);
	blas::matrix<double,blas::row_major> matrix(MatRows,MatColumns);
	
	//initialize everything
	for(std::size_t i = 0; i != MatColumns; ++i){
		for(std::size_t j = 0; j != numVec; ++j)
			argument(j,i)=Rng::uni(-1,1);
	}
	for(std::size_t j = 0; j != MatRows; ++j){
		for(std::size_t i = 0; i != MatColumns; ++i){
			matrix(j,i)=Rng::uni(-1,1);
		}
		for(std::size_t k = 0; k != numVec; ++k)
			result(k,j)=testResult(k,j)=Rng::uni(-1,1);
	}
	
	//evaluate the expected result
	testResult+=prod(argument,trans(matrix));
	detail::generalPanelBlockOperation(argument,matrix,result,Product());
	
	//std::cout<<result-testResult<<std::endl;
	
	for(std::size_t i = 0; i != numVec; ++i){
		double error = distanceSqr(row(result,i),row(testResult,i));
		BOOST_CHECK_SMALL(error,1.e-10);
	}
}
BOOST_AUTO_TEST_CASE( LinAlg_detail_BlockPanelOperation){
	const std::size_t MatRows=61;
	const std::size_t MatColumns = 21;
	const std::size_t numVec = 5;
	
	blas::matrix<double,blas::column_major> result(MatColumns,numVec);
	blas::matrix<double,blas::column_major> testResult(MatColumns,numVec);
	
	blas::matrix<double,blas::column_major> argument(MatRows,numVec);
	blas::matrix<double,blas::column_major> matrix(MatRows,MatColumns);
	
	//initialize everything
	for(std::size_t i = 0; i != MatColumns; ++i){
		for(std::size_t k = 0; k != numVec; ++k)
			result(i,k)=testResult(i,k)=Rng::uni(-1,1);
	}
	for(std::size_t j = 0; j != MatRows; ++j){
		for(std::size_t i = 0; i != MatColumns; ++i){
			matrix(j,i)=Rng::uni(-1,1);
		}
		for(std::size_t k = 0; k != numVec; ++k)
			argument(j,k)=Rng::uni(-1,1);
	}
	
	//evaluate the expected result
	testResult+=prod(trans(matrix),argument);
	detail::generalBlockPanelOperation(matrix,argument,result,Product());
	
	//std::cout<<result-testResult<<std::endl;
	
	for(std::size_t i = 0; i != numVec; ++i){
		double error = distanceSqr(column(result,i),column(testResult,i));
		BOOST_CHECK_SMALL(error,1.e-10);
	}
}
BOOST_AUTO_TEST_CASE( LinAlg_PanelPanelOperations_row_major ){
	std::size_t rowsC = 128;
	std::size_t columnsC = 128;
	std::size_t samples = 2048;
	
	blas::matrix<double,blas::row_major> result(rowsC,columnsC);
	blas::matrix<double,blas::row_major> testResult(rowsC,columnsC);
	
	blas::matrix<double,blas::row_major> matA(rowsC,samples);
	blas::matrix<double,blas::row_major> matB(samples,columnsC);
	
	for(std::size_t i = 0; i != rowsC; ++i){
		for(std::size_t j = 0; j != columnsC; ++j){
			result(i,j)=testResult(i,j)=Rng::uni(-1,1);
		}
	}
	for(std::size_t i = 0; i != rowsC; ++i){
		for(std::size_t j = 0; j != samples; ++j){
			matA(i,j)=Rng::uni(-1,1);
		}
	}
	for(std::size_t i = 0; i != samples; ++i){
		for(std::size_t j = 0; j != columnsC; ++j){
			matB(i,j)=Rng::uni(-1,1);
		}
	}
	testResult+=prod(matA,matB);
	generalPanelPanelOperation(matA,matB,result,Product());
	
	//std::cout<<result-testResult<<std::endl;
	
	for(std::size_t i = 0; i != rowsC; ++i){
		double error = distanceSqr(row(result,i),row(testResult,i));
		BOOST_CHECK_SMALL(error,1.e-10);
	}
}
BOOST_AUTO_TEST_CASE( LinAlg_MatrixMatrixOperations_row_major ){
	std::size_t rowsC = 254;
	std::size_t columnsC = 255;
	std::size_t samples = 256;
	
	blas::matrix<double,blas::row_major> result(rowsC,columnsC);
	blas::matrix<double,blas::row_major> testResult(rowsC,columnsC);
	
	blas::matrix<double,blas::row_major> matA(rowsC,samples);
	blas::matrix<double,blas::row_major> matB(samples,columnsC);
	
	for(std::size_t i = 0; i != rowsC; ++i){
		for(std::size_t j = 0; j != columnsC; ++j){
			result(i,j)=testResult(i,j)=Rng::uni(-1,1);
		}
	}
	for(std::size_t i = 0; i != rowsC; ++i){
		for(std::size_t j = 0; j != samples; ++j){
			matA(i,j)=Rng::uni(-1,1);
		}
	}
	for(std::size_t i = 0; i != samples; ++i){
		for(std::size_t j = 0; j != columnsC; ++j){
			matB(i,j)=Rng::uni(-1,1);
		}
	}
	testResult+=prod(matA,matB);
	generalMatrixMatrixOperation(matA,matB,result,Product());
	
	//std::cout<<result-testResult<<std::endl;
	
	for(std::size_t i = 0; i != rowsC; ++i){
		double error = distanceSqr(row(result,i),row(testResult,i));
		BOOST_CHECK_SMALL(error,1.e-10);
	}
}
BOOST_AUTO_TEST_CASE( LinAlg_PanelPanelOperations_column_major ){
	std::size_t rowsC = 128;
	std::size_t columnsC = 128;
	std::size_t samples = 2048;
	
	blas::matrix<double,blas::column_major> result(rowsC,columnsC);
	blas::matrix<double,blas::column_major> testResult(rowsC,columnsC);
	
	blas::matrix<double,blas::column_major> matA(rowsC,samples);
	blas::matrix<double,blas::column_major> matB(samples,columnsC);
	
	for(std::size_t i = 0; i != rowsC; ++i){
		for(std::size_t j = 0; j != columnsC; ++j){
			result(i,j)=testResult(i,j)=Rng::uni(-1,1);
		}
	}
	for(std::size_t i = 0; i != rowsC; ++i){
		for(std::size_t j = 0; j != samples; ++j){
			matA(i,j)=Rng::uni(-1,1);
		}
	}
	for(std::size_t i = 0; i != samples; ++i){
		for(std::size_t j = 0; j != columnsC; ++j){
			matB(i,j)=Rng::uni(-1,1);
		}
	}
	testResult+=prod(matA,matB);
	generalPanelPanelOperation(matA,matB,result,Product());
	
	//std::cout<<result-testResult<<std::endl;
	
	for(std::size_t i = 0; i != rowsC; ++i){
		double error = distanceSqr(row(result,i),row(testResult,i));
		BOOST_CHECK_SMALL(error,1.e-10);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_MatrixVectorOperations_row_major ){
	std::size_t args = 211;
	std::size_t results = 113;
	
	RealVector arguments(args);
	RealVector result(results);
	RealVector testResult(results);
	
	blas::matrix<double,blas::row_major> matA(results,args);
	
	for(std::size_t i = 0; i != results; ++i){
		result(i)=testResult(i)=Rng::uni(-1,1);
	}
	for(std::size_t i = 0; i != args; ++i){
		arguments(i)=Rng::uni(-1,1);
	}
	
	for(std::size_t i = 0; i != results; ++i){
		for(std::size_t j = 0; j != args; ++j){
			matA(i,j)=Rng::uni(-1,1);
		}
	}
	
	testResult+=prod(matA,arguments);
	generalMatrixVectorOperation(matA,arguments,result,Product());
	
	double error = distanceSqr(result,testResult);
	BOOST_CHECK_SMALL(error,1.e-10);
}

/////////////////BENCHMARKS/////////////////////////////
#ifdef NDEBUG 
BOOST_AUTO_TEST_CASE( LinAlg_MatrixVectorOperations_row_major_BENCHMARK ){
	std::size_t args = 1000;
	std::size_t results = 1000;
	std::size_t iterations=1000;
	
	RealVector arguments(args);
	RealVector result(results);
	
	blas::matrix<double,blas::row_major> matA(results,args);
	
	for(std::size_t i = 0; i != results; ++i){
		result(i)=Rng::uni(-1,1);
	}
	for(std::size_t i = 0; i != args; ++i){
		arguments(i)=Rng::uni(-1,1);
	}
	
	for(std::size_t i = 0; i != results; ++i){
		for(std::size_t j = 0; j != args; ++j){
			matA(i,j)=Rng::uni(-1,1);
		}
	}
	
	std::cout<<"starting benchmark: gemv_prod"<<std::endl;
	
	double start1=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		generalMatrixVectorOperation(matA,arguments,result,Product());
	}
	double end1=Timer::now();
	double start2=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		fast_prod(matA,arguments,result,1.0);
	}
	double end2=Timer::now();
	std::cout<<"gemv_prod: "<<end1-start1<<" fast_prod: "<<end2-start2<<std::endl;
	
	double output= 0;
	output += normSqr(result);
	std::cout<<"anti optimization output: "<<output<<std::endl;
}
BOOST_AUTO_TEST_CASE( LinAlg_detail_PanelBlockOperationBenchmark_row_major_BENCHMARK){
	const std::size_t MatRows=512;
	const std::size_t MatColumns = 512;
	const std::size_t numVectors = 2048;
	
	std::size_t iterations= 5;
	//std::cin>>iterations;	
	
	blas::matrix<double,blas::row_major> result(numVectors ,MatRows);
	
	blas::matrix<double,blas::row_major> argument(numVectors ,MatColumns);
	blas::matrix<double,blas::row_major> matrix(MatRows,MatColumns);
	
	//initialize everything
	for(std::size_t j = 0; j != MatRows; ++j){
		for(std::size_t i = 0; i != MatColumns; ++i){
			matrix(j,i)=Rng::uni(-1,1);
		}
	}
	for(std::size_t k = 0; k != numVectors; ++k){
		for(std::size_t i = 0; i != MatColumns; ++i){
			argument(k,i)=Rng::uni(-1,1);
		}
	}
	
	std::cout<<"starting benchmark: gepb_prod"<<std::endl;
	
	double start1=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		detail::generalPanelBlockOperation(argument,matrix,result,Product());
	}
	double end1=Timer::now();
	double start2=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		fast_prod(argument,trans(matrix),result,true);
	}
	double end2=Timer::now();
	std::cout<<"gepb_prod: "<<end1-start1<<" fast_prod: "<<end2-start2<<std::endl;
	
	double output= 0;
	for(std::size_t i = 0; i != result.size1(); ++i){
		output += inner_prod(row(result,i),row(result,i));
	}
	std::cout<<"anti optimization output: "<<output<<std::endl;
}
BOOST_AUTO_TEST_CASE( LinAlg_PanelPanelOperations_row_major_BENCHMARK ){
	std::size_t rowsC = 256;
	std::size_t columnsC = 256;
	std::size_t samples = 2048;
	
	std::size_t iterations = 5;
	
	blas::matrix<double,blas::row_major> result(rowsC,columnsC);
	blas::matrix<double,blas::row_major> testResult(rowsC,columnsC);
	
	blas::matrix<double,blas::row_major> matA(rowsC,samples);
	blas::matrix<double,blas::row_major> matB(samples,columnsC);
	
	for(std::size_t i = 0; i != rowsC; ++i){
		for(std::size_t j = 0; j != columnsC; ++j){
			result(i,j)=testResult(i,j)=0;
		}
	}
	for(std::size_t i = 0; i != rowsC; ++i){
		for(std::size_t j = 0; j != samples; ++j){
			matA(i,j)=Rng::uni(-1,1);
		}
	}
	for(std::size_t i = 0; i != samples; ++i){
		for(std::size_t j = 0; j != columnsC; ++j){
			matB(i,j)=Rng::uni(-1,1);
		}
	}
	std::cout<<"starting benchmark: gepp_prod "<<std::endl;
	
	double start1=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		generalPanelPanelOperation(matA,matB,result,Product());
	}
	double end1=Timer::now();
	double start2=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		fast_prod(matA,matB,testResult,1.0);
	}
	double end2=Timer::now();
	std::cout<<"gepp_prod: "<<end1-start1<<" fast_prod: "<<end2-start2<<std::endl;
	
	double output= 0;
	for(std::size_t i = 0; i != result.size1(); ++i){
		output += inner_prod(row(result,i),row(result,i));
	}
	std::cout<<"anti optimization output: "<<output<<std::endl;
}

BOOST_AUTO_TEST_CASE( LinAlg_MatrixMatrixOperations_row_major_BENCHMARK ){
	std::size_t rowsC = 2048;
	std::size_t columnsC = 2048;
	std::size_t samples = 200;
	
	std::size_t iterations = 8;
	
	blas::matrix<double,blas::row_major> result(rowsC,columnsC);
	blas::matrix<double,blas::row_major> testResult(rowsC,columnsC);
	
	blas::matrix<double,blas::row_major> matA(rowsC,samples);
	blas::matrix<double,blas::row_major> matB(samples,columnsC);
	
	for(std::size_t i = 0; i != rowsC; ++i){
		for(std::size_t j = 0; j != columnsC; ++j){
			result(i,j)=testResult(i,j)=0;
		}
	}
	for(std::size_t i = 0; i != rowsC; ++i){
		for(std::size_t j = 0; j != samples; ++j){
			matA(i,j)=Rng::uni(-1,1);
		}
	}
	for(std::size_t i = 0; i != samples; ++i){
		for(std::size_t j = 0; j != columnsC; ++j){
			matB(i,j)=Rng::uni(-1,1);
		}
	}
	std::cout<<"starting benchmark: gemm_prod "<<std::endl;
	RealMatrix transMatB=trans(matB);
	
	double start1=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		generalMatrixMatrixOperation(matA,matB,result,Product());
		//detail::generalPanelBlockOperation(matA,transMatB,result,Product());
	}
	double end1=Timer::now();
	double start2=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		fast_prod(matA,matB,testResult,1.0);
	}
	double end2=Timer::now();
	std::cout<<"gemm_prod: "<<end1-start1<<" fast_prod: "<<end2-start2<<std::endl;
	
	double output= 0;
	for(std::size_t i = 0; i != result.size1(); ++i){
		output += inner_prod(row(result,i),row(result,i));
	}
	std::cout<<"anti optimization output: "<<output<<std::endl;
}

#endif


