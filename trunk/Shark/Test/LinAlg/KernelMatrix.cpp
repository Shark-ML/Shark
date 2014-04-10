#define BOOST_TEST_MODULE ALGORITHMS_QP_KERNELMATRIX
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/QP/QuadraticProgram.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/Data/DataDistribution.h>
#include <shark/LinAlg/BlockMatrix2x2.h>
#include <shark/LinAlg/CachedMatrix.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/ModifiedKernelMatrix.h>
#include <shark/LinAlg/PrecomputedMatrix.h>
#include <shark/LinAlg/RegularizedKernelMatrix.h>

using namespace shark;


class Problem : public LabeledDataDistribution<RealVector,unsigned int>
{
public:
	void draw(RealVector& input,unsigned int& label) const{
		input.resize(5);
		label = Rng::coinToss(0.5)*2+1;
		for(std::size_t i = 0; i != 5; ++i){
			input(i) = Rng::uni(-1,1);
		}
	}
};


struct Fixture {

	Fixture():size(100){
		Problem problem;
		data = problem.generateDataset(size,9);
		
		//create standard kernel matrix
		kernelMatrix = calculateRegularizedKernelMatrix(kernel,data.inputs());
	}
	std::size_t size;
	
	LabeledData<RealVector,unsigned int> data;
	LinearKernel<> kernel;
	
	RealMatrix kernelMatrix;
};


template<class MatrixType, class Result>
void testFullMatrix(MatrixType& matrix, Result const& result){
	std::size_t size = matrix.size();
	//calculate full matrix
	RealMatrix matrixResult(size,size);
	matrix.matrix(matrixResult);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j != size; ++j){
			BOOST_CHECK_SMALL(matrixResult(i,j)-result(i,j),1.-13);
		}
	}
}

template<class MatrixType, class Result>
void testMatrix(MatrixType& matrix, Result const& result){
	std::size_t size = matrix.size();
	//check entry
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j != size; ++j){
			BOOST_CHECK_SMALL(matrix(i,j)-result(i,j),1.-13);
		}
	}
	//check row
	RealVector matrixRow(size);
	std::size_t start = size/4;
	std::size_t end = size/2;
	for(std::size_t i = 0; i != size; ++i){
		matrix.row(i,0,size,&matrixRow[0]);//full row
		for(std::size_t j = 0; j != size; ++j)
			BOOST_CHECK_SMALL(matrixRow(j)-result(i,j),1.e-13);
		matrix.row(i,start,end,&matrixRow[start]);//subrow
		for(std::size_t j = start; j != end; ++j)
			BOOST_CHECK_SMALL(matrixRow(j)-result(i,j),1.e-13);
	}

	//flip columns
	matrix.flipColumnsAndRows(start,end);
	matrix.row(start,0,size,&matrixRow[0]);//full row
	for(std::size_t j = 0; j != start; ++j)
		BOOST_CHECK_SMALL(matrixRow(j)-result(end,j),1.e-13);
	BOOST_CHECK_SMALL(matrixRow(start)-result(end,end),1.e-13);
	for(std::size_t j = start+1; j != end; ++j)
		BOOST_CHECK_SMALL(matrixRow(j)-result(end,j),1.e-13);
	BOOST_CHECK_SMALL(matrixRow(end)-result(end,start),1.e-13);
	for(std::size_t j = end+1; j != size; ++j)
		BOOST_CHECK_SMALL(matrixRow(j)-result(end,j),1.e-13);
	
	matrix.row(end,0,size,&matrixRow[0]);//full row
	for(std::size_t j = 0; j != start; ++j)
		BOOST_CHECK_SMALL(matrixRow(j)-result(start,j),1.e-13);
	BOOST_CHECK_SMALL(matrixRow(start)-result(end,start),1.e-13);
	for(std::size_t j = start+1; j != end; ++j)
		BOOST_CHECK_SMALL(matrixRow(j)-result(start,j),1.e-13);
	BOOST_CHECK_SMALL(matrixRow(end)-result(start,start),1.e-13);
	for(std::size_t j = end+1; j != size; ++j)
		BOOST_CHECK_SMALL(matrixRow(j)-result(start,j),1.e-13);
}

BOOST_FIXTURE_TEST_SUITE(Algorithms_QP_KernelMatrix, Fixture)

BOOST_AUTO_TEST_CASE( QP_KernelMatrix ) {
	RealMatrix matrix = kernelMatrix;
	KernelMatrix<RealVector,double> km(kernel,data.inputs());
	testFullMatrix(km,matrix);
	testMatrix(km,matrix);
}

BOOST_AUTO_TEST_CASE( QP_RegularizedKernelMatrix ) {
	RealMatrix matrix = kernelMatrix;
	RealVector diagVec(size);
	for(std::size_t i = 0; i != size; ++i){
		diagVec(i) = i;
	}
	diag(matrix) += diagVec;
	RegularizedKernelMatrix<RealVector,double> km(kernel,data.inputs(),diagVec);
	
	testFullMatrix(km,matrix);
	testMatrix(km,matrix);
	
}
BOOST_AUTO_TEST_CASE( QP_ModifiedKernelMatrix ) {
	double sameClass = 2;
	double diffClass = -0.5;
	RealMatrix matrix = kernelMatrix;
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j != size; ++j){
			if(data.element(i).label == data.element(j).label)
				matrix(i,j) *= sameClass;
			else
				matrix(i,j) *= diffClass;
		}
	}
	ModifiedKernelMatrix<RealVector,double> km(kernel,data,sameClass,diffClass);

	testFullMatrix(km,matrix);
	testMatrix(km,matrix);
}

BOOST_AUTO_TEST_CASE( QP_BlockMatrix ) {
	RealMatrix matrix(2*size,2*size);
	subrange(matrix,0,100,0,100) = kernelMatrix;
	subrange(matrix,0,100,100,200) = kernelMatrix;
	subrange(matrix,100,200,0,100) = kernelMatrix;
	subrange(matrix,100,200,100,200) = kernelMatrix;
	KernelMatrix<RealVector,double> kmbase(kernel,data.inputs());
	BlockMatrix2x2<KernelMatrix<RealVector,double> > km(&kmbase);

	testFullMatrix(km,matrix);
	testMatrix(km,matrix);
}

BOOST_AUTO_TEST_CASE( QP_PrecomputedMatrix ) {

	KernelMatrix<RealVector,double> km(kernel,data.inputs());
	PrecomputedMatrix<KernelMatrix<RealVector,double> > cache(&km);
	testMatrix(cache,kernelMatrix);
}

BOOST_AUTO_TEST_CASE( QP_CachedMatrix_Simple ) {
	std::size_t numRowsToStore = 10;
	std::size_t cacheSize = numRowsToStore*size;
	
	KernelMatrix<RealVector,double> km(kernel,data.inputs());
	CachedMatrix<KernelMatrix<RealVector,double> > cache(&km,cacheSize);
	BOOST_REQUIRE_EQUAL(cache.getMaxCacheSize(),cacheSize);
	BOOST_REQUIRE_EQUAL(cache.getCacheSize(),0);
	
	testMatrix(cache,kernelMatrix);
	//this last call should not have cached anything
	BOOST_REQUIRE_EQUAL(cache.getCacheSize(),0);
	for(std::size_t i = 0; i != size; ++i){
		BOOST_CHECK_EQUAL(cache.isCached(i), false);
		BOOST_CHECK_EQUAL(cache.getCacheRowSize(i), 0);
	}
}
BOOST_AUTO_TEST_CASE( QP_CachedMatrix_Flipping ) {
	std::size_t numRowsToStore = 10;
	std::size_t cacheSize = numRowsToStore*size;
	std::size_t simulationSteps = 1000;
	
	KernelMatrix<RealVector,double> km(kernel,data.inputs());
	KernelMatrix<RealVector,double> groundTruthMatrix(kernel,data.inputs());
	CachedMatrix<KernelMatrix<RealVector,double> > cache(&km,cacheSize);
	BOOST_REQUIRE_EQUAL(cache.getMaxCacheSize(),cacheSize);
	BOOST_REQUIRE_EQUAL(cache.getCacheSize(),0);

	//next do a running check that the cache works with flipping of rows and random accesses
	for(std::size_t t = 0; t != simulationSteps; ++t){
		std::size_t index = Rng::discrete(0,size-1);
		std::size_t accessSize = Rng::discrete(0.5*size,size-1);
		std::size_t flipi = Rng::discrete(0,size-1);
		std::size_t flipj = Rng::discrete(0,size-1);
		
		//access matrix cache and check whether the right values are returned
		double* line = cache.row(index,0,accessSize);
		for(std::size_t i = 0; i != accessSize; ++i){
			BOOST_CHECK_CLOSE(line[i],groundTruthMatrix(index,i), 1.e-10);
		}
		//flip
		cache.flipColumnsAndRows(flipi,flipj);
		groundTruthMatrix.flipColumnsAndRows(flipi,flipj);
	}
	
	//truncate the cache
	//~ cache.setMaxCachedIndex(10);
	//~ for(std::size_t i = 0; i != 10; ++i){
		//~ BOOST_CHECK(!cache.isCached(i) || cache.getCacheRowSize(i) <= 10 );
	//~ }
	
	//finally clear the cache, afterwards it should be empty
	cache.clear();
	BOOST_REQUIRE_EQUAL(cache.getCacheSize(),0);
	for(std::size_t i = 0; i != size; ++i){
		BOOST_CHECK_EQUAL(cache.isCached(i), false);
		BOOST_CHECK_EQUAL(cache.getCacheRowSize(i), 0);
	}
	
}


BOOST_AUTO_TEST_SUITE_END()
