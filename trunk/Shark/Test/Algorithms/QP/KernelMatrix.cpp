#define BOOST_TEST_MODULE ALGORITHMS_QP_KERNELMATRIX
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/QP/QuadraticProgram.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/Data/DataDistribution.h>

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

	testMatrix(km,matrix);
}


BOOST_AUTO_TEST_SUITE_END()
