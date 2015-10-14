#define BOOST_TEST_MODULE LinAlg_Initialize
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <vector>
#include <shark/LinAlg/Initialize.h>
#include <shark/Models/FFNet.h>

using namespace shark;
using namespace std;

BOOST_AUTO_TEST_SUITE (LinAlg_Initialize)

BOOST_AUTO_TEST_CASE( LinAlg_init_test_base ){
	//check that the first step allready works
	UIntVector vec1(1),vec2(1);
	vec1.clear(); 
	vec2.clear();
	init(vec1)<<1u;
	init(vec2)<<vec1;
	BOOST_CHECK_EQUAL(vec1(0),1u);
	BOOST_CHECK_EQUAL(vec2(0),1u);
	
	//const vector arguments and variable arguments
	const UIntVector constVec = vec1;
	vec2.clear();
	int a = 1u;
	init(vec1)<< a ;
	init(vec2)<< constVec;
	BOOST_CHECK_EQUAL(vec1(0),1u);
	BOOST_CHECK_EQUAL(vec2(0),1u);
	
	//now multiple values
	UIntVector vec3(5);
	vec3.clear();
	init(vec3)<<1u,2u,3u,4u,5u;
	for(std::size_t i = 0; i != 5;++i){
		BOOST_CHECK_EQUAL(vec3(i),i+1);
	}
	
	//vector and multiple values
	UIntVector vec4(5);
	vec4.clear();
	init(vec4)<<vec1,2u,3u,4u,5u;
	for(std::size_t i = 0; i != 5;++i){
		BOOST_CHECK_EQUAL(vec4(i),i+1);
	}
	
	//multiple vectors
	UIntVector vec5(6);
	vec5.clear();
	init(vec5)<<vec3,vec1;
	for(std::size_t i = 0; i != 5;++i){
		BOOST_CHECK_EQUAL(vec5(i),i+1);
	}
	BOOST_CHECK_EQUAL(vec5(5),1u);

	//values and vector
	UIntVector vec6(6);
	vec6.clear();
	init(vec6)<<8,vec3;
	for(std::size_t i = 1u; i != 6;++i){
		BOOST_CHECK_EQUAL(vec6(i),i);
	}
	BOOST_CHECK_EQUAL(vec6(0),8u);
	
	//subrange and values
	init(subrange(vec6,0,3))<<3,4,5;
	for(std::size_t i = 0; i != 3;++i){
		BOOST_CHECK_EQUAL(vec6(i),i+3);
		BOOST_CHECK_EQUAL(vec6(i+3),i+3);
	}
}
BOOST_AUTO_TEST_CASE( LinAlg_init_test_vectorSet ){
	std::vector<UIntVector> vectors(3,UIntVector(3));
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			vectors[i](j)=1+i*3+j;
		}
	}
	
	//check vectorSet in the middle of an expression
	UIntVector result(11);
	init(result)<<0,vectorSet(vectors),10;
	for(std::size_t i = 0; i != 11; ++i){
		BOOST_CHECK_EQUAL(result(i),i);
	}
	
	//check vectorSet in the beginning of an expression
	result.clear();
	init(result)<<vectorSet(vectors),10,11;
	for(std::size_t i = 0; i != 11; ++i){
		BOOST_CHECK_EQUAL(result(i),i+1);
	}
	
	//now const
	const std::vector<UIntVector> constVectors = vectors;
	result.clear();
	init(result)<<0,vectorSet(constVectors),10;
	for(std::size_t i = 0; i != 11; ++i){
		BOOST_CHECK_EQUAL(result(i),i);
	}
	result.clear();
	init(result)<<vectorSet(constVectors),10,11;
	for(std::size_t i = 0; i != 11; ++i){
		BOOST_CHECK_EQUAL(result(i),i+1);
	}
}
BOOST_AUTO_TEST_CASE( LinAlg_init_test_toVector ){
	//test dense matrix
	IntMatrix matrix(3,3);
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			matrix(i,j)=1u+i*3+j;
		}
	}
	
	//check in the middle of an expression
	UIntVector result(11);
	init(result)<<0,toVector(matrix),10;
	for(std::size_t i = 0; i != 11; ++i){
		BOOST_CHECK_EQUAL(result(i),i);
	}
	//check in the beginning of an expression
	result.clear();
	init(result)<<toVector(matrix),10,11;
	for(std::size_t i = 0; i != 11; ++i){
		BOOST_CHECK_EQUAL(result(i),i+1);
	}
	
	//test sparse matrix
	CompressedIntMatrix sparseMatrix(3,3);
	sparseMatrix.clear();
	for(std::size_t i = 0; i != 3; ++i){
		sparseMatrix(i,i)=i+1;
	}
	
	//checkin the middle of an expression
	RealVector sparseResult(5);
	init(sparseResult)<<0,toVector(sparseMatrix),4;
	for(std::size_t i = 0; i != 5; ++i){
		BOOST_CHECK_EQUAL(sparseResult(i),i);
	}
	//check vectorSet in the beginning of an expression
	sparseResult.clear();
	init(sparseResult)<<toVector(sparseMatrix),4,5;
	for(std::size_t i = 0; i != 5; ++i){
		BOOST_CHECK_EQUAL(sparseResult(i),i+1);
	}
	
}
BOOST_AUTO_TEST_CASE( LinAlg_init_test_matrixSet ){
	IntMatrix matrix(3,3);
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			matrix(i,j)=1u+i*3+j;
		}
	}
	std::vector<IntMatrix> matrices(3);
	matrices[0]=matrix;
	matrices[1]=2*matrix;
	matrices[2]=3*matrix;
	
	//check matrixSet in the middle of an expression
	UIntVector result(29);
	init(result)<<0,matrixSet(matrices),28u;
	BOOST_CHECK_EQUAL(result(0),0u);
	BOOST_CHECK_EQUAL(result(28),28u);
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 9; ++j){
			BOOST_CHECK_EQUAL(result(i*9+j+1),(1+j)*(1+i));
		}
	}
	
	//check vectorSet in the beginning of an expression
	result.clear();
	init(result)<<matrixSet(matrices),28u,29u;
	BOOST_CHECK_EQUAL(result(27),28u);
	BOOST_CHECK_EQUAL(result(28),29u);
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 9; ++j){
			BOOST_CHECK_EQUAL(result(i*9+j),(1+j)*(1+i));
		}
	}
	
	//now const
	const std::vector<IntMatrix> constMatrices = matrices;
	result.clear();
	init(result)<<0,matrixSet(constMatrices),28u;
	BOOST_CHECK_EQUAL(result(0),0u);
	BOOST_CHECK_EQUAL(result(28),28u);
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 9; ++j){
			BOOST_CHECK_EQUAL(result(i*9+j+1),(1+j)*(1+i));
		}
	}
	result.clear();
	init(result)<<matrixSet(constMatrices),28u,29u;
	BOOST_CHECK_EQUAL(result(27),28u);
	BOOST_CHECK_EQUAL(result(28),29u);
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 9; ++j){
			BOOST_CHECK_EQUAL(result(i*9+j),(1+j)*(1+i));
		}
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_init_test_parameters ){
	//initialize network
	FFNet<LogisticNeuron,LogisticNeuron> network;
	network.setStructure(2,5,2);
	initRandomNormal(network,1);
	
	RealVector result(network.numberOfParameters());
	init(result)<<blas::parameters(network);

	BOOST_CHECK_SMALL(norm_sqr(result-network.parameterVector()),1.e-10);
}

BOOST_AUTO_TEST_CASE( LinAlg_init_test_parameterSet ){
	//initialize networks
	std::vector<FFNet<LogisticNeuron,LogisticNeuron> > networks(3);
	for(std::size_t i = 0; i != 3; ++i){
		networks[i].setStructure(2,5,2);
		initRandomNormal(networks[i],1);
	}
	std::size_t n = networks[0].numberOfParameters();
	
	
	RealVector result(3*n);
	init(result)<<blas::parameterSet(networks);
	
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != n;++j){
			BOOST_CHECK_SMALL(result(i*n+j)-networks[i].parameterVector()(j),1.e-20);
		}
	}
}

//splitting of vectors

BOOST_AUTO_TEST_CASE( LinAlg_split_test_base ){
	//check that the first step allready works
	UIntVector vec1(1),vec2(1);
	vec1(0) = 1u;
	vec2.clear();
	unsigned int value = 0;
	init(vec1) >> value;
	init(vec1) >> vec2;
	BOOST_CHECK_EQUAL(value,1u);
	BOOST_CHECK_EQUAL(vec2(0),1u);
	
	//check again with const input
	const UIntVector constVec = vec1;
	vec2.clear();
	value = 0;
	init(constVec) >> value;
	init(constVec) >> vec2;
	BOOST_CHECK_EQUAL(value,1u);
	BOOST_CHECK_EQUAL(vec2(0),1u);
	
	//now multiple values
	UIntVector vec3(3);
	vec3(0) = 1u; vec3(1u) = 2; vec3(2) = 3;
	unsigned int a = 0;
	unsigned int b = 0;
	unsigned int c = 0;
	init(vec3) >> a,b,c;
	BOOST_CHECK_EQUAL(a,1u);
	BOOST_CHECK_EQUAL(b,2u);
	BOOST_CHECK_EQUAL(c,3u);

	//multiple vectors
	UIntVector vec4(1); vec4.clear();
	UIntVector vec5(2); vec5.clear();
	init(vec3) >> vec4,vec5;
	BOOST_CHECK_EQUAL(vec4(0),1u);
	BOOST_CHECK_EQUAL(vec5(0),2u);
	BOOST_CHECK_EQUAL(vec5(1),3u);
	
	//vector and multiple values
	vec4.clear();
	a=0;
	b=0;
	init(vec3) >> vec4,b,c;
	BOOST_CHECK_EQUAL(vec4(0),1u);
	BOOST_CHECK_EQUAL(b,2u);
	BOOST_CHECK_EQUAL(c,3u);
	
	//value and vector
	vec5.clear();
	a=0;
	init(vec3) >> a,vec5;
	BOOST_CHECK_EQUAL(a,1u);
	BOOST_CHECK_EQUAL(vec5(0),2u);
	BOOST_CHECK_EQUAL(vec5(1),3u);
	
	//subrange as source vector
	b = 0;
	c = 0;
	init(subrange(vec3,1,3)) >> b,c;
	BOOST_CHECK_EQUAL(b,2u);
	BOOST_CHECK_EQUAL(c,3u);
	
	//subrange as target vector
	a = 0;
	b= 0;
	vec5.clear();
	init(vec3) >> a,b,subrange(vec5,1u,2); 
	BOOST_CHECK_EQUAL(a,1u);
	BOOST_CHECK_EQUAL(b,2u);
	BOOST_CHECK_EQUAL(vec5(0),0u);
	BOOST_CHECK_EQUAL(vec5(1u),3u);
	
	b = 0;
	c = 0;
	vec5.clear();
	init(vec3) >> subrange(vec5,1,2),b,c; 
	BOOST_CHECK_EQUAL(vec5(0),0u);
	BOOST_CHECK_EQUAL(vec5(1),1u);
	BOOST_CHECK_EQUAL(b,2u);
	BOOST_CHECK_EQUAL(c,3u);
}

BOOST_AUTO_TEST_CASE( LinAlg_split_test_vectorSet ){
	std::vector<UIntVector> vectors(3,RealVector(3));
	UIntVector input(11);
	for(std::size_t i = 0; i != 11; ++i){
		input(i) = i+1;
	}
	
	//check vectorSet in the middle of an expression
	unsigned int a = 0;
	unsigned int b = 0;
	init(input) >> a,vectorSet(vectors),b;
	BOOST_CHECK_EQUAL(a,1u);
	BOOST_CHECK_EQUAL(b,11u);
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			BOOST_CHECK_EQUAL(vectors[i](j),2+i*3+j);
		}
	}
	
	
	//check vectorSet in the beginning of an expression
	a = 0;
	b = 0;
	for(std::size_t i = 0; i != 3; ++i){
		vectors[i].clear();
	}
	
	init(input ) >> vectorSet(vectors),a,b;
	
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			BOOST_CHECK_EQUAL(vectors[i](j),1+i*3+j);
		}
	}
	BOOST_CHECK_EQUAL(a,10u);
	BOOST_CHECK_EQUAL(b,11u);
	
}
BOOST_AUTO_TEST_CASE( LinAlg_split_test_toMatrix ){
	
	//test dense matrix
	//input
	UIntVector input(11);
	for(std::size_t i = 0; i != 11; ++i){
		input(i) = i+1;
	}
	//check in the middle of an expression
	UIntMatrix matrix(3,3);
	matrix.clear();
	unsigned int a = 0;
	unsigned int b = 0;
	init(input)>>a,toVector(matrix),b;
	BOOST_CHECK_EQUAL(a,1u);
	BOOST_CHECK_EQUAL(b,11u);
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			BOOST_CHECK_EQUAL(matrix(i,j),2u+i*3+j);
		}
	}
	//check in the beginning of an expression
	a = 0;
	b = 0;
	matrix.clear();
	init(input)>>toVector(matrix),a,b;
	BOOST_CHECK_EQUAL(a,10u);
	BOOST_CHECK_EQUAL(b,11u);
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			BOOST_CHECK_EQUAL(matrix(i,j),1u+i*3u+j);
		}
	}
	
	//test sparse matrix
	//first define entries and input
	CompressedUIntMatrix sparseMatrix(3,3);
	sparseMatrix(0,0)=25; sparseMatrix(1u,1u)=25; sparseMatrix(2,2)=25; 
	UIntVector sparseInput(5);
	for(std::size_t i = 0; i != 5; ++i){
		sparseInput(i) = i+1;
	}
	
	//check in the middle of an expression
	a = 0;
	b = 0;
	init(sparseInput)>>a,toVector(sparseMatrix),b;
	BOOST_CHECK_EQUAL(a,1u);
	BOOST_CHECK_EQUAL(b,5u);
	for(std::size_t i = 0; i != 3; ++i){
		BOOST_CHECK_EQUAL(sparseMatrix(i,i),i+2u);
	}
	//check vectorSet in the beginning of an expression
	sparseMatrix.clear();
	sparseMatrix(0,0)=25; sparseMatrix(1u,1u)=25; sparseMatrix(2,2)=25; 
	init(sparseInput)>>toVector(sparseMatrix),a,b;
	BOOST_CHECK_EQUAL(a,4u);
	BOOST_CHECK_EQUAL(b,5u);
	for(std::size_t i = 0; i != 3; ++i){
		BOOST_CHECK_EQUAL(sparseMatrix(i,i),i+1);
	}
	
}

BOOST_AUTO_TEST_CASE( LinAlg_split_test_matrixSet ){
	UIntVector input(29);
	for(std::size_t i = 0; i != 29; ++i){
		input(i) = i+1;
	}
	std::vector<UIntMatrix> matrices(3,IntMatrix(3,3));
	
	//check matrixSet in the middle of an expression
	unsigned int a = 0;
	unsigned int b = 0;
	init(input)>>a,matrixSet(matrices),b;
	
	BOOST_CHECK_EQUAL(a,1u);
	BOOST_CHECK_EQUAL(b,29u);
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			for(std::size_t k = 0; k != 3; ++k){
				BOOST_CHECK_EQUAL(matrices[i](j,k),i*9u+3u*j+k+2u);
			}
		}
	}
	
	//check vectorSet in the beginning of an expression
	a = 0;
	b = 0;
	matrices[0].clear(); 
	matrices[1u].clear(); 
	matrices[2].clear();
	init(input)>>matrixSet(matrices),a,b;
	BOOST_CHECK_EQUAL(a,28u);
	BOOST_CHECK_EQUAL(b,29u);
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			for(std::size_t k = 0; k != 3; ++k){
				BOOST_CHECK_EQUAL(matrices[i](j,k),i*9u+3u*j+k+1u);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_split_test_parameters ){
	//initialize network
	FFNet<LogisticNeuron,LogisticNeuron> network;
	network.setStructure(2,5,2);
	initRandomNormal(network,1u);
	
	RealVector input = network.parameterVector();
	initRandomNormal(network,1u);
	init(input) >> blas::parameters(network);

	BOOST_CHECK_SMALL(norm_sqr(input-network.parameterVector()),1.e-10);
}

BOOST_AUTO_TEST_CASE( LinAlg_split_test_parameterSet ){
	//initialize network
	std::vector<FFNet<LogisticNeuron,LogisticNeuron> > networks(3);
	for(std::size_t i = 0; i != 3; ++i){
		networks[i].setStructure(2,5,2);
		initRandomNormal(networks[i],1);
	}
	std::size_t n = networks[0].numberOfParameters();
	RealVector input(3*n);
	for(std::size_t i = 0; i != 3*n;++i){
		input(i) = Rng::uni(0,1);
	}
	
	init(input) >> blas::parameterSet(networks);
	
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != n;++j){
			BOOST_CHECK_SMALL(input(i*n+j)-networks[i].parameterVector()(j),1.e-20);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
