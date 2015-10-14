#define BOOST_TEST_MODULE Trainers_Kernel_Normalization
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/NormalizeKernelUnitVariance.h>
#include <shark/Models/Kernels/LinearKernel.h>
//#include <shark/Models/Kernels/MklKernel.h>
#include <shark/Models/Kernels/PolynomialKernel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>

using namespace shark;



BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_KernelNormalization)

BOOST_AUTO_TEST_CASE( Normalize_Kernel_Unit_Variance_InFeatureSpace_Simple )
{
	std::size_t num_points = 3;
	std::vector<RealVector> input(num_points);
	RealVector v(1);
	v(0) = 0.0; input[0] = v;
	v(0) = 1.0; input[1] = v;
	v(0) = 2.0; input[2] = v;
	UnlabeledData<RealVector> data = createDataFromRange(input);
	DenseLinearKernel lin;
	DenseScaledKernel scale( &lin );
	NormalizeKernelUnitVariance<> normalizer;
	normalizer.train( scale, data );
	std::cout << "    done training. factor is " << scale.factor() << std::endl;
	std::cout << "    mean                   = " << normalizer.mean() << std::endl;
	std::cout << "    trace                  = " << normalizer.trace() << std::endl;
	BOOST_CHECK_SMALL( scale.factor() - 1.5, 1e-12);
	{ //check in feature space
		double control = 0.0;
		for ( std::size_t i=0; i<num_points; i++ ) {
			control += scale.eval(input[i], input[i]);
			for ( std::size_t j=0; j<num_points; j++ ) {
				control -= scale.eval(input[i],input[j]) / num_points;
			}
		}
		control /= num_points;
		BOOST_CHECK_SMALL( control - 1.0, 1e-12 );
	}
	{ //check in input space
		double control = 0.0;
		for ( std::size_t i=0; i<num_points; i++ ) {
			control += scale.factor()*inner_prod(input[i], input[i]);
			for ( std::size_t j=0; j<num_points; j++ ) {
				control -= scale.factor()*inner_prod(input[i],input[j]) / num_points;
			}
		}
		control /= num_points;
		BOOST_CHECK_SMALL( control - 1.0, 1e-12 );
	}
}

BOOST_AUTO_TEST_CASE( Normalize_Kernel_Unit_Variance_InFeatureSpace_LinearKernel )
{
	std::size_t num_dims = 10;
	std::size_t num_points = 1000;
	std::vector<RealVector> input(num_points);
	RealVector v(num_dims);
	for ( std::size_t i=0; i<num_points; i++ ) {
		for ( std::size_t j=0; j<num_dims; j++ ) {
			v(j) = Rng::uni(-20,20);
		}
		input[i] = v;
	}
	UnlabeledData<RealVector> data = createDataFromRange(input);
	DenseLinearKernel lin;
	DenseScaledKernel scale( &lin );
	NormalizeKernelUnitVariance<> normalizer;
	normalizer.train( scale, data );
	std::cout << "    done training. factor is " << scale.factor() << std::endl;
	std::cout << "    mean                   = " << normalizer.mean() << std::endl;
	std::cout << "    trace                  = " << normalizer.trace() << std::endl;
	{ //check in feature space
		double control = 0.0;
		for ( std::size_t i=0; i<num_points; i++ ) {
			control += scale.eval(input[i], input[i]);
			for ( std::size_t j=0; j<num_points; j++ ) {
				control -= scale.eval(input[i],input[j]) / num_points;
			}
		}
		control /= num_points;
		BOOST_CHECK_SMALL( control - 1.0, 1e-12 );
	}
	{ //check in input space
		double control = 0.0;
		for ( std::size_t i=0; i<num_points; i++ ) {
			control += scale.factor()*inner_prod(input[i], input[i]);
			for ( std::size_t j=0; j<num_points; j++ ) {
				control -= scale.factor()*inner_prod(input[i],input[j]) / num_points;
			}
		}
		control /= num_points;
		BOOST_CHECK_SMALL( control - 1.0, 1e-12 );
	}
}

BOOST_AUTO_TEST_CASE( Normalize_Kernel_Unit_Variance_InFeatureSpace_GaussianKernel )
{
	std::size_t num_dims = 10;
	std::size_t num_points = 1000;
	std::vector<RealVector> input(num_points);
	RealVector v(num_dims);
	for ( std::size_t i=0; i<num_points; i++ ) {
		for ( std::size_t j=0; j<num_dims; j++ ) {
			v(j) = Rng::uni(-1,1);
		}
		input[i] = v;
	}
	UnlabeledData<RealVector> data = createDataFromRange(input);
	DenseRbfKernel kernel(0.01);
	DenseScaledKernel scale( &kernel );
	NormalizeKernelUnitVariance<> normalizer;
	normalizer.train( scale, data );
	std::cout << "    done training. factor is " << scale.factor() << std::endl;
	std::cout << "    mean                   = " << normalizer.mean() << std::endl;
	std::cout << "    trace                  = " << normalizer.trace() << std::endl;
	//check in feature space
	double control = 0.0;
	for ( std::size_t i=0; i<num_points; i++ ) {
		control += scale.eval(input[i], input[i]);
		for ( std::size_t j=0; j<num_points; j++ ) {
			control -= scale.eval(input[i],input[j]) / num_points;
		}
	}
	control /= num_points;
	BOOST_CHECK_SMALL( control - 1.0, 1e-12 );
}
//BOOST_AUTO_TEST_CASE( Normalize_Kernel_Unit_Variance_InFeatureSpace_MklKernel )
//{
//	std::size_t num_dims = 9;
//	std::size_t num_points = 1000;
//	std::vector<RealVector> input(num_points);
//	RealVector v(num_dims);
//	for ( std::size_t i=0; i<num_points; i++ ) {
//		for ( std::size_t j=0; j<num_dims; j++ ) {
//			v(j) = Rng::uni(-1,1);
//		}
//		input[i] = v;
//	}
//	UnlabeledData<RealVector> data(input);
//	
//	DenseRbfMklKernel   	  basekernel1(0.1);
//	DenseLinearMklKernel      basekernel2;
//	DensePolynomialMklKernel  basekernel3(2, 1.0);
//	
//	std::vector< DenseMklKernelFunction * > kernels;
//	kernels.push_back(&basekernel1);
//	kernels.push_back(&basekernel2);
//	kernels.push_back(&basekernel3);
//	
//	std::vector< std::pair< std::size_t, std::size_t > > frs;
//	frs.push_back( std::make_pair( 0,3 ) );
//	frs.push_back( std::make_pair( 3,6 ) );
//	frs.push_back( std::make_pair( 6,9 ) );
//	
//	DenseMklKernel kernel( kernels, frs );
//	DenseScaledKernel scale( &kernel );
//	
//	NormalizeKernelUnitVariance<> normalizer;
//	normalizer.train( scale, data );
//	std::cout << "    done training. factor is " << scale.factor() << std::endl;
//	std::cout << "    mean                   = " << normalizer.mean() << std::endl;
//	std::cout << "    trace                  = " << normalizer.trace() << std::endl;
//	//check in feature space
//	double control = 0.0;
//	for ( std::size_t i=0; i<num_points; i++ ) {
//		control += scale.eval(input[i], input[i]);
//		for ( std::size_t j=0; j<num_points; j++ ) {
//			control -= scale.eval(input[i],input[j]) / num_points;
//		}
//	}
//	control /= num_points;
//	BOOST_CHECK_SMALL( control - 1.0, 1e-12 );
//}


BOOST_AUTO_TEST_SUITE_END()
