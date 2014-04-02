
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <shark/Data/Statistics.h>

#define BOOST_TEST_MODULE Rng_MultivariateNormal
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

BOOST_AUTO_TEST_CASE( MULTIVARIATENORMAL_EIGENVALUES ) {
	std::size_t Dimensions = 5;
	std::size_t Samples = 10000;
	
	//Generate covariance matrix
	RealMatrix base(Dimensions,2*Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j != 2*Dimensions; ++j){//2* to guarantue full rank.
			base(i,j) = Rng::gauss(0,1);
		}
	}
	RealMatrix covariance=prod(base,trans(base));
	covariance /= 2*Dimensions;
	
	
	MultiVariateNormalDistribution dist(covariance);
	Data<RealVector> sampleSet(Samples,RealVector(Dimensions));
	Data<RealVector> normalSampleSet(Samples,RealVector(Dimensions));
	
	for(std::size_t i = 0; i != Samples; ++i){
		MultiVariateNormalDistribution::result_type sample = dist();
		sampleSet.element(i) = sample.first;
		normalSampleSet.element(i) = sample.second;
	}
	
	RealVector meanSampled;
	RealMatrix covarianceSampled;
	RealVector normalMeanSampled;
	RealMatrix normalCovarianceSampled;
	
	meanvar(sampleSet,meanSampled,covarianceSampled);
	meanvar(normalSampleSet,normalMeanSampled,normalCovarianceSampled);
	
	//check that means are correct
	BOOST_CHECK_SMALL(norm_2(meanSampled)/Dimensions,1.e-2);
	BOOST_CHECK_SMALL(norm_2(normalMeanSampled)/Dimensions,1.e-2);
	
	//check that covariances are correct
	BOOST_CHECK_SMALL(norm_frobenius(covarianceSampled-covariance)/sqr(Dimensions),1.e-2);
	BOOST_CHECK_SMALL(
	norm_frobenius(
		normalCovarianceSampled-blas::identity_matrix<double>(Dimensions)
	)/sqr(Dimensions)
	,1.e-2);
	
}

BOOST_AUTO_TEST_CASE( MULTIVARIATENORMAL_Cholesky) {
	std::size_t Dimensions = 5;
	std::size_t Samples = 10000;
	
	//Generate covariance matrix
	RealMatrix base(Dimensions,2*Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j != 2*Dimensions; ++j){//2* to guarantue full rank.
			base(i,j) = Rng::gauss(0,1);
		}
	}
	RealMatrix covariance=prod(base,trans(base));
	covariance /= 2*Dimensions;
	
	
	MultiVariateNormalDistributionCholesky dist(covariance);
	Data<RealVector> sampleSet(Samples,RealVector(Dimensions));
	Data<RealVector> normalSampleSet(Samples,RealVector(Dimensions));
	
	for(std::size_t i = 0; i != Samples; ++i){
		MultiVariateNormalDistribution::result_type sample = dist();
		sampleSet.element(i) = sample.first;
		normalSampleSet.element(i) = sample.second;
	}
	
	RealVector meanSampled;
	RealMatrix covarianceSampled;
	RealVector normalMeanSampled;
	RealMatrix normalCovarianceSampled;
	
	meanvar(sampleSet,meanSampled,covarianceSampled);
	meanvar(normalSampleSet,normalMeanSampled,normalCovarianceSampled);
	
	//check that means are correct
	BOOST_CHECK_SMALL(norm_2(meanSampled)/Dimensions,1.e-2);
	BOOST_CHECK_SMALL(norm_2(normalMeanSampled)/Dimensions,1.e-2);
	
	//check that covariances are correct
	BOOST_CHECK_SMALL(norm_frobenius(covarianceSampled-covariance)/sqr(Dimensions),1.e-2);
	BOOST_CHECK_SMALL(
	norm_frobenius(
		normalCovarianceSampled-blas::identity_matrix<double>(Dimensions)
	)/sqr(Dimensions)
	,1.e-2);
	
}
