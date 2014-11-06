#define BOOST_TEST_MODULE Trainers_Normalization
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitInterval.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsWhitening.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsZCA.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_Normalization)

BOOST_AUTO_TEST_CASE( NORMALIZE_TO_UNIT_VARIANCE )
{
	std::vector<RealVector> input(3);
	RealVector v(1);
	v(0) = 0.0; input[0] = v;
	v(0) = 1.0; input[1] = v;
	v(0) = 2.0; input[2] = v;
	UnlabeledData<RealVector> set = createDataFromRange(input);
	NormalizeComponentsUnitVariance<> normalizer(true);
	Normalizer<> map;
	normalizer.train(map, set);
	Data<RealVector> transformedSet = map(set);
	double error = std::abs(-std::sqrt(1.5) - transformedSet.element(0)(0)) 
				+ std::abs(transformedSet.element(1)(0)) 
				+ std::abs(sqrt(1.5) - transformedSet.element(2)(0));
	BOOST_CHECK_SMALL(error, 1e-10);
}

BOOST_AUTO_TEST_CASE( NORMALIZE_TO_UNIT_INTERVAL )
{
	std::vector<RealVector> input(3);
	RealVector v(1);
	v(0) = 0.0; input[0] = v;
	v(0) = 1.0; input[1] = v;
	v(0) = 2.0; input[2] = v;
	UnlabeledData<RealVector> set = createDataFromRange(input);
	NormalizeComponentsUnitInterval<> normalizer;
	Normalizer<> map;
	normalizer.train(map, set);
	Data<RealVector> transformedSet = map(set);
	BOOST_CHECK_SMALL(transformedSet.element(0)(0),1.e-10);
	BOOST_CHECK_SMALL(0.5 - transformedSet.element(1)(0),1.e-10);
	BOOST_CHECK_SMALL(1.0 - transformedSet.element(2)(0),1.e-10);
}

BOOST_AUTO_TEST_CASE( NORMALIZE_WHITENING)
{

	RealMatrix mat(3,3);
	mat(0,0)=2;   mat(0,1)=0.1; mat(0,2)=0.3;
	mat(1,0)=0.1; mat(1,1)=5;   mat(1,2)=0.05;
	mat(2,0)=0.3; mat(2,1)=0.05;mat(2,2)=8;
	
	RealVector mean(3);
	mean(0)=1;
	mean(1)=-1;
	mean(2)=3;
	
	MultiVariateNormalDistribution dist(mat);
	
	
	std::vector<RealVector> input(1000,RealVector(3));
	for(std::size_t i = 0; i != 1000;++i)
		input[i]=dist().first+mean;

	UnlabeledData<RealVector> set = createDataFromRange(input);
	NormalizeComponentsWhitening normalizer(1.5);
	LinearModel<> map(3, 3);
	normalizer.train(map, set);
	Data<RealVector> transformedSet = map(set);
	
	RealMatrix covariance;
	meanvar(transformedSet, mean, covariance);
	std::cout<<mean<<" "<<covariance<<std::endl;
	for(std::size_t i = 0; i != 3;++i){
		BOOST_CHECK_SMALL(mean(i),1.e-10);
		for(std::size_t j = 0; j != 3;++j){
			if(j != i){
				BOOST_CHECK_SMALL(covariance(i,j),1.e-5);
			}
			else
				BOOST_CHECK_SMALL(covariance(i,j)-1.5,1.e-5);
		}
	}
}

BOOST_AUTO_TEST_CASE( NORMALIZE_WHITENING_RANK_2)
{
	RealVector v0(3), v1(3);
	v0(0)=1.0; v1(0)= -0.1;
	v0(1)=-0.1; v1(1) = 3.0;
	v0(2)=3.0; v1(2) = 0;
	RealMatrix mat = outer_prod(v0,v0)+outer_prod(v1,v1);
	
	RealVector mean(3);
	mean(0)=1;
	mean(1)=-1;
	mean(2)=3;
	
	MultiVariateNormalDistribution dist(mat);
	
	std::vector<RealVector> input(1000,RealVector(3));
	for(std::size_t i = 0; i != 1000;++i)
		input[i]=dist().first+mean;

	UnlabeledData<RealVector> set = createDataFromRange(input);
	NormalizeComponentsWhitening normalizer(1.5);
	LinearModel<> map(3, 3);
	normalizer.train(map, set);
	Data<RealVector> transformedSet = map(set);
	
	RealMatrix covariance;
	meanvar(transformedSet, mean, covariance);
	std::cout<<mean<<" "<<covariance<<std::endl;
	for(std::size_t i = 0; i != 3;++i){
		BOOST_CHECK_SMALL(mean(i),1.e-10);
		for(std::size_t j = 0; j != 3;++j){
			if(j != i){
				BOOST_CHECK_SMALL(covariance(i,j),1.e-5);
			}
			else if(i != 0)
			{
				BOOST_CHECK_SMALL(covariance(i,j)-1.5,1.e-5);
			}else
				BOOST_CHECK_SMALL(covariance(i,j),1.e-5);
		}
	}
}

BOOST_AUTO_TEST_CASE( NORMALIZE_ZCA)
{

	RealMatrix mat(3,3);
	mat(0,0)=2;   mat(0,1)=0.1; mat(0,2)=0.3;
	mat(1,0)=0.1; mat(1,1)=5;   mat(1,2)=0.05;
	mat(2,0)=0.3; mat(2,1)=0.05;mat(2,2)=8;
	
	RealVector mean(3);
	mean(0)=1;
	mean(1)=-1;
	mean(2)=3;
	
	MultiVariateNormalDistribution dist(mat);
	
	
	std::vector<RealVector> input(1000,RealVector(3));
	for(std::size_t i = 0; i != 1000;++i)
		input[i]=dist().first+mean;

	UnlabeledData<RealVector> set = createDataFromRange(input);
	NormalizeComponentsZCA normalizer(1.5);
	LinearModel<> map(3, 3);
	normalizer.train(map, set);
	Data<RealVector> transformedSet = map(set);
	
	RealMatrix covariance;
	meanvar(transformedSet, mean, covariance);
	std::cout<<mean<<" "<<covariance<<std::endl;
	for(std::size_t i = 0; i != 3;++i){
		BOOST_CHECK_SMALL(mean(i),1.e-10);
		for(std::size_t j = 0; j != 3;++j){
			if(j != i){
				BOOST_CHECK_SMALL(covariance(i,j),1.e-5);
			}
			else
				BOOST_CHECK_SMALL(covariance(i,j)-1.5,1.e-5);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
