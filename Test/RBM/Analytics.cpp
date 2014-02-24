#include <shark/Unsupervised/RBM/Energy.h>
#include <shark/Unsupervised/RBM/Neuronlayers/BinaryLayer.h>
#include <shark/Unsupervised/RBM/RBM.h>
#include <shark/Unsupervised/RBM/analytics.h>

#define BOOST_TEST_MODULE RBM_Analytics
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

BOOST_AUTO_TEST_CASE( Energy_Partition_VisibleGreaterHidden )
{
	//all possible state combinations for 2 hidden units
	RealMatrix hiddenStateSpace(4,2);
	hiddenStateSpace(0,0)=0;
	hiddenStateSpace(0,1)=0;
	hiddenStateSpace(1,0)=0;
	hiddenStateSpace(1,1)=1;
	hiddenStateSpace(2,0)=1;
	hiddenStateSpace(2,1)=0;
	hiddenStateSpace(3,0)=1;
	hiddenStateSpace(3,1)=1;
	
	//create RBM with 4 visible and 2 hidden units and initialize it randomly
	RBM<BinaryLayer,BinaryLayer,Rng::rng_type > rbm(Rng::globalRng);
	rbm.setStructure(4,2);
	initRandomNormal(rbm,2);
	
	//now test for several choices of beta
	for(std::size_t i = 0; i <= 10; ++i){
		double beta=i*0.1;
		//calculate the result by integrating over all states using the energies of every state
		double partitionTest = 0;
		for(std::size_t x = 0; x != 16; ++x){
			RealMatrix visibleState(4,4);//we need it 4 times for easier batch processing
			BinarySpace::state(row(visibleState,0),x);
			BinarySpace::state(row(visibleState,1),x);
			BinarySpace::state(row(visibleState,2),x);
			BinarySpace::state(row(visibleState,3),x);
			
		
			partitionTest+=sum(exp(-beta*rbm.energy().energy(hiddenStateSpace,visibleState)));
		}
		//now calculate the test
		double logPartition = logPartitionFunction(rbm,beta);
		
		BOOST_CHECK_CLOSE(logPartition,std::log(partitionTest),1.e-5);
	}
}

BOOST_AUTO_TEST_CASE( Energy_Partition_HiddenGreaterVisible )
{
	//all possible state combinations for 2 visible units
	RealMatrix visibleStateSpace(4,2);
	visibleStateSpace(0,0)=0;
	visibleStateSpace(0,1)=0;
	visibleStateSpace(1,0)=0;
	visibleStateSpace(1,1)=1;
	visibleStateSpace(2,0)=1;
	visibleStateSpace(2,1)=0;
	visibleStateSpace(3,0)=1;
	visibleStateSpace(3,1)=1;
	
	//create RBM with 2 visible and 4 hidden units and initialize it randomly
	RBM<BinaryLayer,BinaryLayer,Rng::rng_type > rbm(Rng::globalRng);
	rbm.setStructure(2,4);
	initRandomNormal(rbm,2);
	
	//now test for several choices of beta
	for(std::size_t i = 0; i <= 10; ++i){
		double beta=i*0.1;
		//calculate the result by integrating over all states using the energies of every state
		double partitionTest = 0;
		for(std::size_t x = 0; x != 16; ++x){
			RealMatrix hiddenState(4,4);//we need it 4 times for easier batch processing
			BinarySpace::state(row(hiddenState,0),x);
			BinarySpace::state(row(hiddenState,1),x);
			BinarySpace::state(row(hiddenState,2),x);
			BinarySpace::state(row(hiddenState,3),x);
			
		
			partitionTest+=sum(exp(-beta*rbm.energy().energy(hiddenState,visibleStateSpace)));
		}
		//now calculate the test
		double logPartition = logPartitionFunction(rbm,beta);
		
		BOOST_CHECK_CLOSE(logPartition,std::log(partitionTest),1.e-5);
	}
}

BOOST_AUTO_TEST_CASE( Energy_NegLogLikelihood )
{
	
	//create RBM with 8 visible and 16 hidden units
	RBM<BinaryLayer,BinaryLayer,Rng::rng_type > rbm(Rng::globalRng);
	rbm.setStructure(8,16);
	
	
	//now test for several random subsets of possible training data and random initializations of the rbm
	for(std::size_t i = 0; i != 10; ++i){
		initRandomNormal(rbm,2);
		std::vector<RealVector> dataVec(50,RealVector(8));
		for(std::size_t j = 0; j != 50; ++j){
			for(std::size_t k = 0; k != 8; ++k){
				dataVec[j](k)=Rng::coinToss(0.5);
			}
		}
		UnlabeledData<RealVector> data = createDataFromRange(dataVec,25);

		//now calculate the test
		long double logPartition = logPartitionFunction(rbm);
		long double logProbTest = 0;
		for(std::size_t j = 0; j != 50; ++j){
			RealMatrix temp(1,8);
			row(temp,0) = dataVec[j];
			logProbTest -= rbm.energy().logUnnormalizedProbabilityVisible(temp,blas::repeat(1,1))(0)-logPartition;
		} 
		long double logProb = negativeLogLikelihood(rbm,data);
		BOOST_CHECK_CLOSE(logProbTest,logProb,1.e-5);
	}
}
