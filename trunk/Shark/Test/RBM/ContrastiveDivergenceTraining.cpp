#define BOOST_TEST_MODULE RBM_ContrastiveDivergenceTraining
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Unsupervised/RBM/BinaryRBM.h>
#include <shark/Unsupervised/RBM/Energy.h>
#include <shark/Unsupervised/RBM/analytics.h>
#include <shark/Unsupervised/RBM/Problems/BarsAndStripes.h>
#include <shark/Algorithms/GradientDescent/SteepestDescent.h>

#include <vector>
#include <fstream>
using namespace shark;
using namespace std;

//calculats the *exact* CD gradient and check that the CD gradient approaches it in the limit
BOOST_AUTO_TEST_CASE( ContrastiveDivergence_ExactGradient)
{
	BarsAndStripes problem;
	UnlabeledData<RealVector> data = problem.data();
	std::size_t inputs = data.numberOfElements();
	
	BinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(16,4);
	BinaryCD cd(&rbm);
	cd.setData(data);
	
	
	
	Energy<BinaryRBM> energy(rbm.energy());
	BinarySpace space;
	std::size_t numHiddenStates = space.numberOfStates(4);
	std::size_t numVisibleStates = space.numberOfStates(16);
	//matrix of all hidden states
	RealMatrix hiddenStates(numHiddenStates,4,0.0);
	for(std::size_t i = 0; i != numHiddenStates; ++i){
		space.state(row(hiddenStates,i),i);
	}

	std::size_t tests = 2;
	//calculate the exact gradient for some random weights of the RBM
	for(std::size_t t = 0; t != tests; ++t){
		initRandomUniform(rbm,-1,1);
		//define the distribution for every input pattern
		RealMatrix hiddenInput(inputs,4);
		
		BinaryLayer::StatisticsBatch hiddenStatistics(inputs,4);
		energy.inputHidden(hiddenInput,data.batch(0));
		rbm.hiddenNeurons().sufficientStatistics(hiddenInput,hiddenStatistics, blas::repeat(1.0,inputs));
		
		//calculate the propability for every hidden state given the visible state p(h|v)
		RealMatrix phv(inputs,numHiddenStates,1.0);
		for(std::size_t i = 0; i != inputs; ++i){
			for(std::size_t j = 0; j != numHiddenStates; ++j){
				for(std::size_t k = 0; k != 4; ++k){//probability of state j given visible state i
					if(hiddenStates(j,k) > 0.5)
						phv(i,j) *= hiddenStatistics(i,k);
					else
						phv(i,j) *= 1.0-hiddenStatistics(i,k);
				}
			}
		}
		
		//marginalize over v -> p_s(h)
		RealVector pSh=sum_rows(phv) / inputs;
		
		//get p(v_i=1|h) for all h
		RealMatrix visibleInput(numHiddenStates,16);
		BinaryLayer::StatisticsBatch visibleStatistics(numHiddenStates,16);
		energy.inputVisible(visibleInput,hiddenStates);
		rbm.visibleNeurons().sufficientStatistics(visibleInput,visibleStatistics, blas::repeat(1.0,numHiddenStates));
		
		//now we calculate for all 2^16 states  sum_h p(v|h)p_s(h)-> p_s(v), this likely makes this test really really slow
		RealVector pSv(numVisibleStates,0.0);
		RealVector state(16);
		for(std::size_t i = 0; i != numVisibleStates; ++i){
			space.state(state,i);
			for(std::size_t j = 0; j != numHiddenStates; ++j){
				double p = 1;//p(v|h)
				for(std::size_t k = 0; k != 16; ++k){
					if(state(k) == 1.0)
						p *= visibleStatistics(j,k);
					else
						p *= 1.0-visibleStatistics(j,k);
				}
				pSv(i) += p*pSh(j);
			}
		}
		
		
		
		//now we compute the gradient
		RealVector visibleGrad(16,0.0);
		RealVector hiddenGrad(4,0.0);
		RealMatrix weightGrad(4,16,0.0);
		
		//again summing over all visible states, this time we compute the weighted gradient
		//we will analytically sum over p(h|v)...talking about slowness, this is likely slower than 
		//the step beforehand ;)
		RealMatrix v(1,16);
		RealMatrix hInput(1,4);
		BinaryLayer::StatisticsBatch hstat(1,4);
		
		for(std::size_t i = 0; i != inputs; ++i){
			row(v,0) = data.element(i);
			energy.inputHidden(hInput,v);
			rbm.hiddenNeurons().sufficientStatistics(hInput,hstat, blas::repeat(1.0,1));
			
			noalias(visibleGrad) -=row(v,0);
			noalias(hiddenGrad) -=row(hstat,0);
			noalias(weightGrad) -=outer_prod(row(hstat,0), row(v,0));
		}
		visibleGrad /= inputs;
		hiddenGrad /= inputs;
		weightGrad /= inputs;
		
		for(std::size_t i = 0; i != numVisibleStates; ++i){
			space.state(row(v,0),i);
			energy.inputHidden(hInput,v);
			rbm.hiddenNeurons().sufficientStatistics(hInput,hstat, blas::repeat(1.0,1));
			
			noalias(visibleGrad) +=pSv(i)*row(v,0);
			noalias(hiddenGrad) +=pSv(i)*row(hstat,0);
			noalias(weightGrad) +=pSv(i)*outer_prod(row(hstat,0), row(v,0));
		}
		
		
		
		RealVector testVisibleGrad(16,0.0);
		RealVector testHiddenGrad(4,0.0);
		RealMatrix testWeightGrad(4,16,0.0);
		
		//compute the real gradient of CD
		RealVector approxCDGrad(rbm.numberOfParameters(),0.0);
		RealVector params = rbm.parameterVector();
		for(std::size_t i = 0; i != 1000; ++i){
			BinaryCD::FirstOrderDerivative der;
			cd.evalDerivative(params,der);
			approxCDGrad+=der;
		}
		approxCDGrad /=1000;
		init(approxCDGrad) >> toVector(testWeightGrad),testHiddenGrad,testVisibleGrad;
		
		double diffH=norm_inf(hiddenGrad-testHiddenGrad);
		double diffV=norm_inf(visibleGrad-testVisibleGrad);
		double diffW=norm_inf(weightGrad-testWeightGrad);
		BOOST_CHECK_SMALL(diffH, 2.e-3);
		BOOST_CHECK_SMALL(diffV, 1.e-2);
		BOOST_CHECK_SMALL(diffW, 3.e-2);
	}
}

BOOST_AUTO_TEST_CASE( ContrastiveDivergenceTraining_Bars ){
	
	unsigned int trials = 1;
	unsigned int steps = 6001;
	unsigned int updateStep = 2000;
	bool save = false;
	std::size_t numUpdates = steps/updateStep+(steps%updateStep > 0); 
	
	std::vector<std::vector<double> > results(trials,std::vector<double>(numUpdates));
	
	BarsAndStripes problem(9);
	UnlabeledData<RealVector> data = problem.data();
	
	Rng::seed(42);
	
	BinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(16,8);
	BinaryCD cd(&rbm);
	cd.setData(data);
	
	for(unsigned int trial = 0; trial != trials; ++trial){
		Rng::seed(42+trial);
		RealVector params(rbm.numberOfParameters());
		for(std::size_t i = 0; i != params.size();++i){
			params(i) = Rng::uni(-0.1,0.1);
		}
		rbm.setParameterVector(params);
		BinaryCD cd(&rbm);
		cd.setData(data);
		SteepestDescent optimizer;
		optimizer.setLearningRate(0.05);
		optimizer.setMomentum(0);
		optimizer.init(cd);
	
		double logLikelyHood = 0;
		for(std::size_t i = 0; i != steps; ++i){
			if(i % updateStep == 0){
				//std::cout<<partitionFunction(rbm);
				rbm.setParameterVector(optimizer.solution().point);
				logLikelyHood = negativeLogLikelihood(rbm,data);
				std::cout<<i<<" "<<logLikelyHood<<std::endl;
				results[trial][i/updateStep] = logLikelyHood;
			}
			optimizer.step(cd);
		}
		BOOST_CHECK( logLikelyHood<400.0 );
	}
	if(save){
		ofstream file("resultNewCD.txt");
		for(std::size_t i = 0; i != numUpdates; ++i){
			file<<i<<" ";
			for(unsigned int trial = 0; trial != trials; ++trial){
				file << results[trial][i]<<" ";
			}
			file<<std::endl;
		}
	}
}
