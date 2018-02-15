/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_UNSUPERVISED_RBM_IMPL_ANALYTICS_H
#define SHARK_UNSUPERVISED_RBM_IMPL_ANALYTICS_H

#include <shark/Unsupervised/RBM/Tags.h>
#include <shark/Unsupervised/RBM/Sampling/EnergyStoringTemperedMarkovChain.h>
#include <shark/Unsupervised/RBM/Sampling/GibbsOperator.h>
#include <shark/Unsupervised/RBM/Energy.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/Core/OpenMP.h>

#include <boost/range/numeric.hpp>
namespace shark {
namespace detail{
	
	///\brief Computes ln Z_1/Z_0 = ln<e^(-1/2*(U_0-U_1))>_1-ln<e^(-1/2*(U_1-U_0))>_0
	///
	/// This is a lower variance solution of two-sided AIS which uses a 1 instead of 1/2
	template<class V0, class V1>
	double twoSidedAIS(V0 const& energyDiff0, V1 const& energyDiff1){
		SIZE_CHECK(energyDiff0.size() == energyDiff1.size());
		return soft_max(-0.5*energyDiff0)-soft_max(-0.5*energyDiff1);
	}

	///\brief Implements the Acceptance Ratio method invented by Bennett to estimate ln Z_1/Z_0 
	///
	/// It tries to find the number C such that <f(U_0-U_1-C)>_1/<f(U_1-U_0+C)>_0 = 1
	/// where f is the logistic function. C turn out to be ln Z_1/Z_0
	template<class V0, class V1>
	double acceptanceRatio(V0 const& energyDiff0, V1 const& energyDiff1){
		SIZE_CHECK(energyDiff0.size() == energyDiff1.size());
		
		class RatioOptimizationProblem : public SingleObjectiveFunction{
		private:
			V0 const& energyDiff0;
			V1 const& energyDiff1;
		public:
			RatioOptimizationProblem(V0 const& v0, V1 const& v1)
			:energyDiff0(v0),energyDiff1(v1){
				m_features |= HAS_FIRST_DERIVATIVE;
			}
			std::string name() const{return "";}

			std::size_t numberOfVariables()const{
				return 1;
			}

			double eval( const SearchPointType & C ) const {
				//~ std::size_t n = energyDiff0.size();
				return 0.5*sqr(sum(sigmoid(energyDiff0 + C(0)))
					- sum(sigmoid(energyDiff1 - C(0))));
				
				//~ return sum(softPlus(energyDiff0 + C(0)))
					//~ + sum(softPlus(energyDiff1- C(0)));
			}

			ResultType evalDerivative( const SearchPointType & C, FirstOrderDerivative & derivative )const {
				derivative.resize(1);
				//~ std::size_t n = energyDiff0.size();
				RealVector sigmoid0 = sigmoid(energyDiff0+ C(0));
				RealVector sigmoid1 = sigmoid(energyDiff1- C(0));
				
				double diff = sum(sigmoid0) - sum(sigmoid1);
				
				derivative(0) = sum(sigmoid0*(1.0-sigmoid0))+sum(sigmoid1*(1.0-sigmoid1));
				derivative*=diff;
				return 0.5*sqr(diff);
				
				//~ derivative(0) = sum(sigmoid(energyDiff0 + C(0)))
					//~ - sum(sigmoid(energyDiff1 - C(0)));
				
				//~ return eval(C);

			}
			
		};
		RatioOptimizationProblem f(energyDiff0,energyDiff1);
		//initialize with solution of AIS-PT
		RealVector C(1,soft_max(-energyDiff0)-std::log(double(energyDiff0.size())));
		Rprop<> optimizer;
		optimizer.init(f,C);
		while(optimizer.solution().value > 1.e-10){
			optimizer.step(f);
		}
		
		//~ if(std::abs(C(0)) > 100){
			//~ RealVector one(1,1);
			//~ RealVector C0(1,soft_max(-energyDiff0)-std::log(double(energyDiff0.size())));
			//~ std::cout<<" "<<C(0)<<" "<<C0(0)<<" "<<devTest<<" "<<f(-3*one)<<" "<<f(0*one)<<" "<<f(3*one)<<" "<<f(6*one)<<" "<<std::endl;
		//~ }
		return optimizer.solution().point(0);
	}

	/// \brief Samples energy difference between different chains from an RBM.
	///
	/// Given a set of beta: beta_0> beta_1>.... beta_n >= 0, draws samples from the RBM with the corresponding beta values using 
	/// Parallel tempering. Forevery such sample the difference in Energy with respect to the neighboringchains is computed. The Difference
	/// when comparing tothe upper chain is energyDiffUp and when comparing to the lower chain is energyDiffLow.
	/// For variance reasons the energy of a sample (h,v) is in this case defined as the negative logarithm of the probability of h,
	/// \f$ -\ln(p_n(h)) = -\ln \int  e^{-\beta_n E(v,h)}\,dv \f$. That means, we integrate over v. Thus energyDiffUp(i) is defined as
	/// energyDiffUp_i =\ln(p_i(h)) - \ln(p_{i+1}(h)) and energyDiffDown_i = \ln(p_i(h)) - \ln(p_{i-1}(h)).
	/// this statistic is gathered for every sample. To have an initial starting state a dataset has to be supplied which offers "close" samples.
	///
	/// \param rbm The rbm for which to compute the energies.
	/// \param initDataset dataset from which samples are picked to initialize the chains
	/// \param beta the inverse temperatures of the chains
	/// \param energyDiffUp energy difference for a state when it is moved to the next higher chain (higher beta, lower temperature)
	/// \param energyDiffDown energyDifference for a state when it is moved to the next lower chain(lower beta, higher temperature).
	template<class RBMType>
	void sampleEnergies(
		RBMType& rbm, 
		Data<RealVector> const& initDataset, 
		RealVector const& beta, 
		RealMatrix& energyDiffUp,
		RealMatrix& energyDiffDown,
		double burnInPercentage
	){
		std::size_t chains = beta.size();
		std::size_t samples = energyDiffUp.size2();
		
		std::size_t burnIn = static_cast<std::size_t>(samples*burnInPercentage);
		if(!burnIn)++burnIn;
		
		//setup sampler
		typedef EnergyStoringTemperedMarkovChain<GibbsOperator<RBMType> > PTSampler;
		PTSampler sampler(&rbm);
		sampler.setNumberOfTemperatures(chains);
		for(std::size_t i = 0; i != chains; ++i){
			sampler.setBeta(i,beta(i));
		}
		sampler.storeEnergyDifferences() = false;
		sampler.step(burnIn);
		sampler.storeEnergyDifferences() = true;
		//acquire the samples
		for(std::size_t s = 0; s != samples; ++s){
			sampler.step(1);
		}
		
		noalias(energyDiffUp) = sampler.getUpDifferences();
		noalias(energyDiffDown) = sampler.getDownDifferences();
	}
	
	template<class RBMType>
	void sampleEnergiesWithTempering(
		RBMType& rbm, 
		RealVector const& beta, 
		RealMatrix& energyDiffUp
	){
		std::size_t chains = beta.size();
		std::size_t samples = energyDiffUp.size2();
		
		//setup sampler
		GibbsOperator<RBMType> gibbsOperator(&rbm);
		typedef typename  GibbsOperator<RBMType>::HiddenSampleBatch Hidden;
		typedef typename  GibbsOperator<RBMType>::VisibleSampleBatch Visible;
		
		//sample and store Energies batchwise
		std::size_t batchSize  = 1024;
		std::size_t numBatches = samples/batchSize;
		if(numBatches*batchSize < samples)
			++numBatches;
		
		SHARK_PARALLEL_FOR (unsigned int b = 0; b < (unsigned int)numBatches; ++b){
			std::size_t batchStart = b*batchSize;
			std::size_t batchEnd = (b== numBatches-1)? samples : batchStart+batchSize;
			std::size_t curSize = batchEnd-batchStart;

			Energy<RBMType> energy = rbm.energy();
			
			Hidden hidden(curSize,rbm.numberOfHN());
			Visible visible(curSize,rbm.numberOfVN());
			//lowest beta must create independent samples, thus we don't need to initialize batches
			
			//westart from the lowest beta (usually 0) and sample up to beta(1)
			//we don't sample beta(0) as we can't generate an energy difference
			for(std::size_t i  = beta.size()-1; i >0; --i){
				//sample at current temperature
				gibbsOperator.precomputeHidden(hidden, visible,blas::repeat(beta(i),curSize));
				gibbsOperator.sampleHidden(hidden);
				gibbsOperator.precomputeVisible(hidden, visible,blas::repeat(beta(i),curSize));
				gibbsOperator.sampleVisible(visible);
				
				
				//calculate The upper energy difference for every chain.
				noalias(subrange(row(energyDiffUp,i),batchStart,batchEnd)) = 
				energy.logUnnormalizedProbabilityHidden(
					hidden.state,
					visible.input,
					blas::repeat(beta(i),curSize)
				) 
				- energy.logUnnormalizedProbabilityHidden(
					hidden.state,
					visible.input,
					blas::repeat(beta(i-1),curSize)
				);
			}
		}
	}
	

	///\brief updates the log partition with the Energy of another state
	///
	///Calculating the partition fucntion itself is not easy. Aside from the computational complexity, 
	///the partition can easily exceed the maximum value of double. So instead we want to calculate
	///the log partition, which is computationally a bit more complex, but feasible.
	///Let's assume we have n Energies E_1,..., E_n for which we allready computed the partial partition
	/// Z_n = exp(-E_1)+...exp(-E_n)
	///given another Energy E we can update this Z_n to Z by
	/// Z = Z_n+exp(-E) = exp(-E)*(Z_n*exp(E)+1)
	/// and for the logarithmic result it holds: 
	/// log Z = -E + log(1+exp( log(Z_n)+E))=-E +softPlus(log(Z_n)+E) 
	/// or equivalently log Z = log(Z_n) + log(1+exp(-log(Z_n)-E)=-log(Z_n) +softPlus(-log(Z_n)-E) 
	/// which is numerically stable to compute since softPlus(x)->x when x>>0 and softPlus(x)-> 0 when x<<0
	///however if these edge cases arise, the result will be in this case be max(log Z, -E) since the values
	/// are not comparable anymore on the scale of double.
	inline double updateLogPartition(double logZn, double E){
		if(logZn == -std::numeric_limits<double>::infinity()){
			return -E;
		}
		double diff = logZn + E;// diff between logZn and -E
		//~ if(diff >= maxExpInput<double>()|| diff <= minExpInput<double>()){
			//~ return std::max(logZn, -E);
		//~ }
		return logZn + softPlus(-diff);
	}

	/// \brief Estimates the partition function with factorization over the hidden variables. 
	///
	/// Instead of summing over the unnormalized joint probability of all states of hidden and visible variables 
	/// this function sums over the unnormalized marginal probability of all states of the visible variables,
	/// which is calculated via factorization over the hidden variables. 
	///
	/// Enumeration is the state space of the hidden variables.
	///
	/// @param rbm the RBM
	/// @param beta the inverse Temperature of the RBM
	/// @return the partition function
	template<class RBMType, class Enumeration>
	double logPartitionFunctionImplFactHidden(const RBMType& rbm, Enumeration, double beta){
		std::size_t values = Enumeration::numberOfStates(rbm.numberOfVN());
		std::size_t const batchSize=4096;
		
		std::size_t batchNum = values/batchSize;
		if(values % batchSize > 0)
			++batchNum;
		//over all possible values of the visible neurons
		RealVector logZ(std::min(batchNum, SHARK_NUM_THREADS),-std::numeric_limits<double>::infinity());
		SHARK_PARALLEL_FOR(int x = 0;x <  (int)values; x+= (int)batchSize) {
			std::size_t currentBatchSize=std::min<std::size_t>(batchSize,values-static_cast<std::size_t>(x));
			RealMatrix stateMatrix(currentBatchSize,rbm.numberOfVN());
			
			for(std::size_t elem = 0; elem != currentBatchSize;++elem){
				//generation of the x+elem-th state vector
				Enumeration::state(row(stateMatrix,elem),x+elem);
			}
			RealVector p =rbm.energy().logUnnormalizedProbabilityVisible(
				stateMatrix, blas::repeat(beta,currentBatchSize)
			);
			//accumulate changes to the log partition
			logZ(SHARK_THREAD_NUM) = boost::accumulate(
				-p,
				logZ(SHARK_THREAD_NUM),updateLogPartition
			);
		}
		return boost::accumulate(
			-logZ,
			-std::numeric_limits<double>::infinity(),
			updateLogPartition
		);
	}
	
	
	/// \brief Estimates the partition function with factorization over the hidden variables. 
	///
	/// Instead of summing over the unnormalized joint probability of all states of hidden and visible variables 
	/// this function sums over the unnormalized marginal probability of all states of the hidden variables,
	/// which is calculated via factorization over the visible variables. 
	///
	/// Enumeration is the state space of the hidden variables.
	///
	/// @param rbm the RBM
	/// @param beta the inverse Temperature of the RBM
	/// @return the partition function
	/// @return the partition function
	template<class RBMType, class Enumeration>
	double logPartitionFunctionImplFactVisible(const RBMType& rbm, Enumeration, double beta){		
		std::size_t values = Enumeration::numberOfStates(rbm.numberOfHN());
		std::size_t const batchSize=4096;
		
		std::size_t batchNum = values/batchSize;
		if(values % batchSize > 0)
			++batchNum;
		//over all possible values of the visible neurons
		RealVector logZ(std::min(batchNum, SHARK_NUM_THREADS),-std::numeric_limits<double>::infinity());
		SHARK_PARALLEL_FOR(int x = 0;x <  (int)values; x+= (int)batchSize) {
			std::size_t currentBatchSize=std::min<std::size_t>(batchSize,values-static_cast<std::size_t>(x));
			RealMatrix stateMatrix(currentBatchSize,rbm.numberOfHN());
			
			
			for(std::size_t elem = 0; elem != currentBatchSize; ++elem){
				//generation of the x-th state vector
				Enumeration::state(row(stateMatrix,elem),x+elem);
			}
			
			RealVector p=rbm.energy().logUnnormalizedProbabilityHidden(
				stateMatrix, blas::repeat(beta,currentBatchSize)
			);
			
			//accumulate changes to the log partition
			logZ(SHARK_THREAD_NUM) = boost::accumulate(
				-p,
				logZ(SHARK_THREAD_NUM),updateLogPartition
			);
		}
		return boost::accumulate(
			-logZ,
			-std::numeric_limits<double>::infinity(),
			updateLogPartition
		);
	}
	
	//===========Warning=========
	//Beyond this line starts scary template boilerplate!
	
	//in logPartitionFunction we dispatch the function call into one of the four following versions.
	//the dispatching is done with respect to the type of state space of the distribution.
	//the exact partition can only be comptued when one of both state spaces is discrete. So if only one of	
	//both state spaces is discrete it is obvious over which variables to integrate. In the case of two discrete
	// stae spaces it is integrated over the space with less neurons. if both are continuous, a compile error is 
	//generated.
	
	template<class RBMType>
	double logPartitionFunction(
		const RBMType& rbm, 
		tags::DiscreteSpace, 
		tags::RealSpace,
		double beta
	){
		// Since we want to factorize over the hidden neurons, we sum over the marginal distribution of
		// the visible and thus have to acquire the visible enumeration type.
		typedef typename RBMType::VisibleType::StateSpace Enumeration;
		
		return logPartitionFunctionImplFactHidden(rbm,Enumeration(),beta);
	}
	
	template<class RBMType>
	double logPartitionFunction(
		const RBMType& rbm, 
		tags::RealSpace,
		tags::DiscreteSpace,
		double beta
	){
		// Since we want to factorize over the visible neurons, we sum over the marginal distribution of
		// the hidden and thus have to acquire the hidden enumeration type.
		typedef typename RBMType::HiddenType::StateSpace Enumeration;
		
		return logPartitionFunctionImplFactVisible(rbm,Enumeration(),beta);
	}
	
	template<class RBMType>
	double logPartitionFunction(
		const RBMType& rbm, 
		tags::DiscreteSpace, 
		tags::DiscreteSpace,
		double beta
	){
		//get both enumeration types and check the number of states
		typedef typename RBMType::HiddenType::StateSpace EnumerationHidden;
		typedef typename RBMType::VisibleType::StateSpace EnumerationVisible;

		if(rbm.numberOfHN() < rbm.numberOfVN()){
			return logPartitionFunctionImplFactVisible(rbm,EnumerationHidden(),beta);
		}
		return logPartitionFunctionImplFactHidden(rbm,EnumerationVisible(),beta);
	}
	
	template<class RBMType>
	double logPartitionFunction(
		const RBMType& rbm, 
		tags::RealSpace, 
		tags::RealSpace,
		double beta
	){
		static_assert(sizeof(RBMType), "Can't calculate the partition of two real valued layers!");
		return 0;
	}
}
}
#endif
