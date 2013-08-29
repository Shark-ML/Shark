/*
*  \par Copyright (c) 1998-2007:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
*/
#ifndef SHARK_UNSUPERVISED_RBM_IMPL_ANALYTICS_H
#define SHARK_UNSUPERVISED_RBM_IMPL_ANALYTICS_H

#include <shark/Unsupervised/RBM/Tags.h>
#include <shark/LinAlg/Base.h>
#include <shark/Data/BatchInterface.h>
#include <boost/static_assert.hpp>
#include <boost/range/numeric.hpp>
namespace shark {
namespace detail{

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
		if(diff >= maxExpInput<double>()|| diff <= minExpInput<double>()){
			return std::max(logZn, -E);
		}
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
		std::size_t batchSize = std::min(values, std::size_t(500));
		
		//over all possible values of the visible neurons
		RealMatrix stateMatrix(batchSize,rbm.numberOfVN());
		
		//double c = 0;
		double logZ = -std::numeric_limits<double>::infinity();
		for (std::size_t x = 0; x < values; x+=batchSize) {
			std::size_t currentBatchSize=std::min(batchSize,values-x);
			stateMatrix.resize(currentBatchSize,rbm.numberOfVN());
			
			for(std::size_t elem = 0; elem != currentBatchSize;++elem){
				//generation of the x+elem-th state vector
				Enumeration::state(row(stateMatrix,elem),x+elem);
			}
	
			//accumulate changes to the log partition
			logZ = boost::accumulate(
				-rbm.energy().logUnnormalizedPropabilityVisible(
					stateMatrix, blas::repeat(beta,currentBatchSize)
				),
				logZ,updateLogPartition
			);
		}
		return logZ;
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
		std::size_t batchSize=std::min(values,std::size_t(500));
		
		//over all possible values of the visible neurons
		RealMatrix stateMatrix(batchSize,rbm.numberOfVN());
		
		//double c = 0;
		double logZ = -std::numeric_limits<double>::infinity();
		for (std::size_t x = 0; x < values; x+=batchSize) {
			std::size_t currentBatchSize=std::min(batchSize,values-x);
			stateMatrix.resize(currentBatchSize,rbm.numberOfHN());
			
			for(std::size_t elem = 0; elem != currentBatchSize; ++elem){
				//generation of the x-th state vector
				Enumeration::state(row(stateMatrix,elem),x+elem);
			}
			
			//accumulate changes to the log partition
			logZ = boost::accumulate(
				-rbm.energy().logUnnormalizedPropabilityHidden(
					stateMatrix, blas::repeat(beta,currentBatchSize)
				),
				logZ,updateLogPartition
			);
		}
		return logZ;
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
		BOOST_STATIC_ASSERT(sizeof(RBMType) && "Can't calculate the partition of two real valued layers!");
		return 0;
	}
}
}
#endif
