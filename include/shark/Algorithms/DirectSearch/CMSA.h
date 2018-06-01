//===========================================================================
/*!
 * 
 *
 * \brief       Implements the CMSA.
 * 
 * The algorithm is described in
 * 
 * H. G. Beyer, B. Sendhoff (2008). 
 * Covariance Matrix Adaptation Revisited: The CMSA Evolution Strategy 
 * In Proceedings of the Tenth International Conference on Parallel Problem Solving from Nature
 * (PPSN X), pp. 123-132, LNCS, Springer-Verlag
 * 
 * \par Copyright (c) 1998-2008:
 * Institut f&uuml;r Neuroinformatik
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
//===========================================================================


#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_CMSA_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_CMSA_H

#include <shark/Core/DLLSupport.h>
#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>


namespace shark {
/// \brief Implements the CMSA.
///
///  The algorithm is described in
///
///  H. G. Beyer, B. Sendhoff (2008). 
///  Covariance Matrix Adaptation Revisited: The CMSA Evolution Strategy
///  In Proceedings of the Tenth International Conference on Parallel Problem Solving from Nature
///  (PPSN X), pp. 123-132, LNCS, Springer-Verlag
/// \ingroup singledirect
class CMSA : public AbstractSingleObjectiveOptimizer<RealVector > {
	/** \cond */

	struct LightChromosome {
		RealVector step;
		double sigma;
	};
	/** \endcond */
public:

	/// \brief Default c'tor.
	CMSA(random::rng_type& rng = random::globalRng)
	: m_mu( 100 )
	, m_lambda( 200 )
	, m_userSetMu(false)
	,m_userSetLambda(false)
	, m_initSigma(0)
	, mpe_rng(&rng){
		m_features |= REQUIRES_VALUE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CMSA"; }

	SHARK_EXPORT_SYMBOL void read( InArchive & archive );
	SHARK_EXPORT_SYMBOL void write( OutArchive & archive ) const;

	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	
	/// \brief Initializes the algorithm for the supplied objective function.
	SHARK_EXPORT_SYMBOL void init( ObjectiveFunctionType const& function, SearchPointType const& p);
	
	/**
	* \brief Initializes the algorithm for the supplied objective function.
	*/
	SHARK_EXPORT_SYMBOL void init( 
		ObjectiveFunctionType const& function, 
		SearchPointType const& initialSearchPoint,
		std::size_t lambda,
		std::size_t mu,
		double initialSigma,				       
		const boost::optional< RealMatrix > & initialCovarianceMatrix = boost::optional< RealMatrix >()
	);

	/// \brief Executes one iteration of the algorithm.
	SHARK_EXPORT_SYMBOL void step(ObjectiveFunctionType const& function);
	
	/// \brief sets the initial step length sigma
	///
	/// It is by default <=0 which means that sigma =1/sqrt(numVariables)
	void setInitialSigma(double initSigma){
		m_initSigma = initSigma;
	}

	/// \brief Sets the number of selected samples
	void setMu(std::size_t mu){
		m_mu = mu;
		m_userSetMu = true;
	}
	/// \brief Sets the number of sampled points
	void setLambda(std::size_t lambda){
		m_lambda = lambda;
		m_userSetLambda = true;
	}	
	/// \brief Accesses the size of the parent population.
	std::size_t mu() const {
		return m_mu;
	}

	/// \brief Accesses the size of the offspring population.
	std::size_t lambda() const {
		return m_lambda;
	}
	
	RealVector eigenValues()const{
		return sqr(diag(m_mutationDistribution.lowerCholeskyFactor()));
	}
	
	double sigma()const{
		return m_sigma;
	}
protected:
	/// \brief The type of individual used by the CMSA
	typedef Individual< RealVector, double, LightChromosome > IndividualType;

	/// \brief Samples lambda individuals from the search distribution	
	SHARK_EXPORT_SYMBOL std::vector<IndividualType> generateOffspring( ) const;

	/// \brief Updates the strategy parameters based on the supplied offspring population.
	SHARK_EXPORT_SYMBOL void updatePopulation( std::vector< IndividualType > const& offspring );

	/// \brief Initializes the internal data structures of the CMSA
	SHARK_EXPORT_SYMBOL  void doInit(
		std::vector<SearchPointType> const& points,
		std::vector<ResultType> const& functionValues,
		std::size_t lambda,
		std::size_t mu,
		double initialSigma
	);
private:	
	std::size_t m_numberOfVariables; ///< Stores the dimensionality of the search space.
	std::size_t m_mu; ///< The size of the parent population.
	std::size_t m_lambda; ///< The size of the offspring population, needs to be larger than mu.

	bool m_userSetMu; /// <The user set a value via setMu, do not overwrite with default
	bool m_userSetLambda; /// <The user set a value via setMu, do not overwrite with default
	double m_initSigma; ///< The initial step size
	
	double m_sigma; ///< The current step size.
	double m_cSigma; 
	double m_cC; ///< Constant for adapting the covariance matrix.

	RealVector m_mean; ///< The current cog of the population.

	MultiVariateNormalDistributionCholesky m_mutationDistribution; ///< Multi-variate normal mutation distribution.   
	random::rng_type* mpe_rng;
};
}

#endif
