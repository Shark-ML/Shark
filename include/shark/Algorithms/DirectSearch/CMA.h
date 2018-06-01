//===========================================================================
/*!
 * 
 *
 * \brief       Implements the most recent version of the non-elitist CMA-ES.
 * 
 * Hansen, N. The CMA Evolution Startegy: A Tutorial, June 28, 2011
 * and the eqation numbers refer to this publication (retrieved April 2014).
 * 
 *
 * \author      Thomas Voss and Christian Igel
 * \date        April 2014
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


#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_CMA_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_CMA_H

#include <shark/Core/DLLSupport.h>
#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <shark/Algorithms/DirectSearch/Individual.h>


namespace shark {
/// \brief Implements the CMA-ES.
///
///	The algorithm is described in
///
///  Hansen, N., S. Kern (2004). Evaluating the CMA Evolution Strategy
///  on Multimodal Test Functions. In Proceedings of the Eighth
/// International Conference on Parallel Problem Solving from Nature
/// (PPSN VIII), pp. 282-291, LNCS, Springer-Verlag
///
/// For noisy function, noise handling is supported using the
/// noise level detection algorithm described in
/// Hansen, N., et al. "A method for handling uncertainty in evolutionary
/// optimization with an application to feedback control of combustion." 
/// IEEE Transactions on Evolutionary Computation 13.1 (2009): 180-197.
/// Our implementation varies in small details, e.g. instead of the average rank
/// the rank of the average function value is used for updating the strategy parameters
/// which ensures asymptotic unbiasedness. We further do not have an upper bound on
/// the number of reevaluations for the same reason.
/// \ingroup singledirect
class CMA : public AbstractSingleObjectiveOptimizer<RealVector >
{
public:
	/// \brief Models the recombination type.
	enum RecombinationType {
		EQUAL = 0,
		LINEAR = 1,
		SUPERLINEAR = 2
	};

	/// \brief Default c'tor.
	SHARK_EXPORT_SYMBOL CMA(random::rng_type& rng = random::globalRng);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CMA-ES"; }

	/// \brief Calculates lambda for the supplied dimensionality n.
	SHARK_EXPORT_SYMBOL static std::size_t suggestLambda( std::size_t dimension ) ;

	/// \brief Calculates mu for the supplied lambda and the recombination strategy.
	SHARK_EXPORT_SYMBOL static std::size_t suggestMu( std::size_t lambda, RecombinationType recomb = SUPERLINEAR ) ;

	SHARK_EXPORT_SYMBOL void read( InArchive & archive );
	SHARK_EXPORT_SYMBOL void write( OutArchive & archive ) const;

	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	/// \brief Initializes the algorithm for the supplied objective function.
	SHARK_EXPORT_SYMBOL void init( ObjectiveFunctionType const& function, SearchPointType const& p);

	/// \brief Initializes the algorithm for the supplied objective function.
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

	/// \brief Accesses the current step size.
	double sigma() const {
		return m_sigma;
	}

	/// \brief Accesses the current population mean. 
	RealVector const& mean() const {
		return m_mean;
	}

	/// \brief Accesses the current weighting vector.
	RealVector const& weights() const {
		return m_weights;
	}

	/// \brief Accesses the evolution path for the covariance matrix update.
	RealVector const& evolutionPath() const {
		return m_evolutionPathC;
	}

	/// \brief Accesses the evolution path for the step size update.
	RealVector const& evolutionPathSigma() const {
		return m_evolutionPathSigma;
	}

	/// \brief Accesses the covariance matrix of the normal distribution used for generating offspring individuals.
	RealMatrix const& covarianceMatrix() const {
		return m_mutationDistribution.covarianceMatrix();
	}

	/// \brief Accesses the recombination type.
	RecombinationType recombinationType() const {
		return m_recombinationType;
	}

	///\brief Returns a mutable reference to the recombination type.
	RecombinationType & recombinationType() {
		return m_recombinationType;
	}

	///\brief Returns a const reference to the lower bound on sigma times smalles eigenvalue.
	const double & lowerBound() const {
		return m_lowerBound;
	}

	///\brief Set the lower bound on sigma times smalles eigenvalue.
	void setLowerBound(double lowerBound) {
		m_lowerBound = lowerBound;
	}

	/// \brief Returns the size of the parent population \f$\mu\f$.
	std::size_t mu() const {
		return m_mu;
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
	
	///\brief Returns a immutable reference to the size of the offspring population \f$\mu\f$.
	std::size_t lambda()const{
		return m_lambda;
	}

	/// \brief Returns eigenvectors of covariance matrix (not considering step size)
	RealMatrix const& eigenVectors() const {
		return m_mutationDistribution.eigenVectors();
	}

	///\brief Returns a eigenvectors of covariance matrix (not considering step size)
	RealVector const& eigenValues() const {
		return m_mutationDistribution.eigenValues();
	}

	///\brief Returns condition of covariance matrix
	double condition() const {
		RealVector const& eigenValues = m_mutationDistribution.eigenValues();
		return max(eigenValues)/min(eigenValues); 
	}
	
	///\brief Returns how often a point is evaluated 
	std::size_t numberOfEvaluations()const{
		return m_numEvaluations;
	}


protected:
	/// \brief The type of individual used for the CMA
	typedef Individual<RealVector, double, RealVector> IndividualType;
	
	/// \brief Samples lambda individuals from the search distribution	
	SHARK_EXPORT_SYMBOL std::vector<IndividualType> generateOffspring( ) const;

	/// \brief Updates the strategy parameters based on the supplied offspring population.
	SHARK_EXPORT_SYMBOL void updatePopulation( std::vector<IndividualType > const& offspring ) ;

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
	double m_initSigma; ///< use provided initial value of sigma<=0 =>algorithm chooses

	RecombinationType m_recombinationType; ///< Stores the recombination type.

	
	double m_sigma;
	double m_cC; 
	double m_c1; 
	double m_cMu; 
	double m_cSigma;
	double m_dSigma;
	double m_muEff;

	double m_lowerBound;

	RealVector m_mean;
	RealVector m_weights;

	RealVector m_evolutionPathC;
	RealVector m_evolutionPathSigma;

	std::size_t m_counter; ///< counter for generations
	
	std::size_t m_numEvaluations;
	double m_numEvalIncreaseFactor;
	double m_rLambda;
	double m_rankChangeQuantile;

	MultiVariateNormalDistribution m_mutationDistribution;
	random::rng_type* mpe_rng;
};
}

#endif
