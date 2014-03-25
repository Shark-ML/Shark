//===========================================================================
/*!
 * 
 *
 * \brief       Implements the most recent version of the elitist CMA-ES.
 * 
 * The algorithm is based on
 * 
 * C. Igel, T. Suttorp, and N. Hansen. A Computational Efficient
 * Covariance Matrix Update and a (1+1)-CMA for Evolution
 * Strategies. In Proceedings of the Genetic and Evolutionary
 * Computation Conference (GECCO 2006), pp. 453-460, ACM Press, 2006
 * 
 * D. V. Arnold and N. Hansen: Active covariance matrix adaptation for
 * the (1+1)-CMA-ES. In Proceedings of the Genetic and Evolutionary
 * Computation Conference (GECCO 2010): pp 385-392, ACM Press 2010
 * 
 *
 * \author      O. Krause T.Voss
 * \date        2014
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
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


#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_ELITIST_CMA_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_ELITIST_CMA_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

namespace shark {

/**
* \brief Implements the elitist CMA-ES.
* 
* The algorithm is based on
* 
* C. Igel, T. Suttorp, and N. Hansen. A Computational Efficient
* Covariance Matrix Update and a (1+1)-CMA for Evolution
* Strategies. In Proceedings of the Genetic and Evolutionary
* Computation Conference (GECCO 2006), pp. 453-460, ACM Press, 2006
* 
* D. V. Arnold and N. Hansen: Active covariance matrix adaptation for
* the (1+1)-CMA-ES. In Proceedings of the Genetic and Evolutionary
*/
class ElitistCMA : public AbstractSingleObjectiveOptimizer<VectorSpace<double> >{	    
public:

	ElitistCMA();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ElitistCMA"; }

	void configure( const PropertyTree & node ) {}

	void read( InArchive & archive );

	void write( OutArchive & archive ) const;

	using AbstractSingleObjectiveOptimizer<VectorSpace<double> >::init;
	/**
	* \brief Initializes the algorithm for the supplied objective function.
	*/
	void init( ObjectiveFunctionType const& function, SearchPointType const& p);

	/**
	* \brief Executes one iteration of the algorithm.
	*/
	void step(ObjectiveFunctionType const& function);

	unsigned int lambda() const {
		return m_lambda;
	}

	unsigned int& lambda() {
		return m_lambda;
	}
	
	bool usesActiveUpdate(){
		return m_activeUpdate;
	}
	
	void setActiveUpdate(bool update){
		m_activeUpdate = update;
	}
	
	double sigma()const{
		return m_sigma;
	}

private:
	/**
	* \brief Updates the covariance matrix of the strategy.
	* \param [in] point Coordinates of the individual to update the covariance matrix for.
	*/
	void updateCovarianceMatrix( RealVector const& point );

	/**
	* \brief Purges the information contributed by the individual from the strategy's covariance matrix.
	* \param [in] point Coordinates of the offspring to purge from the strategy's covariance matrix.
	*/
	void activeCovarianceMatrixUpdate( RealVector const& point );

	/**
	* \brief Updates the strategy parameters based on the supplied offspring population.
	*/
	void updateStrategyParameters( RealVector const& point, double fitness );

	unsigned int m_lambda; ///< The size of the offspring population, defaults to one.
	bool m_activeUpdate;///< Should the matrix be updated for non-successful offspring which is better than the previous?

	//current individual
	double m_fitness; ///< fitness of the current individual
	double m_lastFitness;///< last recorded fitness of a previous individual
	unsigned int m_fitnessUpdateFrequency;///<frequency of update of the previous fitness
	unsigned int m_generationCounter;///< Counter of the current generation

	double m_targetSuccessProbability;///<target probability that the best offspring is successfull
	double m_successProbability;///< current probability that an offspring is successfull
	double m_successProbabilityThreshold;///< Threshold for the success probability (TODO: CHECK WHAT THIS DOES)
	RealVector m_mean; ///< point of the current individual, also the mutation mean
	RealVector m_evolutionPathC;///< evolution path of the covariance matrix
	shark::MultiVariateNormalDistribution m_mutationDistribution;///< mutation distribbution around the mean
	double m_sigma;///< current step size of the mutation

	//learning rates
	double m_cSuccessProb; ///< Learning rate for the success probability
	double m_cC; ///< learning rate for the evolution path of the covariance matrix
	double m_cCov;///< learning rate for the covariance matrix
	double m_cCovMinus;///< Learning rate for active unlearning
	double m_cSigma; ///< Larning rate for the step size
	double m_dSigma;
};
}

#endif
