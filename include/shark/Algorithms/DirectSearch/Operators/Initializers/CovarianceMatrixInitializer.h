/*!
 * 
 *
 * \brief       Initializer for chromosomes/individuals of the CMA-ES.
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010-2011
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_INITIALIZERS_COVARIANCEMATRIX_INITIALIZER_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_INITIALIZERS_COVARIANCEMATRIX_INITIALIZER_H

namespace shark {
namespace cma {
/**
 *  \brief Initializer for chromosomes/individuals of the CMA-ES.
 */
struct Initializer {

	/**
	 *  \brief Initializer for chromosomes/individuals of the CMA-ES.
	 */
	template<typename Individual>
	void operator()(Individual &individual,
	        unsigned int numberOfVariables,
	        unsigned int lambda,
	        unsigned int mu,
	        double sigma,
	        const boost::optional< RealVector > &variances = boost::optional<RealVector>()
	               ) {


		individual.m_sigma = sigma;

		individual.setDimension(numberOfVariables);
		individual.m_weights.resize(mu);
		switch (individual.m_recombinationType) {
		case 0: {
				for (unsigned int i = 0; i < mu; i++)
					individual.m_weights(i) = 1;
				break;
			}
		case 1/*shark::cma::LINEAR*/: {
				for (unsigned int i = 0; i < mu; i++)
					individual.m_weights(i) = mu-i;
				break;
			}
		case 2/*shark::cma::SUPERLINEAR*/: {
				for (unsigned int i = 0; i < mu; i++)
					individual.m_weights(i) = ::log(mu + 1.) - ::log(1. + i);
				break;
			}
		}

		double sumOfWeights = 0;
		double sumOfSquaredWeights = 0;
		for (unsigned int i = 0; i < individual.m_weights.size(); i++) {
			sumOfWeights += individual.m_weights(i);
			sumOfSquaredWeights += sqr(individual.m_weights(i));
		}
		individual.m_weights /= sumOfWeights;
		sumOfSquaredWeights /= sqr(sumOfWeights);
		individual.m_muEff = 1. / sumOfSquaredWeights;

		// Step size control
		individual.m_cSigma = (individual.m_muEff + 2.)/(numberOfVariables + individual.m_muEff + 3.);
		individual.m_dSigma = 1. + 2. * std::max(0., ::sqrt((individual.m_muEff-1.)/(numberOfVariables+1)) - 1.) + individual.m_cSigma;

		// Covariance matrix adaptation
		individual.m_muCov = individual.m_muEff;
		switch (individual.m_updateType) {
		case 0/*shark::cma::RANK_ONE*/:
			individual.m_muCov = 1;
			break;
		case 1/*shark::cma::RANK_MU*/:
			individual.m_muCov = individual.m_muEff;
			break;
		case 2/*shark::cma::RANK_ONE_AND_MU*/:
			break;
		}

		individual.m_cC = 4. / (4. + numberOfVariables);
		individual.m_cCov = 1. / individual.m_muCov * 2. / sqr(numberOfVariables + ::sqrt(2.)) +
		        (1 - 1. / individual.m_muCov) * std::min(1., (2 * individual.m_muEff - 1) / (sqr(numberOfVariables + 2) + individual.m_muEff));

		individual.m_cCU = ::sqrt((2 - individual.m_cC) * individual.m_cC);
		individual.m_cSigmaU = ::sqrt((2 - individual.m_cSigma) * individual.m_cSigma);

		if (variances) {
			/*for( unsigned int i = 0; i < numberOfVariables; i++ )
			  covarianceMatrix( i, i ) = (*variances)( i );*/

			// TODO: individual.m_mutationDistribution.setCovarianceMatrix( covarianceMatrix );
		}
	}
};
}
}

#endif // SHARK_EA_COVARIANCEMATRIX_INITIALIZER_H
