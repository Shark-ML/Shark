/*!
 * 
 *
 * \brief       Chromosome of the CMA-ES.
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_CMA_CHROMOSOME_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_CMA_CHROMOSOME_H

#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

namespace shark {

	namespace elitist_cma {

		/**
		* \brief Models a chromosome of the elitist (MO-)CMA-ES that encodes strategy parameters.
		*/
		struct Chromosome {

			MultiVariateNormalDistribution m_mutationDistribution; ///< Models the search distribution.

			RealVector m_evolutionPath; ///< Low-pass filtered accumulation of successful mutative steps.
			RealVector m_lastStep; ///< The most recent successful mutative step.

			unsigned int m_lambda; ///< The default number of offspring individuals that are generated from this chromosome, default value: 1.
			double m_noSuccessfulOffspring; ///< The number of successful offspring individuals generated from this chromosome in \f$[0,\lambda]\f$.
			double m_stepSize; ///< The step-size used to scale the normally-distributed mutative steps. Dynamically adapted during the run.
			double m_stepSizeDampingFactor; ///< Damping factor \f$d\f$ used in the step-size update procedure.
			double m_stepSizeLearningRate; ///< The learning rate for the step-size.
			double m_successProbability; ///< Current success probability of this parameter set.
			double m_targetSuccessProbability; ///< Target success probability, close \f$ \frac{1}{5}\f$.
			double m_evolutionPathLearningRate; ///< Learning rate (constant) for updating the evolution path.
			double m_covarianceMatrixLearningRate; ///< Learning rate (constant) for updating the covariance matrix.

			bool m_needsCovarianceUpdate; ///< If true, the covariance matrix of the search distribution needs to be update.

			Chromosome * mep_parent; ///< Points to the parent chromosome that this chromosome has been generated from.

			Chromosome() : m_lambda( 0 ),
				m_noSuccessfulOffspring( 0 ),
				m_stepSize( 0 ),
				m_stepSizeDampingFactor( 0 ),
				m_stepSizeLearningRate( 0 ),
				m_successProbability( 0 ),
				m_targetSuccessProbability( 0 ),
				m_evolutionPathLearningRate( 0 ),
				m_covarianceMatrixLearningRate( 0 ),
				m_needsCovarianceUpdate( false ),
				mep_parent( NULL ) {
			}

			/**
			* \brief Prints the chromosome to the supplied stream.
			* \tparam Stream The type of the stream the chromosome shall be printed to.
			* \param [in,out] out The stream to print to.
			*/
			template<typename Stream>
			void print( Stream & out ) {
				out << "Lambda: " << m_lambda << std::endl;
				out << "NoSuccessfulOffspring: " << m_noSuccessfulOffspring << std::endl;
				out << "StepSize: " << m_stepSize << std::endl;
				out << "StepSizeDampingFactor: " << m_stepSizeDampingFactor << std::endl;
				out << "StepSizeLearningRate: " << m_stepSizeLearningRate << std::endl;
				out << "SuccessProbability: " << m_successProbability << std::endl;
				out << "TargetSuccessProbability: " << m_targetSuccessProbability << std::endl;
				out << "EvolutionPathLearningRate: " << m_evolutionPathLearningRate << std::endl;
				out << "CovarianceMatrixLearningRate: " << m_covarianceMatrixLearningRate << std::endl;

			}

			/**
			* \brief Serializes the chromosome to the supplied archive.
			* \tparam Archive The type of the archive the chromosome shall be serialized to.
			* \param [in,out] archive The archive to serialize to.
			* \param [in] version Version information (optional and not used here).
			*/
			template<typename Archive>
			void serialize( Archive & archive, const unsigned int version ) {

				archive & BOOST_SERIALIZATION_NVP( m_mutationDistribution );

				archive & BOOST_SERIALIZATION_NVP( m_evolutionPath );
				archive & BOOST_SERIALIZATION_NVP( m_lastStep );

				archive & BOOST_SERIALIZATION_NVP( m_lambda );
				archive & BOOST_SERIALIZATION_NVP( m_noSuccessfulOffspring );
				archive & BOOST_SERIALIZATION_NVP( m_stepSize );
				archive & BOOST_SERIALIZATION_NVP( m_stepSizeDampingFactor );
				archive & BOOST_SERIALIZATION_NVP( m_stepSizeLearningRate );
				archive & BOOST_SERIALIZATION_NVP( m_successProbability );
				archive & BOOST_SERIALIZATION_NVP( m_targetSuccessProbability );
				archive & BOOST_SERIALIZATION_NVP( m_evolutionPathLearningRate );
				archive & BOOST_SERIALIZATION_NVP( m_covarianceMatrixLearningRate );

				archive & BOOST_SERIALIZATION_NVP( m_needsCovarianceUpdate );
			}

			bool operator==( const Chromosome & rhs ) const {
				return( 
					m_mutationDistribution == rhs.m_mutationDistribution &&
					shark::blas::norm_2( m_evolutionPath - rhs.m_evolutionPath ) == 0 &&
					shark::blas::norm_2( m_lastStep - rhs.m_lastStep ) == 0 &&
					m_lambda == rhs.m_lambda &&
					m_noSuccessfulOffspring == rhs.m_noSuccessfulOffspring &&
					m_stepSize == rhs.m_stepSize &&
					m_stepSizeDampingFactor == rhs.m_stepSizeDampingFactor &&
					m_stepSizeLearningRate == rhs.m_stepSizeLearningRate &&
					m_successProbability == rhs.m_successProbability &&
					m_targetSuccessProbability == rhs.m_targetSuccessProbability &&
					m_evolutionPathLearningRate == rhs.m_evolutionPathLearningRate &&
					m_covarianceMatrixLearningRate == rhs.m_covarianceMatrixLearningRate &&
					m_needsCovarianceUpdate == rhs.m_needsCovarianceUpdate &&
					mep_parent == rhs.mep_parent
				);
			}
		};
	}

}

#endif 