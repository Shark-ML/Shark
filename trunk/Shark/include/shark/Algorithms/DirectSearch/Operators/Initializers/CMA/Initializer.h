/*!
 * 
 * \file        Initializer.h
 *
 * \brief       Initializer for chromosomes/individuals of the (MO-)CMA-ES.
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_INITIALIZERS_CMA
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_INITIALIZERS_CMA

#include <shark/Core/Exception.h>
//#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

namespace shark {

	namespace elitist_cma {
		/**
		* \brief Initialization operator that initializes individuals and their chromosomes, respectively.
		* \tparam Individual The individual type, needs to provide a method template<unsigned int index> get().
		* \tparam Chromosome Chromosome type, needs to be a model of shark::elitist_cma::Chromosome.
		* \tparam ChromosomeIndex Makes the class consider the chromosome at index ChromosomeIndex.
		*/
		template<typename Individual, typename Chromosome, unsigned int ChromosomeIndex>
		struct Initializer {			

			/**
			* \brief Index of the chromosome considered by this operator.
			*/
			static const unsigned int CHROMOSOME_INDEX = ChromosomeIndex;

			/**
			* \brief Chromosome type.
			*/
			typedef Chromosome chromosome_type;

			/**
			* \brief Individual type.
			*/
			typedef Individual individual_type;

			Initializer() : m_searchSpaceDimension( 0 ),
				m_noObjectives( 0 ),
				m_initialSigma( 0 ),
                                m_isInitialSigmaProvidedByUser( false ),
				m_useNewUpdate( false ),
				m_constrainedFitnessFunction( false ) {
			}
			/**
			* \brief Initializes a \f$(\mu+1)\f$-MO-CMA-ES chromosome.
			* \param [in,out] c The chromosome to be initialized.
			*/
			void operator()( chromosome_type & c ) {

				if( m_searchSpaceDimension == 0 ) 
					throw( SHARKEXCEPTION( "shark::mocma::Initializer: Search space dimension must be greater than zero" ) );
				if( m_noObjectives < 2 )
					throw( SHARKEXCEPTION( "shark::mocma::Initializer: Objective space dimension must be greater or equal to 2." ) );

				c.mep_parent = NULL;
				c.m_lambda = 1;

				c.m_mutationDistribution.resize( m_searchSpaceDimension );

				c.m_evolutionPath.resize( m_searchSpaceDimension );
				c.m_lastStep.resize( m_searchSpaceDimension );

				c.m_stepSize = m_initialSigma;
				c.m_targetSuccessProbability = 1.0 / ( 5.0 + ::sqrt( static_cast<double>( c.m_lambda ) )/2.0 );
				c.m_successProbability = c.m_targetSuccessProbability;			
				c.m_stepSizeDampingFactor = 1.0 + m_searchSpaceDimension / ( 2. * c.m_lambda );

				c.m_stepSizeLearningRate = (c.m_lambda * c.m_targetSuccessProbability) / (2. + c.m_lambda * c.m_targetSuccessProbability );

				c.m_evolutionPathLearningRate = 2.0 / (2.0 + m_searchSpaceDimension);
				c.m_covarianceMatrixLearningRate = 2.0 / (m_searchSpaceDimension * m_searchSpaceDimension + 6.);

				c.m_noSuccessfulOffspring = 0.0;
				c.m_needsCovarianceUpdate = false;
			}

			/**
			* \brief Initializes a \f$(\mu+1)\f$-MO-CMA-ES individual.
			*
			* Integrates with the STL algorithms and allows for initializing
			* whole population:
			* \code
			* Population pop( 101 );
			* // Initialize the first 100 individuals
			* std::transform( pop.begin(), pop.end(), Initializer() );
			* \endcode
			*
			* \param [in,out] individual The chromosome to be initialized.
			*/			
			void operator()( individual_type & individual ) {

				// individual = Individual();
				individual.age() = 0;			
				individual.setNoObjectives( m_noObjectives );
				(*this)( individual.template get< CHROMOSOME_INDEX >() );
			}

			/**
			* \brief Stores/restores the initializer's state.
			* \tparam Archive Type of the archive.
			* \param [in,out] archive The archive to serialize to.
			* \param [in] version Version number, currently unused.
			*/
			template<typename Archive>
			void serialize( Archive & archive, const unsigned int version ) {
				archive & BOOST_SERIALIZATION_NVP( m_searchSpaceDimension );
				archive & BOOST_SERIALIZATION_NVP( m_noObjectives );

				archive & BOOST_SERIALIZATION_NVP( m_lowerBound );
				archive & BOOST_SERIALIZATION_NVP( m_upperBound );

				archive & BOOST_SERIALIZATION_NVP( m_initialSigma );
                                archive & BOOST_SERIALIZATION_NVP( m_isInitialSigmaProvidedByUser );
				archive & BOOST_SERIALIZATION_NVP( m_useNewUpdate );
				archive & BOOST_SERIALIZATION_NVP( m_constrainedFitnessFunction );
			}

			bool operator==( const Initializer & rhs ) const {
				return( 
					m_searchSpaceDimension == rhs.m_searchSpaceDimension &&
					m_noObjectives == rhs.m_noObjectives &&
					shark::blas::norm_2( m_lowerBound - rhs.m_lowerBound ) < 1E-10 &&
					shark::blas::norm_2( m_upperBound - rhs.m_upperBound ) < 1E-10 &&
					m_initialSigma == rhs.m_initialSigma &&
					m_useNewUpdate == rhs.m_useNewUpdate &&
					m_constrainedFitnessFunction == rhs.m_constrainedFitnessFunction
					);
			}

			unsigned int m_searchSpaceDimension;
			unsigned int m_noObjectives;

			RealVector m_lowerBound;
			RealVector m_upperBound;

			double m_initialSigma;
                        bool m_isInitialSigmaProvidedByUser;
			bool m_useNewUpdate;
			bool m_constrainedFitnessFunction;

		};
	}
}

#endif 
