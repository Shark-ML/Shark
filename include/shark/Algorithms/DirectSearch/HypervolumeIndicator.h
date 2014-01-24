/*!
 * 
 * \file        HypervolumeIndicator.h
 *
 * \brief       Calculates the hypervolume covered by a set of non-dominated points.
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMEINDICATOR_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMEINDICATOR_H

#include <shark/Algorithms/DirectSearch/HypervolumeCalculator.h>
#include <shark/Algorithms/DirectSearch/Traits/QualityIndicatorTraits.h>

#include <algorithm>
#include <vector>

namespace shark {

	/**
	*  \brief Calculates the hypervolume covered by a set of non-dominated points.
	*/
	struct HypervolumeIndicator {
	public:

		/**
		* \brief Executes the algorithm and calls to an instance of HypervolumeCalculator.
		* \pre Both the nadir fitness and the utopian fitness vectors need to be set.
		* \param [in,out] extractor Extractor instance that maps elements of the set to \f$\mathbb{R}^d\f$.
		* \param [in] set Set of points to calculate the hypervolume for. 
		* \param [in] noObjectives Defines the dimensioniality d.
		*/
		template<typename Extractor, typename Set>
		double operator()( Extractor & extractor, const Set & set , unsigned int noObjectives ) {
			
			if( m_nadirFitness.size() != m_utopianFitness.size() )
				throw shark::Exception( "HypervolumeIndicator: Dimension of utopian and nadir fitness vectors do not match.", __FILE__, __LINE__ );
			if( m_nadirFitness.size() < noObjectives )
				throw shark::Exception( "HypervolumeIndicator: Dimension of utopian and nadir fitness vectors are smaller than required by parameter noObjectives.", __FILE__, __LINE__ );

			RealVector ref( m_nadirFitness );
			for( unsigned int i = 0; i < ref.size(); i++ )
				ref[i] += m_nadirFitness[i] - m_utopianFitness[i];

			return( m_hv( extractor, set, ref, noObjectives ) );
		}

		/**
		* \brief Adjusts the nadir fitness vector.
		* \param [in] fitness The new nadir fitness vector.
		*/
		void setNadirFitness( const RealVector & fitness ) {
			m_nadirFitness = fitness;
		}

		/**
		* \brief Adjusts the utopian fitness vector.
		* \param [in] fitness The new utopian fitness vector.
		*/
		void setUtopianFitness( const RealVector & fitness ) {
			m_utopianFitness = fitness;
		}

		/**
		* \brief Serializes/Deserializes the state of the indicator to the supplied archive.
		* \tparam Archive Archive type, needs to be a model of a boost::serialization archive.
		* \param [in,out] archive Archive to store to/load from.
		* \param [in] version Currently unused.
		*/
		template<typename Archive>
		void serialize( Archive & archive, const unsigned int version ) {
			archive & BOOST_SERIALIZATION_NVP( m_hv );
			archive & BOOST_SERIALIZATION_NVP( m_nadirFitness );
			archive & BOOST_SERIALIZATION_NVP( m_utopianFitness );
		}


		HypervolumeCalculator m_hv;
		RealVector m_nadirFitness;
		RealVector m_utopianFitness;
	};
}

DECLARE_UNARY_QUALITY_INDICATOR( shark::HypervolumeIndicator );

#endif
