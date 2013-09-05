/**
*
*  \brief Determine the volume of the union of objects by an FPRAS
*
*  \author T.Voss
*  \date 2010-2011
*
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
*
*/
#ifndef HYPERVOLUME_APPROXIMATOR_H
#define HYPERVOLUME_APPROXIMATOR_H

#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Algorithms/DirectSearch/ParetoDominanceComparator.h>

#include <shark/LinAlg/Base.h>

namespace shark {

	/**
	* \brief Implements an FPRAS for approximating the volume of a set of high-dimensional objects.
	*
	* See Bringmann, Friedrich: Approximating the volume of unions and intersections of high-dimensional geometric objects, Computational Geometry, Volume 43, 2010, 601-610.
	* and refer to the unit tests for examples:
	* \tparam Rng The type of the random number generator. Please note that the performance of the algorithm is determined by the speed of the RNG.
	*
	*/
	template<typename Rng>
	struct HypervolumeApproximator {

		/**
		* \brief Default error bound
		*/
		static double DEFAULT_EPSILON;

		/**
		* \brief Default error probability
		*/
		static double DEFAULT_DELTA;


		/**
		* \brief Approximates the volume of the union of high-dimensional
		* axis-parallel boxes.
		* 
		* \tparam Iterator Type of the iterator over the range of axis-parallel boxes.
		* \tparam Extractor Extractor type for extracting point information from objects.
		* \param [in] begin Iterator to the beginning of the range of boxes.
		* \param [in] end Iterator pointer after the last valid box.
		* \param [in] e Extractor instance
		* \param [in] referencePoint Minimization is considered and the reference point usually is chosen as the Nadir-point.
		* \param [in] eps Error bound, default value: \f$10^{-2}\f$.
		* \param [in] delta Error probability, default value: \f$10^{-2}\f$.
		*
		* \returns The volume or a negative value if the range of objects is empty.
		*/
		template<typename Iterator, typename Extractor, typename VectorType>
		double operator()( 
			Iterator begin, 
			Iterator end, 
			Extractor e,
			const VectorType & referencePoint,
			double eps = HypervolumeApproximator::DEFAULT_EPSILON,
			double delta = HypervolumeApproximator::DEFAULT_DELTA
		) const {

			std::size_t noPoints = std::distance( begin, end );

			if( noPoints == 0 )
				return( -1 );

			// runtime (O.K: added static_cast to preveent warning on VC10)
			boost::uint_fast64_t maxSamples=static_cast<boost::uint_fast64_t>( 12. * std::log( 1. / delta ) / std::log( 2. ) * noPoints/eps/eps ); 

			// calc separate volume of each box
			VectorType vol( noPoints, 1. );
			typename VectorType::iterator itv = vol.begin();
			for( Iterator it = begin;
				 it != end;
				 ++it, ++itv
				) {
				*itv = referencePoint[ 0 ] - e( *it )[ 0 ];
				for( std::size_t i = 1; i < e( *it ).size(); i++ )
					*itv *= referencePoint[ i ] - e( *it )[ i ];
			}

			// calc total volume and partial sum
			double T = 0;
			for( size_t i = 0; i < noPoints; i++) {
				vol[i]+=T;
				T+=vol[i]-T;
			}

			shark::ParetoDominanceComparator< tag::PenalizedFitness > pdc;

			double r;
			VectorType rndpoint( referencePoint );

			Iterator itt;
			boost::uint_fast64_t samples_sofar=0;
			boost::uint_fast64_t round=0;

			shark::IdentityFitnessExtractor ext;

			while( 1 ) {
				r = T * Rng::uni();

				// point is randomly chosen with probability proportional to volume
				itv = vol.begin();
				for( itt = begin; itt != end; ++itt, ++itv ) {
					if( r <= *itv )
						break;
				}

				// calc rnd point    
				for( std::size_t i = 0; i < rndpoint.size(); i++ )
					rndpoint[ i ] = e( (*itt) )[ i ] + Rng::uni() * ( referencePoint[ i ] - e( (*itt) )[ i ] );

				do {
					if(samples_sofar>=maxSamples) 
						return maxSamples * T / noPoints / round;
					itt = begin + static_cast<std::size_t>(noPoints*Rng::uni());
					samples_sofar++;
				} 
				while( pdc( e( *itt ), rndpoint, ext ) < ParetoDominanceComparator< tag::PenalizedFitness >::A_WEAKLY_DOMINATES_B );

				round++;
			}
		}
	};

	template<typename T>
	double HypervolumeApproximator<T>::DEFAULT_EPSILON = 1E-2;

	template<typename T>
	double HypervolumeApproximator<T>::DEFAULT_DELTA = 0.1;
}

#endif // HYPERVOLUME_APPROXIMATOR_H
