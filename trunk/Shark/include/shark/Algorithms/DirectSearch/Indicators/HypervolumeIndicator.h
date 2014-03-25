/*!
 * 
 *
 * \brief       Calculates the hypervolume covered by a front of non-dominated points.
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

#include <shark/Core/Exception.h>
#include <shark/Core/OpenMP.h>
#include <shark/Algorithms/DirectSearch/HypervolumeCalculator.h>
#include <shark/LinAlg/Base.h>

#include <algorithm>
#include <vector>

namespace shark {

/**
*  \brief Calculates the hypervolume covered by a front of non-dominated points.
*/
struct HypervolumeIndicator {
public:
	
	/**
	* \brief Calculates the volume of a hyperfront using a reference point
	*
	* \param [in,out] extractor Extractor instance that maps elements of the front to \f$\mathbb{R}^d\f$.
	* \param [in] front pareto front of points to calculate the hypervolume for. 
	* \param [in] referencePoint reference for measuring the hypervolume
	*/
	template<typename Extractor, typename ParetoFrontType, typename VectorType>
	double operator()( Extractor & extractor, ParetoFrontType const& front, VectorType const& referencePoint) {
		
		if(front.empty()) return 0;

		return (m_hv( extractor, front, referencePoint) );
	}

	/**
	* \brief Executes the algorithm and calls to an instance of HypervolumeCalculator.
	*
	* This version uses the reference point estimated by the last call to updateInternals.
	*
	* \param [in] extractor Extractor instance that maps elements of the front to \f$\mathbb{R}^d\f$.
	* \param [in] front front of points to calculate the hypervolume for. 
	*/
	template<typename Extractor, typename ParetoFrontType>
	double operator()( Extractor extractor, ParetoFrontType const& front) {
		return (*this)( extractor, front, m_reference);
	}
		
	/**
	* \brief Determines the individual contributing the least to the front it belongs to.
	*
	* \param [in] extractor Maps the individuals to the objective space.
	* \param [in] front The front of non-dominated individuals.
	* \param [in] referencePoint reference for measuring the hypervolume
	*/
	template<typename Extractor, typename ParetoFrontType, typename VectorType>
	std::size_t leastContributor( Extractor extractor, ParetoFrontType const& front, VectorType const& referencePoint)
	{
		std::vector<double> indicatorValues( front.size() );
		SHARK_PARALLEL_FOR( int i = 0; i < static_cast< int >( front.size() ); i++ ) {

			HypervolumeIndicator ind(*this);

			ParetoFrontType copy( front );
			copy.erase( copy.begin() + i );

			indicatorValues[i] = ind( extractor, copy,referencePoint);
		}

		std::vector<double>::iterator it = std::min_element( indicatorValues.begin(), indicatorValues.end() );

		return std::distance( indicatorValues.begin(), it );
	}
	
	/**
	 * \brief Determines the point contributing the least hypervolume to the overall front of points.
	 *
	 * This version uses the reference point estimated by the last call to updateInternals.
	 * 
	 * \param [in] extractor Extracts point information from front elements.
	 * \param [in] front pareto front of points
	 */
	template<typename Extractor, typename ParetoFrontType>
	std::size_t leastContributor( Extractor extractor, ParetoFrontType const& front)
	{
		return leastContributor(extractor,front,m_reference);
	}
	
	/// \brief Updates the internal variables of the indicator using a whole population.
	///
	/// Calculates the reference point of the volume from the population
	/// using the maximum value in every dimension+1
	/// \param set The set of points.
	template<typename Extractor, typename PointSet>
	void updateInternals(Extractor extractor, PointSet const& set){
		m_reference.clear();
		if(set.empty()) return;
		
		//calculate reference point
		std::size_t noObjectives = extractor(set[0]).size();
		m_reference.resize(noObjectives);
		
		for( unsigned int i = 0; i < set.size(); i++ )
			noalias(m_reference) = max(m_reference, extractor(set[i]));
		
		noalias(m_reference)+=blas::repeat(1.0,noObjectives);
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
	}

	HypervolumeCalculator m_hv;
	RealVector m_reference;
};
}

#endif
