/*!
 * 
 *
 * \brief       Implements the default variation operator of the multi-objective
 * covariance matrix ES.

 * 
 *
 * \author      T.Voss
 * \date        2010
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_MUTATION_CMA_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_MUTATION_CMA_H

namespace shark {

namespace cma {
/**
* \brief Implements the mutation operator of the (MO)-CMA.
*/
template<typename Individual, typename Chromosome, unsigned int ChromosomeIndex>
struct Variator {

	/**
	* \brief Chromosome type.
	*/
	typedef Chromosome chromosome_type;

	/**
	* \brief Individual type.
	*/
	typedef Individual individual_type;

	/**
	* \brief Mutates the supplied individual.
	*
	* Mutates the supplied individual according to:
	* \f[
	* \vec{x}' \leftarrow \mathcal{N}( \vec{x}, \sigma^2 \vec{C} )
	* \f]
	* where \f$\vec{x}, \sigma, \vec{C}\f$ refer to the individuals search point, 
	* its step-size and its covariance matrix, respectively.
	* \param [in,out] ind The individual to be mutated.
	*/
	void operator()( individual_type & ind ) {
		std::pair<
			typename individual_type::search_point_type,
			typename individual_type::search_point_type
		> result = (*this)( ind.template get< ChromosomeIndex >() );
		
		*ind += result.second;
	}

	std::pair<
		typename individual_type::search_point_type,
		typename individual_type::search_point_type
	> operator()( chromosome_type & c ) {
		std::pair<
			typename individual_type::search_point_type,
			typename individual_type::search_point_type
		> result;
		MultiVariateNormalDistribution::ResultType sample = c.m_mutationDistribution();
		result.first = sample.first;
		result.second = c.m_stepSize * sample.first;
		c.m_lastStep = result.first;
		c.m_needsCovarianceUpdate = true;
		return result;
	}


	/**
	* \brief Serialization/deserialization, implemented empty.
	*/
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
		(void) archive;
		(void) version;
	}

};
}
}
#endif 