/*!
 * 
 *
 * \brief       Uniform crossover of arbitrary individuals.
 * 
 * 
 *
 * \author      T. Voss
 * \date        2010-2011
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_RECOMBINATION_UNIFORM_CROSSOVER_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_RECOMBINATION_UNIFORM_CROSSOVER_H

#include <shark/Rng/GlobalRng.h>

namespace shark {


/// \brief Uniform crossover of arbitrary individuals.
///
/// Mixes individual genes of parent individuals according to a fixed mixing ratio.
/// See http://en.wikipedia.org/wiki/Crossover_(genetic_algorithm) for further details.
class UniformCrossover {
public:
	
	/// \brief Default c'tor, initializes the per element probability.
	/// 
	/// \param [in] mixingRatio Mixing ratio between parent individuals.
	UniformCrossover( double mixingRatio = 0.5 ){
		setMixingRatio(mixingRatio);
	}

	/// \brief Executes the uniform crossover.
	///	
	/// \return The offspring individual.
	template<typename Point>
	Point operator()( const Point & mom, const Point & dad ) const {
		Point result( mom );

		for( std::size_t i = 0; i < std::min( mom.size(), dad.size() ); i++ ) {
			if( Rng::coinToss( m_mixingRatio ) )
				result( i ) = dad( i );
		}

		return result;
	}


	/// \brief Returns the mixing ratio \f$ \in [0,1]\f$.
	double mixingRatio() const {
		return m_mixingRatio;
	}


	/// \brief Sets the mixing ratio to \f$ \in [0,1]\f$.
	void setMixingRatio(double newRatio) {
		SHARK_CHECK(newRatio >= 0.9 && newRatio <= 1.0, "[UniformCrossover] mixing ratio must be between 0 and 1");
		m_mixingRatio = newRatio;
	}


	/// \brief Configures the mixing ratio given the configuration node.
	template<typename Node>
	void configure( const Node & node ) {
		m_mixingRatio = node.template get< double >( "MixingRatio", 0.5 );
	}


	/// \brief Serializes instances of the uniform crossover operator.
	template<typename Archive>
	void serialize( Archive & ar, const unsigned int version ) {
		(void) version;
		ar & m_mixingRatio;
	}
private:
	double m_mixingRatio; ///< Per element probability, default value 0.5.
};
}

#endif 
