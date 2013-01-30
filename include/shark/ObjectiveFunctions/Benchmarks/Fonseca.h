#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_FONSECA_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_FONSECA_H

//===========================================================================
/*!
*  \brief Bi-objective real-valued benchmark function proposed by Fonseca and Flemming.
*
*  \date 2011
*
*  \par Copyright (c) 2011
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
//===========================================================================

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {

/// \brief Bi-objective real-valued benchmark function proposed by Fonseca and Flemming.
///
/// Fonseca, C. M. and P. J. Fleming (1998). Multiobjective
/// optimization and multiple constraint handling with evolutionary
/// algorithms-Part II: Application example. IEEE Transactions on
/// Systems, Man, and Cybernetics, Part A: Systems and Humans 28(1),
/// 38-47
/// 
/// The default search space dimension is 3, but the function can
/// handle more dimensions.

struct Fonseca : public AbstractObjectiveFunction< VectorSpace<double>, std::vector<double> > {

	Fonseca() : m_numberOfVariables( 3 ) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= IS_CONSTRAINED_FEATURE;
		m_name = "Fonseca";
	}

	unsigned int noObjectives() const {
		return 2 ;
	}

	void init() {
	}

	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		std::vector<double> value( 2 );

		const double d = 1. / ::sqrt( static_cast<double>( x.size() ) );
		double sum1 = 0., sum2 = 0.;
		for( unsigned int i = 0; i < x.size(); i++ ) {
			sum1 += boost::math::pow<2>( x( i ) - d );
			sum2 += boost::math::pow<2>( x( i ) + d );
		}

		value[0] = 1-::exp( -1 * sum1 );
		value[1] = 1-::exp( -1 * sum2 );

		return value;
	}

	void proposeStartingPoint( SearchPointType & x ) const {
		x.resize( m_numberOfVariables );
		for( unsigned int i = 0; i < m_numberOfVariables; i++ )
			x( i ) = Rng::uni( -4., 4. );
	}

	bool isFeasible( const SearchPointType & v ) const {
		for( unsigned int i = 0; i < m_numberOfVariables; i++ ) {
			if( v( i ) < -4 || v( i ) > 4 )
				return false;
		}
		return true;
	}

	void closestFeasible( SearchPointType & v ) const {
		for( unsigned int i = 0; i < numberOfVariables(); i++ ) {
			v( i ) = std::min( v( i ), 4. );
			v( i ) = std::max( v( i ), -4. );
		}
	}
private:
	unsigned int m_numberOfVariables;
};

//template<> struct ObjectiveFunctionTraits<Fonseca> {
//
//	static Fonseca::SearchPointType lowerBounds( unsigned int n ) {
//		return( Fonseca::SearchPointType( n, -4. ) );
//	}
//
//	static Fonseca::SearchPointType upperBounds( unsigned int n ) {
//		return( Fonseca::SearchPointType( n, 4. ) );
//	}
//
//};
}
#endif
