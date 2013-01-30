//===========================================================================
/*!
*  \file AbstractBoxConstraintsProvider.h
*
*  \brief AbstractBoxConstraintsProvider
*
*  \author T.Voss, T. Glasmachers, O.Krause
*  \date 2010-2011
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
//===========================================================================
#ifndef SHARK_CORE_ABSTRACTBOXCONSTRAINTSPROVIDER_H
#define SHARK_CORE_ABSTRACTBOXCONSTRAINTSPROVIDER_H

#include <shark/Core/Traits/ObjectiveFunctionTraits.h>

#include <boost/numeric/interval.hpp>

namespace shark {

	/**
	* \brief Models an entity that lives in a box-bounded space, e.g., an objective function.
	* \tparam PointType Specifies the vector type
	*/
	template<typename PointType>
	class AbstractBoxConstraintsProvider {
	public:
		
		/**
		* \brief For every dimension, the bounds are modeled as an interval.
		*/
		typedef boost::numeric::interval< typename PointType::value_type > IntervalType;	

		/**
		* \brief Virtual d'tor.
		*/
		virtual ~AbstractBoxConstraintsProvider() {}

		/**
		* \brief Access the bounds for the given dimensionality.
		* \param [in] dimension The dimensionality of the space.
		* \returns A vector v of intervals with v.size() <= dimension.
		*/
		virtual std::vector< IntervalType > bounds( std::size_t dimension ) const = 0;

		/**
		* \brief Proposes a starting point based on the bounds provided by the implementation.
		* \param [out] p The result is placed here.
		* \param [in] dimension The target dimension.
		*/
		virtual void proposeStartingPoint( PointType & p, std::size_t dimension ) const = 0;

		/**
		* \brief Checks whether a point lies in the feasible region defined by the bounds.
		* \param [in] v The point to check.
		* \returns True if v lies in the feasible region, false otherwise.
		*/
		virtual bool isFeasible( const PointType & v ) const = 0;



		/**
		* \brief Repairs the supplied point if it is not feasible.
		* \param [in,out] v The point to be repaired.
		* \post isFeasible( v ) returns true.
		*/

		virtual void closestFeasible( PointType & v ) const = 0;
	};

	/**
	* \brief Implements AbstractBoxConstraintsProvider based on the ObjectiveFunctionTraits concept.
	*/
	template<typename PointType, typename Base>
	class TraitsBoxConstraintsProvider : public AbstractBoxConstraintsProvider<PointType> {
	public:
		typedef AbstractBoxConstraintsProvider<PointType> super;
		typedef Base BaseType;

		std::vector< typename super::IntervalType > bounds( std::size_t dimension ) const {
			typename Base::SearchPointType lb, ub;

			lb = ObjectiveFunctionTraits<Base>::lowerBounds( dimension );
			ub = ObjectiveFunctionTraits<Base>::upperBounds( dimension );

			std::vector< typename super::IntervalType > result( lb.size() );

			for( std::size_t i = 0; i < result.size(); i++ )
				result[ i ].set( lb[ i ], ub[ i ] );
			
			return( result );
		}

		void proposeStartingPoint( PointType & x, std::size_t dimension ) const {
			std::vector< typename super::IntervalType > b = bounds( dimension );
			x.resize( dimension );
			for( unsigned int i = 0; i < dimension; i++ )
				x( i ) = Rng::uni( b[ i ].lower(), b[ i ].upper() );
		}

		bool isFeasible( const PointType & v ) const {
			std::vector< typename super::IntervalType > b = bounds( v.size() );

			for( unsigned int i = 0; i < v.size(); i++ ) {
				if( v( i ) < b[ i ].lower() || v( i ) > b[ i ].upper() )
					return false;
			}
			return true;
		}

		void closestFeasible( PointType & v ) const {
			std::vector< typename super::IntervalType > b = bounds( v.size() );

			for( unsigned int i = 0; i < v.size(); i++ ) {
				v( i ) = std::min( v( i ), b[ i ].upper() );
				v( i ) = std::max( v( i ), b[ i ].lower() );
			}
		}
	};
}

#endif // SHARK_CORE_ABSTRACTBOXCONSTRAINTSPROVIDER_H
