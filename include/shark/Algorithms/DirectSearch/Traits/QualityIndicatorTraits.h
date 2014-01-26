/*!
 * 
 *
 * \brief       Summarizes common properties of unary and binary quality indicators.
 * 
 * 
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_TRAITS_QUALITY_INDICATOR_TRAITS_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_TRAITS_QUALITY_INDICATOR_TRAITS_H

#include <boost/mpl/void.hpp>

namespace shark {

	namespace tag {

		/** \brief Tags a unary quality indicator. */
		struct Unary {};

		/** \brief Tags a binary quality indicator. */
		struct Binary {};

	}

	/**
	* \brief Abstracts common properties of unary and binary quality indicators.
	*/
	template<typename Indicator>
	struct QualityIndicatorTraits {

		/** \brief Models the arity of the quality indicator. */
		static const int ARITY = 0;

		/** \brief Tags the type (unary or binary) of the quality indicator. */
		typedef boost::mpl::void_ type;
	};

}

/**
* \brief Convenience macro to declare a unary performance indicator.
*/
#define DECLARE_UNARY_QUALITY_INDICATOR( Indicator ) \
	namespace shark {\
		template<>\
		struct QualityIndicatorTraits< Indicator > {\
			static const int ARITY = 1;\
			typedef tag::Unary type;\
		};\
	}\

/**
* \brief Convenience macro to declare a binary performance indicator.
*/
#define DECLARE_BINARY_QUALITY_INDICATOR( Indicator ) \
	namespace shark {\
		template<>\
		struct QualityIndicatorTraits< Indicator > {\
			static const int ARITY = 2;\
			typedef tag::Binary type;\
		};\
	}\

#endif