//===========================================================================
/*!
 * 
 *
 * \brief       Compares two individuals w.r.t. their level of non-dominance and 
 * w.r.t. the share they contribute to the front both of them belong to.
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
//===========================================================================
#ifndef RANKSHARECOMPARATOR_H
#define RANKSHARECOMPARATOR_H

namespace shark {
	/**
	*  \brief Compares two individuals w.r.t. their level of non-dominance and 
	*  w.r.t. the share they contribute to the front both of them belong to.
	*/
	struct RankShareComparator {

		/**
		* \brief Carries out the actual comparison.
		*
		*/
		template<typename IndividualType>
		bool operator()( const IndividualType & i1, const IndividualType & i2 ) {
			return( 
				i1.rank() < i2.rank() || 
				(
					i1.rank() == i2.rank() && 
					i1.share() > i2.share() 
				) 
			);
		}
	};
}

#endif // RANKSHARECOMPARATOR_H
