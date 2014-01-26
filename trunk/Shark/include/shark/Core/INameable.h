//===========================================================================
/*!
 * 
 *
 * \brief       INameable interface.
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
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

#ifndef SHARK_CORE_INAMEABLE_H
#define SHARK_CORE_INAMEABLE_H

#include <string>

namespace shark {

//! This class is an interface for all objects which can have a name. 
class INameable {
public:
	virtual ~INameable() { }

	///returns the name of the object
	virtual std::string name() const { return "unnamed"; }
};

}

#endif // SHARK_CORE_INAMEABLE_H
