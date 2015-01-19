/*!
 * 
 *
 * \brief       Base class for type erasure in error functions.
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2014
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_IMPL_FUNCTIONWRAPPERBASE_H
#define SHARK_OBJECTIVEFUNCTIONS_IMPL_FUNCTIONWRAPPERBASE_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>

namespace shark{

namespace detail{
///\brief Base class for implementations of the Error Function.
class FunctionWrapperBase: public SingleObjectiveFunction{
public:
	virtual FunctionWrapperBase* clone()const = 0;
};
}
}
#endif
