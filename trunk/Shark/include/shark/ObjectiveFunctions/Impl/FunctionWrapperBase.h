/*!
 *  \brief Base class for type erasure in error functions.
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_IMPL_FUNCTIONWRAPPERBASE_H
#define SHARK_OBJECTIVEFUNCTIONS_IMPL_FUNCTIONWRAPPERBASE_H

#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>

namespace shark{

namespace detail{
///\brief Base class for implementations of the Error Function.
template<class InputType,class LabelType>
class FunctionWrapperBase: public SupervisedObjectiveFunction<InputType,LabelType>{
public:
	virtual FunctionWrapperBase<InputType,LabelType>* clone()const = 0;
};
}
}
#endif
