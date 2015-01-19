/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_UNSUPERVISED_RBM_TAGS_H
#define SHARK_UNSUPERVISED_RBM_TAGS_H

#include <shark/Core/Flags.h>

namespace shark{
///\brief Tags are empty types which can be used as a function argument. 
///
///A Tag enables the compiler to automatically choose the correct version of the function based on the tag.
///This is usefull to circumvent writing if-else cascades in multiple functions. Also it prevents the instantiation of unneeded code.
///It also enables the use of compile time errors when certain combination of tags must be prevented.
///This happens for example in the exact computation of the partition fucntion, which can't be evaluated for two real enumeration spaces.
///usage function(argument,SomeType::tag());
///for a function defined as T function(U,tag_type);
namespace tags{
///\brief A Tag for EnumerationSpaces. It tells the Functions, that the space is discrete and can be enumerated.
///
///It does not tell, however, whether it is computationally feasible.
struct DiscreteSpace{};
///\brief A Tag for EnumerationSpaces. It tells the Functions, that the space is real and can't be enumerated.
struct RealSpace{};
}

}

#endif