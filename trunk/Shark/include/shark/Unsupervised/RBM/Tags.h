/*!
 * 
 * \file        Tags.h
 *
 * \brief       -
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

///\brief Possible values a Sampler might need to store.
enum SamplingFlagTypes{
	StoreHiddenStatistics = 1,
	StoreHiddenInput = 2,
	StoreHiddenState = 4,
	StoreHiddenFeatures = 8,
	StoreVisibleStatistics = 16,
	StoreVisibleInput = 32,
	StoreVisibleState = 64,
	StoreVisibleFeatures = 128,
	StoreEnergyComponents = 256
};

///\brief Values a gradient may require.
enum GradientFlagTypes{
	RequiresStatistics = 1,
	RequiresInput = 2,
	RequiresState = 4 ,
	RequiresFeatures = 8
};

typedef TypedFlags<SamplingFlagTypes> SamplingFlags;
typedef TypedFlags<GradientFlagTypes> GradientFlags;

///\brief Transforms a set of visible and hidden gradient flags to a list of general sampling flags
inline TypedFlags<SamplingFlagTypes> convertToSamplingFlags(TypedFlags<GradientFlagTypes> hiddenFlags, TypedFlags<GradientFlagTypes> visibleFlags){
	TypedFlags<SamplingFlagTypes> resultFlags;
	//hidden
	if(hiddenFlags & RequiresStatistics){
		resultFlags |= StoreHiddenStatistics;
	}
	if(hiddenFlags & RequiresInput){
		resultFlags |= StoreHiddenInput;
	}
	if(hiddenFlags & RequiresState){
		resultFlags |= StoreHiddenState;
	}
	if(hiddenFlags & RequiresFeatures){
		resultFlags |= StoreVisibleFeatures;
	}
	//visible
	if(visibleFlags & RequiresStatistics){
		resultFlags |= StoreVisibleStatistics;
	}
	if(visibleFlags & RequiresInput){
		resultFlags |= StoreVisibleInput;
	}
	if(visibleFlags & RequiresState){
		resultFlags |= StoreVisibleState;
	}
	if(visibleFlags & RequiresFeatures){
		resultFlags |= StoreHiddenFeatures;
	}
	return resultFlags;
}


}

#endif