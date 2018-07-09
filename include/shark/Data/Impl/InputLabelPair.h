/*!
 *
 * \brief Type for representing pairs of inputs and labels
 *
 *  \author O. Krause
 *  \date 2012
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_DATA_IMPL_INPUTLABELPAIR_H
#define SHARK_DATA_IMPL_INPUTLABELPAIR_H

#include <shark/Data/BatchInterfaceAdaptStruct.h>

namespace shark{
///\brief Input-Label pair of data
template<class InputType,class LabelType>
struct InputLabelPair{
	InputType input;
	LabelType label;
	
	InputLabelPair(){}

	template<class I, class L>
	InputLabelPair(
		I&& input,
		L&& label
	):input(input),label(label){}
	
	InputLabelPair(
		InputType const& input,
		LabelType const& label
	):input(input),label(label){}
	
	template<class InputT, class LabelT>
	InputLabelPair(
		InputLabelPair<InputT,LabelT> const& pair
	):input(pair.input),label(pair.label){}
	
	InputLabelPair& operator=(
		InputLabelPair const& pair
	){
		input = pair.input;
		label = pair.label;
		return *this;
	}
	
	template<class InputT, class LabelT>
	InputLabelPair& operator=(
		InputLabelPair<InputT,LabelT> const& pair
	){
		input = pair.input;
		label = pair.label;
		return *this;
	}
		
	friend bool operator<(InputLabelPair const& op1, InputLabelPair const& op2){
		return op1.label < op2.label;
	}
};

template<class I1, class L1, class I2, class L2>
void swap(InputLabelPair<I1, L1>&& p1, InputLabelPair<I2, L2>&& p2){
	using std::swap;
	swap(p1.input,p2.input);
	swap(p1.label,p2.label);
}


#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class InputType, class LabelType>
SHARK_CREATE_BATCH_INTERFACE(
	InputLabelPair<InputType BOOST_PP_COMMA() LabelType>,
	(InputType, input)(LabelType, label)
)
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

}
#endif