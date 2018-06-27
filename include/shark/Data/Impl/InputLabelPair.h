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

#include <shark/Data/BatchInterface.h>

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

///\brief Input label pair of batches
template<class Batch1Type,class Batch2Type>
struct InputLabelBatch{
private:
	typedef typename BatchTraits<typename std::decay<Batch1Type>::type >::type Batch1Traits;
	typedef typename BatchTraits<typename std::decay<Batch2Type>::type >::type Batch2Traits;
public:
	Batch1Type input;
	Batch2Type label;

	typedef InputLabelPair<
		typename Batch1Traits::value_type,
		typename Batch2Traits::value_type
	> value_type;
	//the decltype below adds correct const semantic if the template arguments are references.
	//the behaviour is the same as mimiking the pair {getBatchElement(input,i), getBatchElement(label,i)}
	//depending on whether input or label are const or not (which for reference types should not make any difference)
	typedef InputLabelPair<
		decltype(getBatchElement(std::declval<Batch1Type&>(),0)),
		decltype(getBatchElement(std::declval<Batch2Type&>(),0))
	> reference;
	typedef InputLabelPair<
		decltype(getBatchElement(std::declval<typename std::add_const<Batch1Type>::type&>(),0)),
		decltype(getBatchElement(std::declval<typename std::add_const<Batch2Type>::type&>(),0))
	> const_reference;

	template<class I, class L>
	InputLabelBatch(
		I&& input,
		L&& label
	):input(input),label(label){}
	
	template<class I, class L>
	InputLabelBatch& operator=(InputLabelBatch<I,L> const& batch){
		input = batch.input;
		label = batch.label;
		return *this;
	}

	std::size_t size()const{
		return Batch1Traits::size(input);
	}
	reference operator[](std::size_t i){
		return reference(getBatchElement(input,i),getBatchElement(label,i));
	}
	const_reference operator[](std::size_t i)const{
		return const_reference(getBatchElement(input,i),getBatchElement(label,i));
	}
};

template<class I1, class L1, class I2, class L2>
void swap(InputLabelPair<I1, L1>&& p1, InputLabelPair<I2, L2>&& p2){
	using std::swap;
	swap(p1.input,p2.input);
	swap(p1.label,p2.label);
}

template<class I1, class L1, class I2, class L2>
void swap(InputLabelBatch<I1, L1>& p1, InputLabelBatch<I2, L2>& p2){
	using std::swap;
	swap(p1.input,p2.input);
	swap(p1.label,p2.label);
}

template<class InputType, class LabelType>
struct Batch<InputLabelPair<InputType, LabelType> >
: public detail::SimpleBatch<
	InputLabelBatch<typename detail::element_to_batch<InputType>::type, typename detail::element_to_batch<LabelType>::type>
>{
	typedef InputLabelPair<typename Batch<InputType>::shape_type, typename Batch<LabelType>::shape_type> shape_type;
	typedef InputLabelBatch<
		typename detail::element_to_batch<InputType>::type,
		typename detail::element_to_batch<LabelType>::type
	> type;
	/// \brief Creates a batch with enough dimensions to store a vector of a specified shape
	static type createBatchFromShape(shape_type const& shape, std::size_t size = 1){
		return type(
			Batch<InputType>::createBatchFromShape(shape.input,size),
			Batch<LabelType>::createBatchFromShape(shape.label,size)
		);
	}
	
	///\brief creates a batch storing the elements referenced by the provided range
	template<class Iterator>
	static type createBatchFromRange(Iterator const& begin, Iterator const& end){
		std::size_t size = end - begin;
		type batch(
			Batch<InputType>::createBatch(begin->input,size),
			Batch<LabelType>::createBatch(begin->label,size)
		);
		auto pos = begin;
		for(std::size_t i = 0; i != size; ++i, ++pos){
			batch[i] = *pos;
		}
		return batch;
	}
	
	/// \brief Returns a feasible shape for the given range of elements
	template<class Iterator>
	static shape_type inferShape(Iterator const& start, Iterator const& end);
	//~ {
		//~ typedef typename Iterator::const_reference ref_type;
		//~ auto getInput = [](ref_type x){return x.input;};
		//~ auto getLabel = [](ref_type x){return x.label;};
		//~ auto inputStart = boost::make_transform_iterator(start,getInput);
		//~ auto inputEnd = boost::make_transform_iterator(end,getInput);
		//~ auto labelStart = boost::make_transform_iterator(start,getLabel);
		//~ auto labelEnd = boost::make_transform_iterator(end,getLabel);
		//~ return shape_type(
			//~ Batch<Input>::inferShape(inputStart, inputEnd),
			//~ Batch<Label>::inferShape(labelStart, labelEnd),
		//~ );
	//~ }
};

template<class InputBatchType, class LabelBatchType>
struct BatchTraits<InputLabelBatch<InputBatchType, LabelBatchType> >{
	typedef typename detail::batch_to_element<InputBatchType>::type InputElem;
	typedef typename detail::batch_to_element<LabelBatchType>::type LabelElem;
	typedef Batch<InputLabelPair<InputElem,LabelElem> > type;
};
}
#endif