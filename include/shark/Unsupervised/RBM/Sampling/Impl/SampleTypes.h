/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
 *
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
#ifndef SHARK_UNSUPERVISED_RBM_SAMPLING_IMPL_SAMPLETYPES_H
#define SHARK_UNSUPERVISED_RBM_SAMPLING_IMPL_SAMPLETYPES_H

#include <shark/LinAlg/Base.h>
#include <utility>
namespace shark{
namespace detail{
///\brief Represents a single sample of the GibbsOperator.
///
///This class is not actually used by the GibbsOperator since it works only on a Batch of samples.
///Therefore it is only useful for direct use, when an explicit copy of a sample is needed.
template<class Statistics>
struct GibbsSample{
public:
	RealMatrix input;
	Statistics statistics;
	RealMatrix state;
	
	///\brief Constructs sample with size neurons
	GibbsSample(){}
	explicit GibbsSample(std::size_t batchSize, std::size_t numNeurons):
	input(batchSize, numNeurons),statistics(batchSize, numNeurons),state(batchSize, numNeurons)
	{}
	GibbsSample(RealMatrix const& input, Statistics const& statistics, RealMatrix const& state)
	:input(input),statistics(statistics),state(state)
	{}
	
	std::size_t size() const{
		return input.size1();
	}
	
	void swap_rows(std::size_t i, std::size_t j){
		input.swap_rows(i,j);
		statistics.swap_rows(i,j);
		state.swap_rows(i,j);
	}
};

template<class HiddenSample, class VisibleSample>
struct MarkovChainSample{
	HiddenSample hidden;
	VisibleSample visible;
	RealVector energy;
		
	MarkovChainSample(){}
	explicit MarkovChainSample(std::size_t size, std::size_t hiddenNeurons, std::size_t visibleNeurons)
	:hidden(size, hiddenNeurons),visible(size, visibleNeurons),energy(size, 0){}

	explicit MarkovChainSample(HiddenSample const& hidden, VisibleSample const& visible, RealVector const& energy)
	:hidden(hidden),visible(visible),energy(energy)
	{}
	
	std::size_t size() const{
		return energy.size();
	}
	
	void swap_rows(std::size_t i, std::size_t j){
		hidden.swap_rows(i,j);
		visible.swap_rows(i,j);
		std::swap(energy(i),energy(j));
	}

};
}}
#endif
