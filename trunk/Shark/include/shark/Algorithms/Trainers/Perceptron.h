//===========================================================================
/*!
 * 
 *
 * \brief       Perceptron
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010
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
#ifndef SHARK_ALGORITHMS_TRAINERS_PERCEPTRON_H
#define SHARK_ALGORITHMS_TRAINERS_PERCEPTRON_H


#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>

namespace shark{

//! \brief Perceptron online learning algorithm
template<class InputType>
class Perceptron : public AbstractTrainer<KernelClassifier<InputType>,unsigned int >
{
public:
	/// \brief Constructor.
	///
	/// @param kernel is the (Mercer) kernel function.
	/// @param maxTimesPattern defines the maximum number of times the data is processed before the algorithms stopps.
	Perceptron(AbstractKernelFunction<InputType>* kernel, std::size_t maxTimesPattern = 10000)
	:mpe_kernel(kernel),m_maxTimesPattern(maxTimesPattern){}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Perceptron"; }

	void train(KernelClassifier<InputType>& classifier, LabeledData<InputType, unsigned int> const& dataset){
		std::size_t patterns = dataset.numberOfElements();
		KernelExpansion<InputType>& model= classifier.decisionFunction();
		model.setStructure(mpe_kernel,dataset.inputs(),false,1);
		model.alpha().clear();

		bool err;
		std::size_t iter = 0;
		do {
			err = false;
			for (std::size_t i = 0; i != patterns; i++){
				double result = model(dataset.element(i).input)(0);
				//perceptron learning rule with modified target from -1;1
				double label = dataset.element(i).label*2.0-1;
				if ( result * label  <= 0.0){
					model.alpha(i,0) += label;
					err = true;
				}
			}
			if (iter > m_maxTimesPattern * patterns) break;	// probably non-separable data
			iter++;
		} while (err);
	}
private:
	AbstractKernelFunction<InputType>* mpe_kernel;
	std::size_t m_maxTimesPattern; //< maximum number of times a training is processed
};


}
#endif
