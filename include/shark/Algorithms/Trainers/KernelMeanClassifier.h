//===========================================================================
/*!
 * 
 *
 * \brief       KernelMeanClassifier
 * 
 * 
 *
 * \author      T. Glasmachers, C. Igel
 * \date        2010, 2011
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
//===========================================================================
#ifndef SHARK_ALGORITHMS_TRAINERS_KERNELMEAN_H
#define SHARK_ALGORITHMS_TRAINERS_KERNELMEAN_H


#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Algorithms/Trainers/AbstractWeightedTrainer.h>
#include <shark/Data/Dataset.h>

namespace shark {

/// \brief Kernelized mean-classifier
///
/// Computes the mean of the training data in feature space for each
/// class and assigns a new data point to the class with the nearest
/// mean. The trainer supports multi-class and weighted data
///
/// The resulting classifier is a kernel expansion as assigning the label
/// with minimum distance (or maximum  negative distance by convention for classifiers)
/// \f[ max -1/2 ||\phi(x) - m_i||^2 = <\phi(x), m_i> - 1/2<m_i,m_i> \f]
/// \ingroup supervised_trainer
template<class InputType>
class KernelMeanClassifier : public AbstractWeightedTrainer<KernelClassifier<InputType>, unsigned int>{
public:
	KernelMeanClassifier(AbstractKernelFunction<InputType>* kernel):mpe_kernel(kernel){}
		
	std::string name() const
	{ return "KernelMeanClassifier"; }

	using AbstractWeightedTrainer<KernelClassifier<InputType>, unsigned int>::train;
	void train(KernelClassifier<InputType>& model, WeightedLabeledData<InputType, unsigned int> const& dataset){
		RealVector normalization = classWeight(dataset);
		std::size_t patterns = dataset.numberOfElements();
		std::size_t numClasses = normalization.size();
		SHARK_RUNTIME_CHECK(min(normalization) > 0, "One class has no member" );

		// compute coefficients and offset term
		RealVector offset(numClasses,0.0);
		RealMatrix alpha(patterns, numClasses,0.0);
		
		//todo: slow implementation without batch processing!
		std::size_t i  = 0; 
		for(auto const& element: dataset.elements()){
		
			unsigned int y = element.data.label;
			double w = element.weight;

			// compute and set coefficients
			alpha(i,y) = w / normalization(y);
			++i;
			// compute values to calculate offset
			for(auto element2: dataset.elements()){
				if (element2.data.label != y) 
					continue;
				//todo: fast implementation should create batches of same class elements and process them!
				offset(y) += w * element2.weight * mpe_kernel->eval(element.data.input, element2.data.input);
			}
		}
		noalias(offset) /= sqr(normalization);
		
		if(numClasses == 2){
			model.decisionFunction().setStructure(mpe_kernel,dataset.inputs(),true);
			noalias(column(model.decisionFunction().alpha(),0)) = column(alpha,1) - column(alpha,0);
			model.decisionFunction().offset()(0) = (offset(0) - offset(1))/2;
		}else{
			model.decisionFunction().setStructure(mpe_kernel,dataset.inputs(),true, numClasses);
			noalias(model.decisionFunction().alpha()) = alpha;
			noalias(model.decisionFunction().offset()) = -offset/2;
		}
	}
	
	AbstractKernelFunction<InputType>* mpe_kernel;
};


}
#endif
