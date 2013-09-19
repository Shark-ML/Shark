//===========================================================================
/*!
 *  \brief KernelMeanClassifier
 *
 *  \author T. Glasmachers, C. Igel
 *  \date 2010, 2011
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
//===========================================================================
#ifndef SHARK_ALGORITHMS_TRAINERS_KERNELMEAN_H
#define SHARK_ALGORITHMS_TRAINERS_KERNELMEAN_H


#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Data/Dataset.h>

namespace shark {

/*! \brief Kernelized mean-classifier
 *
 * Computes the mean of the training data in feature space for each
 * class and assigns a new data point to the class with the nearest
 * mean.
 */
template<class InputType>
class KernelMeanClassifier : public AbstractTrainer<KernelClassifier<InputType>, unsigned int>
{
public:
	KernelMeanClassifier(AbstractKernelFunction<InputType>* kernel):mpe_kernel(kernel){}
		
	std::string name() const
	{ return "KernelMeanClassifier"; }

	void train(KernelClassifier<InputType>& model, LabeledData<InputType, unsigned int> const& dataset){
		SHARK_CHECK(numberOfClasses(dataset) ==2, "[KernelMeanClassifier::train] not a binary class problem");
		
		model.decisionFunction().setStructure(mpe_kernel,dataset.inputs(),true);

		std::size_t patterns = dataset.numberOfElements();
		std::vector<std::size_t> numClasses = classSizes(dataset);
		double coeffs[]     = {0,0};
		
		SHARK_CHECK(numClasses[0] > 0, "[KernelMeanClassifier::train] class 0 has no class members" );
		SHARK_CHECK(numClasses[1] > 0, "[KernelMeanClassifier::train] class 1 has no class members" );
		
		coeffs[0] =  1.0 / numClasses[0];
		coeffs[1] = -1.0 / numClasses[1];

		// compute coefficients and bias term
		double classBias[]={0.0,0.0};
		RealVector params(patterns + 1);
		
		//todo: slow implementation without batch processing!
		typedef typename LabeledData<InputType, unsigned int>::const_element_reference ElementRef;
		std::size_t i  = 0; 
		BOOST_FOREACH(ElementRef element,dataset.elements()){
		
			unsigned int y = element.label;

			// compute and set coefficients
			params(i) = coeffs[y];
			++i;
			// compute values to calculate bias
			BOOST_FOREACH(ElementRef element2,dataset.elements()){
				if (element2.label != y) 
					continue;
				//todo: fast implementation should create batches of same class elements and process them!
				classBias[y] += mpe_kernel->eval(element.input, element2.input);
			}
		}
		// set bias
		params(patterns) = 0.5 * (classBias[0] * sqr(coeffs[0]) - classBias[1] * sqr(coeffs[1]));
		// pass parameters to model, note the negation
		model.setParameterVector(-params); 
	}
	
	AbstractKernelFunction<InputType>* mpe_kernel;
};


}
#endif
