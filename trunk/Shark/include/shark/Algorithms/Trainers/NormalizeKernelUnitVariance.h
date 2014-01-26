//===========================================================================
/*!
 * 
 *
 * \brief       Determine the scaling factor of a ScaledKernel so that it has unit variance in feature space one on a given dataset.
 * 
 * 
 * 
 *
 * \author      M. Tuma
 * \date        2012
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


#ifndef SHARK_ALGORITHMS_TRAINERS_NORMALIZEKERNELUNITVARIANCE_H
#define SHARK_ALGORITHMS_TRAINERS_NORMALIZEKERNELUNITVARIANCE_H


#include <shark/Models/Kernels/ScaledKernel.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>

namespace shark {


///
/// \brief Determine the scaling factor of a ScaledKernel so that it has unit variance in feature space one on a given dataset.
///
/// \par
/// For example in the multiple kernel learning setting, it can be important that the sub-kernels
/// are normalized to unit variance in feature space. This class computes both the trace and the
/// mean of a kernel matrix, and uses both to employ the "Multiplicative Kernel Scaling" laid out
/// in "Kloft, Brefeld, Sonnenburg, Zien: l_p-Norm Multiple Kernel Learning. JMLR 12, 2011.".
/// Given a ScaledKernel, which itself holds an arbitrary underlying kernel k, we compute
/// \f[ \frac{1}{N}\sum_{i=1}^N k(x_i,x_i) - \frac{1}{N^2} \sum_{i,j=1}^N k(x_i,x_j) \f]
/// 
/// 
/// 
template < class InputType = RealVector >
class NormalizeKernelUnitVariance : public AbstractUnsupervisedTrainer<ScaledKernel<InputType> >
{
public:

	NormalizeKernelUnitVariance()
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NormalizeKernelUnitVariance"; }

	double trace() const {
		return m_matrixTrace;
	}
	double mean() const {
		return m_mean;
	}

	void train( ScaledKernel<InputType>& kernel, UnlabeledData<InputType> const& input )
	{
		SHARK_CHECK(input.numberOfElements() >= 2, "[NormalizeKernelUnitVariance::train] input needs to contain at least two points");
		AbstractKernelFunction< InputType > const& k = *kernel.base(); //get direct access to the kernel we want to use.		
		
		// Next compute the trace and mean of the kernel matrix. This means heavy lifting: 
		// calculate each entry of one diagonal half of the kernel matrix exactly once.
		// [MT] This part was taken from the PrecomputedMatrix constructor in QuadraticProgram.h
		// Blockwise version, with order of computations optimized for better use of the processor
		// cache. This can save around 10% computation time for fast kernel functions.
		std::size_t N = input.numberOfElements();
		
		//O.K. tried to make it more efficient (and shorter)
		m_mean = 0.0;
		m_matrixTrace = 0.0;
		for(std::size_t i = 0; i != input.numberOfBatches(); ++i){
			typename UnlabeledData<InputType>::const_batch_reference batch = input.batch(i);
			//off diagonal entries
			for(std::size_t j = 0; j < i; ++j){
				RealMatrix matrixBlock = k(batch, input.batch(j));
				m_mean += 2*sum(matrixBlock);
			}
			RealMatrix matrixBlock = k(batch, batch);
			m_mean += sum(matrixBlock);
			m_matrixTrace += blas::trace(matrixBlock);
		}
		double tm = m_matrixTrace/N - m_mean/N/N;
		SHARK_ASSERT( tm > 0 );
		double scaling_factor = 1.0 / tm;
		kernel.setFactor( scaling_factor );
	}

protected:
	double m_mean; //store for other uses, external queries, etc.
	double m_matrixTrace;
};


}
#endif
