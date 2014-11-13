//===========================================================================
/*!
 * 
 *
 * \brief       Variant of WeightedSumKernel which works on subranges of Vector inputs
 * 
 * 
 *
 * \author      S., O.Krause
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

#ifndef SHARK_MODELS_KERNELS_SUBRANGE_KERNEL_H
#define SHARK_MODELS_KERNELS_SUBRANGE_KERNEL_H


#include <shark/Models/Kernels/WeightedSumKernel.h>
namespace shark {
namespace detail{
/// \brief given two vectors of input x = (x_1,...,x_n), y = (y_1,...,y_n), a subrange 1<=k<l<=n and a kernel k, computes the result of
///   th subrange k((x_k,...x_l),(y_k,...,y_l))
template<class InputType>
class SubrangeKernelWrapper : public AbstractKernelFunction<InputType>{
private:
	typedef AbstractKernelFunction<InputType> base_type;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstInputReference ConstInputReference;
	typedef typename base_type::ConstBatchInputReference ConstBatchInputReference;

	SubrangeKernelWrapper(AbstractKernelFunction<InputType>* kernel,std::size_t start, std::size_t end)
	:m_kernel(kernel),m_start(start),m_end(end){
		if(kernel->hasFirstParameterDerivative())
			this->m_features|=base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		if(kernel->hasFirstInputDerivative())
			this->m_features|=base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SubrangeKernelWrapper"; }

	RealVector parameterVector() const {
		return m_kernel->parameterVector();
	}

	void setParameterVector(RealVector const& newParameters) {
		m_kernel->setParameterVector(newParameters);
	}

	std::size_t numberOfParameters() const {
		return m_kernel->numberOfParameters();
	}

	///\brief creates the internal state of the kernel
	boost::shared_ptr<State> createState()const{
		return m_kernel->createState();
	}

	double eval(ConstInputReference x1, ConstInputReference x2) const{
		return m_kernel->eval(blas::subrange(x1,m_start,m_end),blas::subrange(x2,m_start,m_end));
	}

	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result, State& state) const{
		m_kernel->eval(columns(batchX1,m_start,m_end),columns(batchX2,m_start,m_end),result,state);
	}

	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result) const{
		m_kernel->eval(columns(batchX1,m_start,m_end),columns(batchX2,m_start,m_end),result);
	}

	void weightedParameterDerivative(
		ConstBatchInputReference batchX1,
		ConstBatchInputReference batchX2,
		RealMatrix const& coefficients,
		State const& state,
		RealVector& gradient
	) const{
		m_kernel->weightedParameterDerivative(
			columns(batchX1,m_start,m_end),
			columns(batchX2,m_start,m_end),
			coefficients,
			state,
			gradient
	);
	}
	void weightedInputDerivative(
		ConstBatchInputReference batchX1,
		ConstBatchInputReference batchX2,
		RealMatrix const& coefficientsX2,
		State const& state,
		BatchInputType& gradient
	) const{
		BatchInputType temp(gradient.size1(),m_end-m_start);
		m_kernel->weightedInputDerivative(
			columns(batchX1,m_start,m_end),
			columns(batchX2,m_start,m_end),
			coefficientsX2,
			state,
			temp
		);
		ensure_size(gradient,batchX1.size1(),batchX2.size2());
		gradient.clear();
		noalias(columns(gradient,m_start,m_end)) = temp;
	}

	//w don't need serializing here, this is done by the implementing Kernel
	void read(InArchive& ar){
	}
	void write(OutArchive& ar) const{
	}

private:
	AbstractKernelFunction<InputType>* m_kernel;
	std::size_t m_start;
	std::size_t m_end;
};

template<class InputType>
class SubrangeKernelBase
{
public:

	template<class Kernels,class Ranges>
	SubrangeKernelBase(Kernels const& kernels, Ranges const& ranges){
		SIZE_CHECK(size(kernels) == size(ranges));
		for(std::size_t i = 0; i != kernels.size(); ++i){
			m_kernelWrappers.push_back(
				SubrangeKernelWrapper<InputType>(get(kernels,i),get(ranges,i).first,get(ranges,i).second)
			);
		}
	}

	std::vector<AbstractKernelFunction<InputType>* > makeKernelVector(){
		std::vector<AbstractKernelFunction<InputType>* > kernels(m_kernelWrappers.size());
		for(std::size_t i = 0; i != m_kernelWrappers.size(); ++i)
			kernels[i] = & m_kernelWrappers[i];
		return kernels;
	}

	std::vector<SubrangeKernelWrapper <InputType> > m_kernelWrappers;
};
}

/// \brief Weighted sum of kernel functions
///
/// For a set of positive definite kernels \f$ k_1, \dots, k_n \f$
/// with positive coeffitients \f$ w_1, \dots, w_n \f$ the sum
/// \f[ \tilde k(x_1, x_2) := \sum_{i=1}^{n} w_i \cdot k_i(x_1, x_2) \f]
/// is again a positive definite kernel function. This still holds when
/// the sub-kernels only operate of a subset of features, that is, when
/// we have a direct sum kernel ( see e.g. the UCSC Technical Report UCSC-CRL-99-10:
/// Convolution Kernels on Discrete Structures by David Haussler ).
///
/// This class is very similar to the #WeightedSumKernel , except that it assumes it's
/// inputs to be tuples of values \f$ x=(x_1,\dots, x_n) \f$ and we calculate the direct
/// sum of kernels
/// \f[ \tilde k(x, y) := \sum_{i=1}^{n} w_i \cdot k_i(x_i, y_i) \f]
///
/// Internally, the weights are represented as \f$ w_i = \exp(\xi_i) \f$
/// to allow for unconstrained optimization.
///
/// The result of the kernel evaluation is devided by the sum of the
/// kernel weights, so that in total, this amounts to fixing the sum
/// of the weights to one.
template<class InputType>
class SubrangeKernel
: private detail::SubrangeKernelBase<InputType>//order is important!
, public WeightedSumKernel<InputType>
{
private:
	typedef detail::SubrangeKernelBase<InputType> base_type1;
	typedef WeightedSumKernel<InputType> base_type2;
public:

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SubrangeKernel"; }

	template<class Kernels,class Ranges>
	SubrangeKernel(Kernels const& kernels, Ranges const& ranges)
	: base_type1(kernels,ranges)
	, base_type2(base_type1::makeKernelVector()){}
};

typedef SubrangeKernel<RealVector> DenseSubrangeKernel;
typedef SubrangeKernel<CompressedRealVector> CompressesSubrangeKernel;

}
#endif
