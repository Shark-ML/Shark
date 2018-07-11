//===========================================================================
/*!
 * 
 *
 * \brief       Defines a helper class which assigns to every element of a tuple of points a kernel of a tuple of kernels
 * 
 * 
 *
 * \author      O.Krause
 * \date        2012
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

#ifndef SHARK_MODELS_KERNELS_IMPL_MKL_KERNEL_BASE_H
#define SHARK_MODELS_KERNELS_IMPL_MKL_KERNEL_BASE_H


#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Data/BatchInterfaceAdaptStruct.h>
#include <shark/Core/utility/integer_sequence.h>

namespace shark {
	
	
namespace detail{
/// \brief given a tuple of Inputs (a_1,...a_n) calculates the kernel k(a_N,a_N) for some chosen N
///
/// \warning This Class assumes that Batch<T> is specialised for the input tuple such, that
/// Batch<T>::type is a tuple of batches 
template<class InputType, std::size_t N>
class MklKernelWrapper : public AbstractKernelFunction<InputType>{
private:
	typedef AbstractKernelFunction<InputType> base_type;
public:
	/// \brief the type of the N-th element of the tuple
	typedef typename std::remove_reference<decltype(Batch<InputType>::template tuple_elem<N>(std::declval<InputType&>()))>::type  KernelInputType;
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstInputReference ConstInputReference;

	MklKernelWrapper(AbstractKernelFunction<KernelInputType>* kernel):m_kernel(kernel){}
	
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
		return m_kernel->eval(Batch<InputType>::template tuple_elem<N>(x1),Batch<InputType>::template tuple_elem<N>(x2));
	}
	
	void eval(BatchInputType const& batchX1, BatchInputType const& batchX2, RealMatrix& result, State& state) const{
		m_kernel->eval(Batch<InputType>::template tuple_elem<N>(batchX1),Batch<InputType>::template tuple_elem<N>(batchX2),result,state);
	}
	
	void eval(BatchInputType const& batchX1, BatchInputType const& batchX2, RealMatrix& result) const{
		m_kernel->eval(Batch<InputType>::template tuple_elem<N>(batchX1),Batch<InputType>::template tuple_elem<N>(batchX2),result);
	}
	
	void weightedParameterDerivative(
		BatchInputType const& batchX1, 
		BatchInputType const& batchX2, 
		RealMatrix const& coefficients,
		State const& state, 
		RealVector& gradient
	) const{
		m_kernel->weightedParameterDerivative(
			Batch<InputType>::template tuple_elem<N>(batchX1),
			Batch<InputType>::template tuple_elem<N>(batchX2),
			coefficients,state,gradient
		);
	}
	void weightedInputDerivative( 
		BatchInputType const& batchX1, 
		BatchInputType const& batchX2, 
		RealVector const& coefficientsX2,
		State const& state, 
		BatchInputType& gradient
	) const{
		m_kernel->weightedInputDerivative(
			Batch<InputType>::template tuple_elem<N>(batchX1),
			Batch<InputType>::template tuple_elem<N>(batchX2),
			coefficientsX2,state,
			Batch<InputType>::template tuple_elem<N>(gradient)
		);
	}
	
	//w don't need serializing here, this is done by the implementing Kernel
	void read(InArchive& ar){}
	void write(OutArchive& ar) const{}
	
private:
	AbstractKernelFunction<KernelInputType>* m_kernel;
};

template<class InputType>
class MklKernelBase
{
public:
	/// \brief number of Kernels stored in the KernelContainer
	static const std::size_t NumKernels = Batch<InputType>::tuple_size::value; 
private:
	
	//figure out what the kernel type for a given tuple should be
	template<int... Is>
	static auto type_generator(integer_sequence<Is...>) -> std::tuple<MklKernelWrapper<InputType,Is>... >;
	typedef  decltype(type_generator(generate_integer_sequence<NumKernels>())) KernelContainer;
	
	//map each stored wrapper to its address
	template<int... Is>
	std::vector<AbstractKernelFunction<InputType>* > makeKernelVectorImpl(integer_sequence<Is...>){
		auto pointer = [](AbstractKernelFunction<InputType>& kernel){
			return static_cast<AbstractKernelFunction<InputType> *>(&kernel);
		};
		return { pointer(std::get<Is>(m_kernelWrappers))... };
	}
public:

	
	template<class... KernelArgs, class = typename std::enable_if<sizeof...(KernelArgs) == NumKernels, void>::type>
	MklKernelBase(KernelArgs const&... kernels)
	: m_kernelWrappers(std::make_tuple(kernels...)){}
	
	std::vector<AbstractKernelFunction<InputType>* > makeKernelVector(){
		return makeKernelVectorImpl(generate_integer_sequence<NumKernels>());
	}
	
	KernelContainer m_kernelWrappers;
};
}
}
#endif
