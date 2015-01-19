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
 * \par Copyright 1995-2015 Shark Development Team
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

#ifndef SHARK_MODELS_KERNELS_IMPL_MKL_KERNEL_BASE_H
#define SHARK_MODELS_KERNELS_IMPL_MKL_KERNEL_BASE_H


#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Data/BatchInterfaceAdaptStruct.h> //need in a lot of MKL-Kernel-Applications
#include <boost/fusion/algorithm/iteration/fold.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/mpl/transform_view.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/fusion/container/generation/make_vector.hpp>
namespace shark {
	
	
namespace detail{
/// \brief given a tuple of Inputs (a_1,...a_n) calculates the kernel k(a_N,a_N) for some chosen N
///
/// \warning This Class assumes that Batch<T> is specialised for the input tuple such, that
/// Batch<T>::type is a tuple of batches which can be queried using boost fusion!
template<class InputType, std::size_t N>
class MklKernelWrapper : public AbstractKernelFunction<InputType>{
private:
	typedef AbstractKernelFunction<InputType> base_type;
public:
	/// \brief the type of the N-th element of the tuple
	typedef typename boost::fusion::result_of::value_at<
		InputType,
		boost::mpl::int_<N>
	>::type KernelInputType;
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
		return m_kernel->eval(boost::fusion::at_c<N>(x1),boost::fusion::at_c<N>(x2));
	}
	
	void eval(BatchInputType const& batchX1, BatchInputType const& batchX2, RealMatrix& result, State& state) const{
		m_kernel->eval(boost::fusion::at_c<N>(batchX1),boost::fusion::at_c<N>(batchX2),result,state);
	}
	
	void eval(BatchInputType const& batchX1, BatchInputType const& batchX2, RealMatrix& result) const{
		m_kernel->eval(boost::fusion::at_c<N>(batchX1),boost::fusion::at_c<N>(batchX2),result);
	}
	
	void weightedParameterDerivative(
		BatchInputType const& batchX1, 
		BatchInputType const& batchX2, 
		RealMatrix const& coefficients,
		State const& state, 
		RealVector& gradient
	) const{
		m_kernel->weightedParameterDerivative(boost::fusion::at_c<N>(batchX1),boost::fusion::at_c<N>(batchX2),coefficients,state,gradient);
	}
	void weightedInputDerivative( 
		BatchInputType const& batchX1, 
		BatchInputType const& batchX2, 
		RealVector const& coefficientsX2,
		State const& state, 
		BatchInputType& gradient
	) const{
		m_kernel->weightedInputDerivative(boost::fusion::at_c<N>(batchX1),boost::fusion::at_c<N>(batchX2),coefficientsX2,state,boost::fusion::at_c<N>(gradient));
	}
	
	//w don't need serializing here, this is done by the implementing Kernel
	void read(InArchive& ar){
	}
	void write(OutArchive& ar) const{
	}
	
private:
	AbstractKernelFunction<KernelInputType>* m_kernel;
};

template<class InputType>
class MklKernelBase
{
public:
	/// \brief number of Kernels stored in the KernelContainer
	//static const std::size_t NumKernels = boost::fusion::size<InputType>::value;
	BOOST_STATIC_CONSTANT(std::size_t, NumKernels = boost::fusion::result_of::size<InputType>::value ); 
private:
	
	///\brief metafunction creating the kernel type of the N-th input element
	template<class N>
	struct KernelType{
		typedef detail::MklKernelWrapper<InputType,N::value> type;
	};
	
	///\brief maps a fusion sequence of kernels to a sequence of pointers to the base class
	struct MakeKernelContainer{
		typedef std::vector<AbstractKernelFunction<InputType>* > result_type;
		
		result_type operator()(result_type container, AbstractKernelFunction<InputType> const& k) const
		{
			container.push_back(const_cast<AbstractKernelFunction<InputType>* >(&k));
			return container;
		};
	};
public:
	
	/// \brief The Type of Container used to hold the Kernels.
	typedef typename boost::fusion::result_of::as_vector<
		boost::mpl::transform_view<
			boost::mpl::range_c<unsigned int,0,NumKernels>,
			KernelType<boost::mpl::_1>
		>
	>::type KernelContainer;
	
	template<class KernelTuple>
	MklKernelBase(KernelTuple const& kernels)
	: m_kernelWrappers(kernels){}
	
	std::vector<AbstractKernelFunction<InputType>* > makeKernelVector(){
		return boost::fusion::fold(//fold the wrapper sequence to a normal vector of pointers
			m_kernelWrappers,
			std::vector<AbstractKernelFunction<InputType>* >(),
			MakeKernelContainer()
		);
	}
	
	KernelContainer m_kernelWrappers;
};
}
}
#endif