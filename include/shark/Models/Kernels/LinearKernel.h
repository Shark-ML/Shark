//===========================================================================
/*!
 * 
 *
 * \brief       linear kernel (standard inner product)
 * 
 * 
 *
 * \author      T.Glasmachers, O. Krause, M. Tuma
 * \date        2010, 2011
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

#ifndef SHARK_MODELS_KERNELS_LINEAR_KERNEL_H
#define SHARK_MODELS_KERNELS_LINEAR_KERNEL_H


#include <shark/Models/Kernels/AbstractKernelFunction.h>

namespace shark{


/// \brief Linear Kernel, parameter free
template<class InputType=RealVector>
class LinearKernel : public AbstractKernelFunction<InputType>
{
private:
	typedef AbstractKernelFunction<InputType> base_type;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstInputReference ConstInputReference;
	typedef typename base_type::ConstBatchInputReference ConstBatchInputReference;
	
	LinearKernel(){
		this->m_features|=base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		this->m_features|=base_type::HAS_FIRST_INPUT_DERIVATIVE;
		this->m_features|= base_type::SUPPORTS_VARIABLE_INPUT_SIZE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearKernel"; }

	RealVector parameterVector() const{ 
		return RealVector(); 
	}
	void setParameterVector(RealVector const& newParameters){ 
		SIZE_CHECK(newParameters.size() == 0); 
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	double eval(ConstInputReference x1, ConstInputReference x2) const{
		SIZE_CHECK(x1.size() == x2.size());
		return inner_prod(x1, x2);
	}
	
	void eval(ConstBatchInputReference x1, ConstBatchInputReference x2, RealMatrix& result, State& state) const{
		eval(x1,x2,result);
	}
	
	void eval(ConstBatchInputReference x1, ConstBatchInputReference x2, RealMatrix& result) const{
		SIZE_CHECK(x1.size2() == x2.size2());
		result.resize(x1.size1(),x2.size1());
		axpy_prod(x1,trans(x2),result);
	}
	
	void weightedParameterDerivative(
		ConstBatchInputReference batchX1, 
		ConstBatchInputReference batchX2, 
		RealMatrix const& coefficients,
		State const& state, 
		RealVector& gradient
	) const{
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		gradient.resize(0);
	}
	void weightedInputDerivative( 
		ConstBatchInputReference batchX1, 
		ConstBatchInputReference batchX2, 
		RealMatrix const& coefficientsX2,
		State const& state, 
		BatchInputType& gradient
	) const{
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		//~ SIZE_CHECK(cofficientsX2.size1() == batchX1.size1());
		//~ SIZE_CHECK(cofficientsX2.size2() == batchX2.size1());
		gradient.resize(batchX1.size1(),batchX1.size2());
		
		axpy_prod(coefficientsX2,batchX2,gradient);
	}
	
	virtual double featureDistanceSqr(ConstInputReference x1, ConstInputReference x2) const{
		return distanceSqr(x1,x2);
	}
	
	virtual RealMatrix featureDistanceSqr(ConstBatchInputReference x1, ConstBatchInputReference x2) const{
		return distanceSqr(x1,x2);
	}
	
	/// \brief The kernel does not serialize anything
	void read(InArchive& ar){}

	/// \brief The kernel does not serialize anything
	void write(OutArchive& ar) const{}
	
};

typedef LinearKernel<> DenseLinearKernel;
typedef LinearKernel<CompressedRealVector> CompressedLinearKernel;


}
#endif
