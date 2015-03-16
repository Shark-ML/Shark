//===========================================================================
/*!
 * 
 *
 * \brief       monomial (polynomial) kernel
 * 
 * 
 *
 * \author      T.Glasmachers, O. Krause, M. Tuma
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

#ifndef SHARK_MODELS_KERNELS_MONOMIAL_KERNEL_H
#define SHARK_MODELS_KERNELS_MONOMIAL_KERNEL_H


#include <shark/Models/Kernels/AbstractKernelFunction.h>
namespace shark {


/// \brief Monomial kernel. Calculates \f$ \left\langle x_1, x_2 \right\rangle^m_exponent \f$
///
/// \par
/// The degree \f$ m_exponent \f$ is a non-trainable but configurable parameter.
/// The default value is one - exactly the same as a LinearKernel.
template<class InputType=RealVector>
class MonomialKernel : public AbstractKernelFunction<InputType>
{
private:
	typedef AbstractKernelFunction<InputType> base_type;
	
	struct InternalState: public State{
		RealMatrix base;//stores the inner product of vectors x_1,x_j which is the base the late rused pow
		RealMatrix exponentedProd;//pow(base,m_exponent)
		
		void resize(std::size_t sizeX1, std::size_t sizeX2){
			base.resize(sizeX1, sizeX2);
			exponentedProd.resize(sizeX1, sizeX2);
		}
	};
	
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstInputReference ConstInputReference;
	typedef typename base_type::ConstBatchInputReference ConstBatchInputReference;
	MonomialKernel():m_exponent(1){
		this->m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		this->m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		this->m_features |= base_type::SUPPORTS_VARIABLE_INPUT_SIZE;
	}
	MonomialKernel(unsigned int n):m_exponent(n){
		this->m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		this->m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		this->m_features |= base_type::SUPPORTS_VARIABLE_INPUT_SIZE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "MonomialKernel"; }

	RealVector parameterVector() const{ 
		return RealVector(0); 
	}
	void setParameterVector(RealVector const& newParameters){ 
		SIZE_CHECK(newParameters.size() == 0); 
	}
	std::size_t numberOfParameters() const{ 
		return 0; 
	}
	
	///\brief creates the internal state of the kernel
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}
	
	/////////////////////////EVALUATION//////////////////////////////
	double eval(ConstInputReference x1, ConstInputReference x2) const{
		SIZE_CHECK(x1.size() == x2.size());
		double prod=inner_prod(x1, x2);
		return std::pow(prod,m_exponent);
	}
	
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result) const{
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		std::size_t sizeX1 = batchX1.size1();
		std::size_t sizeX2 = batchX2.size1();
		result.resize(sizeX1,sizeX2);
		//calculate the inner product
		axpy_prod(batchX1,trans(batchX2),result);
		if(m_exponent != 1)
			noalias(result) = pow(result,m_exponent);
	}
	
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result, State& state) const{
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		std::size_t sizeX1 = batchX1.size1();
		std::size_t sizeX2 = batchX2.size1();
		result.resize(sizeX1,sizeX2);
		

		InternalState& s = state.toState<InternalState>();
		s.resize(sizeX1,sizeX2);
		
		//calculate the inner product
		axpy_prod(batchX1,trans(batchX2),s.base);
		//now do exponentiation
		if(m_exponent != 1)
			noalias(result) = pow(s.base,m_exponent);
		else
			noalias(result) = s.base;
		//store also in state
		noalias(s.exponentedProd) = result;
			
	}
	
	////////////////////////DERIVATIVES////////////////////////////
	
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
		
		std::size_t sizeX1 = batchX1.size1();
		std::size_t sizeX2 = batchX2.size1();
		gradient.resize(sizeX1,batchX1.size2());
		InternalState const& s = state.toState<InternalState>();
		
		//internal checks
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		SIZE_CHECK(s.base.size1() == sizeX1);
		SIZE_CHECK(s.base.size2() == sizeX2);
		SIZE_CHECK(s.exponentedProd.size1() == sizeX1);
		SIZE_CHECK(s.exponentedProd.size2() == sizeX2);
		
		//first calculate weights(i,j) = coeff(i)*exp(i,j)/prod(i,j)
		//we have to take the usual division by 0 into account
		RealMatrix weights = coefficientsX2 * safe_div(s.exponentedProd,s.base,0.0);
		//The derivative of input i of batch x1 is 
		//g = sum_j m_exponent*weights(i,j)*x2_j
		//we now sum over j which is a matrix-matrix product
		axpy_prod(weights,batchX2,gradient);
		gradient*= m_exponent;
	}
	
	void read(InArchive& ar){
		ar >> m_exponent;
	}

	void write(OutArchive& ar) const{
		ar << m_exponent;
	}

protected:
	///the exponent of the monomials
	int m_exponent;
};

typedef MonomialKernel<> DenseMonomialKernel;
typedef MonomialKernel<CompressedRealVector> CompressedMonomialKernel;


}
#endif
