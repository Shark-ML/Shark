//===========================================================================
/*!
*
*  \brief Polynomial kernel.
*
*  \author  T.Glasmachers
*  \date    2011
*
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

#ifndef SHARK_MODELS_KERNELS_POLYNOMIAL_KERNEL_H
#define SHARK_MODELS_KERNELS_POLYNOMIAL_KERNEL_H

#include <shark/Models/Kernels/AbstractKernelFunction.h>

namespace shark {


/// \brief Polynomial kernel.
///
/// \par
/// The polynomial kernel is defined as
/// \f$ \left( \left\langle x_1, x_2 \right\rangle + b \right)^n \f$
/// with degree \f$ n \in \mathbb{N} \f$ and non-negative offset
/// \f$ b \geq 0 \f$. For n=1 and b=0 it matches the linear kernel
/// (standard inner product).
template<class InputType = RealVector>
class PolynomialKernel : public AbstractKernelFunction<InputType>
{
private:
	typedef AbstractKernelFunction<InputType> base_type;
	
	struct InternalState: public State{
		RealMatrix base;//stores the inner product of vectors x_1,x_j
		RealMatrix exponentedProd;//pow(innerProd,m_exponent)
		
		void resize(std::size_t sizeX1, std::size_t sizeX2){
			base.resize(sizeX1, sizeX2);
			exponentedProd.resize(sizeX1, sizeX2);
		}
	};
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstInputReference ConstInputReference;
	typedef typename base_type::ConstBatchInputReference ConstBatchInputReference;

	/// Constructor.
	///
	/// \param  degree  exponent of the polynomial
	/// \param  offset  constant added to the standard inner product
	/// \param  degree_is_parameter should the degree be a regular model parameter? if yes, the kernel will not be differentiable
	/// \param  unconstrained should the offset internally be represented as exponential of the externally visible parameter?
	PolynomialKernel( unsigned int degree = 2, double offset = 0.0, bool degree_is_parameter = true, bool unconstrained = false )
	: m_degree( degree ),
	  m_offset( offset ),
	  m_degreeIsParam( degree_is_parameter ),
	  m_unconstrained( unconstrained ) {
		SHARK_CHECK(degree > 0, "[PolynomialKernel::PolynomialKernel] degree must be positive");
		this->m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		this->m_features |= base_type::SUPPORTS_VARIABLE_INPUT_SIZE;
		  if ( !m_degreeIsParam )
			this->m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "PolynomialKernel"; }

	void configure(PropertyTree const& node){
		m_degree = node.get("degree", 2);
		m_offset = node.get("offset", 0.0);
	}
	
	void setDegree( unsigned int deg ) {
		RANGE_CHECK( deg > 0 );
		SHARK_CHECK( !m_degreeIsParam, "[PolynomialKernel::setDegree] Please use setParameterVector when the degree is a parameter.");
		m_degree = deg;
	}
	
	unsigned int degree() const {
		return m_degree;
	}

	RealVector parameterVector() const {
		if ( m_degreeIsParam ) {
			RealVector ret(2);
			ret(0) = m_degree;
			if ( m_unconstrained ) 
				ret(1) = std::log( m_offset );
			else 
				ret(1) = m_offset;
			return ret;
		} else {
			RealVector ret(1);
			if ( m_unconstrained ) 
				ret(0) = std::log( m_offset );
			else 
				ret(0) = m_offset;
			return ret;
		}
	}

	void setParameterVector(RealVector const& newParameters) {
		if ( m_degreeIsParam ) {
			SIZE_CHECK(newParameters.size() == 2);
			SHARK_ASSERT(std::abs((unsigned int)newParameters(0) - newParameters(0)) <= 2*std::numeric_limits<double>::epsilon());
			RANGE_CHECK(newParameters(0) >= 1.0);
			m_degree = (unsigned int)newParameters(0);
			if ( m_unconstrained ) {
				m_offset = std::exp(newParameters(1));
			} else {
				RANGE_CHECK(newParameters(1) >= 0.0);
				m_offset = newParameters(1);
			}
		} else {
			SIZE_CHECK(newParameters.size() == 1);
			if ( m_unconstrained ) {
				m_offset = std::exp(newParameters(0));
			} else {
				RANGE_CHECK(newParameters(0) >= 0.0);
				m_offset = newParameters(0);
			}
		}
	}

	std::size_t numberOfParameters() const { 
		if ( m_degreeIsParam ) 
			return 2; 
		else 
			return 1;
	}
	
	///\brief creates the internal state of the kernel
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}

	/////////////////////////EVALUATION//////////////////////////////
	
	/// \f$ k(x_1, x_2) = \left( \langle x_1, x_2 \rangle + b \right)^n \f$
	double eval(ConstInputReference x1, ConstInputReference x2) const{
		SIZE_CHECK(x1.size() == x2.size());
		double base = inner_prod(x1, x2) + m_offset;
		return std::pow(base,m_degree);
	}
	
	void eval(ConstBatchInputReference batchX1,ConstBatchInputReference batchX2, RealMatrix& result) const {
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		std::size_t sizeX1 = batchX1.size1();
		std::size_t sizeX2 = batchX2.size1();
		result.resize(sizeX1,sizeX2);
		
		//calculate the inner product
		axpy_prod(batchX1,trans(batchX2),result);
		result += blas::repeat(m_offset,sizeX1,sizeX2);
		//now do exponentiation
		if(m_degree != 1)
			noalias(result) = pow(result,m_degree);
	}
	
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result, State& state) const{
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		
		std::size_t sizeX1 = batchX1.size1();
		std::size_t sizeX2 = batchX2.size1();

		InternalState& s = state.toState<InternalState>();
		s.resize(sizeX1,sizeX2);
		result.resize(sizeX1,sizeX2);
		
		//calculate the inner product
		axpy_prod(batchX1,trans(batchX2),s.base);
		s.base += blas::repeat(m_offset,sizeX1,sizeX2);
		
		//now do exponentiation
		if(m_degree != 1)
			noalias(result) = pow(s.base,m_degree);
		else
			noalias(result) = s.base;
		noalias(s.exponentedProd) = result;
	}
	
	/////////////////////DERIVATIVES////////////////////////////////////
	
	void weightedParameterDerivative(
		ConstBatchInputReference batchX1, 
		ConstBatchInputReference batchX2, 
		RealMatrix const& coefficients,
		State const& state, 
		RealVector& gradient
	) const{
		
		std::size_t sizeX1 = batchX1.size1();
		std::size_t sizeX2 = batchX2.size1();
		gradient.resize(1);
		InternalState const& s = state.toState<InternalState>();
		
		//internal checks
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		SIZE_CHECK(s.base.size1() == sizeX1);
		SIZE_CHECK(s.base.size2() == sizeX2);
		SIZE_CHECK(s.exponentedProd.size1() == sizeX1);
		SIZE_CHECK(s.exponentedProd.size2() == sizeX2);
		
		
		//m_degree == 1 is easy
		if(m_degree == 1){//result_ij/base_ij = 1
			gradient(0) = sum(coefficients);
			if ( m_unconstrained ) 
				gradient(0) *= m_offset;
			return;
		}
		
		//we just do a looped version of the single gradient since the test for 0 is that awful..
		gradient(0) =sum(element_prod(safeDiv(s.exponentedProd,s.base,0.0),coefficients));
		gradient(0) *= m_degree;
		if ( m_unconstrained ) 
			gradient(0) *= m_offset;
	}
	
	/// \f$ k(x_1, x_2) = \left( \langle x_1, x_2 \rangle + b \right)^n \f$
	/// <br/>
	/// \f$ \frac{\partial k(x_1, x_2)}{\partial x_1} = \left[ n \cdot (\langle x_1, x_2 \rangle + b)^{n-1} \right] \cdot x_2 \f$
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
		SIZE_CHECK(coefficientsX2.size1() == sizeX1);
		SIZE_CHECK(coefficientsX2.size2() == sizeX2);
		
		
		//again m_degree == 1 is easy, as it is for the i-th row
		//just c_i X2;
		if(m_degree == 1){
			axpy_prod(coefficientsX2,batchX2,gradient);
			return;
		}
		
		//first calculate weights(i,j) = coeff(i)*exp(i,j)/prod(i,j)
		//we have to take the usual division by 0 into account
		RealMatrix weights = element_prod(coefficientsX2,safeDiv(s.exponentedProd,s.base,0.0));
		//and the derivative of input i of batch x1 is 
		//g = sum_j m_n*weights(i,j)*x2_j
		//we now sum over j which is a matrix-matrix product
		axpy_prod(weights,batchX2,gradient);
		gradient*= m_degree;
	}
	
	void read(InArchive& ar){
		ar >> m_degree;
		ar >> m_offset;
		ar >> m_degreeIsParam;
		ar >> m_unconstrained;
	}

	void write(OutArchive& ar) const{
		ar << m_degree;
		ar << m_offset;
		ar << m_degreeIsParam;
		ar << m_unconstrained;
	}

protected:
	int m_degree;                ///< exponent n
	double m_offset;                      ///< offset b
	bool m_degreeIsParam;                 ///< is the degree a model parameter?
	bool m_unconstrained;                 ///< is the degree internally represented as exponential of the parameter?
};

typedef PolynomialKernel<> DensePolynomialKernel;
typedef PolynomialKernel<CompressedRealVector> CompressedPolynomialKernel;


}
#endif
