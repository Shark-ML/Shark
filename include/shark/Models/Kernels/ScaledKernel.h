//===========================================================================
/*!
 * 
 *
 * \brief       A kernel function that wraps a member kernel and multiplies it by a scalar.
 * 
 * 
 *
 * \author      M. Tuma, T. Glasmachers, O. Krause
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

#ifndef SHARK_MODELS_KERNELS_SCALED_KERNEL_H
#define SHARK_MODELS_KERNELS_SCALED_KERNEL_H


#include <shark/Models/Kernels/AbstractKernelFunction.h>
namespace shark {


/// \brief Scaled version of a kernel function
///
/// For a positive definite kernel k, the scaled kernel
/// \f[ \tilde k(x_1, x_2) := c k(x_1, x_2) \f]
/// is again a positive definite kernel function as long as \f$ c > 0 \f$.
template<class InputType=RealVector>
class ScaledKernel : public AbstractKernelFunction<InputType>
{
private:
	typedef AbstractKernelFunction<InputType> base_type;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstInputReference ConstInputReference;
	typedef typename base_type::ConstBatchInputReference ConstBatchInputReference;

	ScaledKernel( AbstractKernelFunction<InputType>* base, double factor = 1.0 )
	: m_base( base ),
	  m_factor( factor )
	{
		RANGE_CHECK( factor > 0 );
		SHARK_ASSERT( base != NULL );
		if ( m_base->hasFirstInputDerivative() ) 
			this->m_features|=base_type::HAS_FIRST_INPUT_DERIVATIVE;
		if ( m_base->hasFirstParameterDerivative() ) 
			this->m_features|=base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ScaledKernel"; }

	RealVector parameterVector() const {
		return m_base->parameterVector();
	}
	void setParameterVector(RealVector const& newParameters) {
		m_base->setParameterVector(newParameters);
	}

	std::size_t numberOfParameters() const {
		return m_base->numberOfParameters();
	}
	
	///\brief creates the internal state of the kernel
	boost::shared_ptr<State> createState()const{
		return m_base->createState();
	}
	
	const double factor() {
		return m_factor;
	}
	void setFactor( double f ) {
		RANGE_CHECK( f > 0 );
		m_factor = f;
	}
	
	const base_type* base() const {
		return m_base;
	}

	double eval(ConstInputReference x1, ConstInputReference x2) const {
		SIZE_CHECK(x1.size() == x2.size());
		return m_factor * m_base->eval(x1, x2);
	}
	
	void eval(ConstBatchInputReference x1, ConstBatchInputReference x2, RealMatrix& result) const{
		m_base->eval(x1, x2,result);
		result *= m_factor;
	}

	void eval(ConstBatchInputReference x1, ConstBatchInputReference x2, RealMatrix& result, State& state) const{
		m_base->eval(x1, x2,result,state);
		result *= m_factor;
	}
	
	/// calculates the weighted derivate w.r.t. the parameters of the base kernel
	void weightedParameterDerivative(
		ConstBatchInputReference batchX1, 
		ConstBatchInputReference batchX2, 
		RealMatrix const& coefficients,
		State const& state, 
		RealVector& gradient
	) const{
		m_base->weightedParameterDerivative( batchX1, batchX2, coefficients, state, gradient );
		gradient *= m_factor;
	}
	/// calculates the weighted derivate w.r.t. argument \f$ x_1 \f$
	void weightedInputDerivative( 
		ConstBatchInputReference batchX1, 
		ConstBatchInputReference batchX2, 
		RealMatrix const& coefficientsX2,
		State const& state, 
		BatchInputType& gradient
	) const{
		SIZE_CHECK(coefficientsX2.size1() == shark::size(batchX1));
		SIZE_CHECK(coefficientsX2.size2() == shark::size(batchX2));
		m_base->weightedInputDerivative( batchX1, batchX2, coefficientsX2, state, gradient );
		gradient *= m_factor;
	}
	
	void read(InArchive& ar){
		ar >> m_factor;
		ar >> *m_base;
	}

	/// \brief The kernel does not serialize anything
	void write(OutArchive& ar) const{
		ar << m_factor;
		//const cast needed to prevent warning
		ar << const_cast<AbstractKernelFunction<InputType> const&>(*m_base);
	}

protected:
	AbstractKernelFunction<InputType>* m_base; ///< kernel to scale
	double m_factor; ///< scaling factor
};

typedef ScaledKernel<> DenseScaledKernel;
typedef ScaledKernel<CompressedRealVector> CompressedScaledKernel;
typedef ScaledKernel<ConstRealVectorRange> DenseScaledMklKernel;


}
#endif
