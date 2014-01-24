//===========================================================================
/*!
 * 
 * \file        GaussianRbfKernel.h
 *
 * \brief       Radial Gaussian kernel
 * 
 * 
 *
 * \author      T.Glasmachers, O. Krause, M. Tuma
 * \date        2010-2012
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

#ifndef SHARK_MODELS_KERNELS_GAUSSIAN_RBF_KERNEL_H
#define SHARK_MODELS_KERNELS_GAUSSIAN_RBF_KERNEL_H


#include <shark/Models/Kernels/AbstractKernelFunction.h>
namespace shark{

/// \brief Gaussian radial basis function kernel.
///
/// Gaussian radial basis function kernel
/// \f$ k(x_1, x_2) = \exp(-\gamma \cdot \| x_1 - x_2 \|^2) \f$
/// with single bandwidth parameter \f$ \gamma \f$.
/// Optionally, the parameter can be encoded as \f$ \exp(\eta) \f$,
/// which allows for unconstrained optimization.
template<class InputType=RealVector>
class GaussianRbfKernel : public AbstractKernelFunction<InputType>
{
private:
	typedef AbstractKernelFunction<InputType> base_type;
	
	struct InternalState: public State{
		RealMatrix norm2;
		RealMatrix expNorm;
		
		void resize(std::size_t sizeX1, std::size_t sizeX2){
			norm2.resize(sizeX1, sizeX2);
			expNorm.resize(sizeX1, sizeX2);
		}
	};
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstInputReference ConstInputReference;
	typedef typename base_type::ConstBatchInputReference ConstBatchInputReference;

	GaussianRbfKernel(double gamma = 1.0, bool unconstrained = false){
		m_gamma = gamma;
		m_unconstrained = unconstrained;
		this->m_features|=base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		this->m_features|=base_type::HAS_FIRST_INPUT_DERIVATIVE;
		this->m_features|=base_type::IS_NORMALIZED;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "GaussianRbfKernel"; }

	RealVector parameterVector() const{
		RealVector ret(1);
		if (m_unconstrained){
			ret(0) = std::log(m_gamma); 
		}
		else{
			ret(0) = m_gamma;
		}
		return ret;
	}
	void setParameterVector(RealVector const& newParameters){
		SHARK_CHECK(newParameters.size() == 1, "[GaussianRbfKernel::setParameterVector] invalid size of parameter vector");
		if (m_unconstrained){
			m_gamma = std::exp(newParameters(0));
		}
		else{
			SHARK_CHECK(newParameters(0) > 0.0, "[GaussianRbfKernel::setParameterVector] gamma must be positive");
			m_gamma = newParameters(0);
		}
	}

	size_t numberOfParameters() const {
		return 1;
	}

	/// Get the bandwidth parameter value.
	inline double gamma() const {
		return m_gamma;
	}

	/// Return ``standard deviation'' of Gaussian.
	inline double sigma() const{ 
		return 1. / std::sqrt(2 * m_gamma); 
	}

	/// Set the bandwidth parameter value.
	/// \throws shark::Exception if gamma <= 0.
	void setGamma(double gamma){
		SHARK_CHECK(gamma > 0.0, "[GaussianRbfKernel::setGamma] gamma must be positive");
		m_gamma = gamma;
	}

	/// Set ``standard deviation'' of Gaussian.
	inline double setSigma(double sigma) const{ 
		return m_gamma = 1. / (2 * sigma * sigma); 
	}

	/// From ISerializable.
	void read(InArchive& ar){
		ar >> m_gamma;
		ar >> m_unconstrained;
	}

	/// From ISerializable.
	void write(OutArchive& ar) const{
		ar << m_gamma;
		ar << m_unconstrained;
	}
	
	///\brief creates the internal state of the kernel
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}

	/// \brief evaluates \f$ k(x_1,x_2)\f$
	///
	/// Gaussian radial basis function kernel
	/// \f[ k(x_1, x_2) = \exp(-\gamma \cdot \| x_1 - x_2 \|^2) \f]
	double eval(ConstInputReference x1, ConstInputReference x2) const{
		SIZE_CHECK(x1.size() == x2.size());
		double norm2 = distanceSqr(x2, x1);
		double exponential = std::exp(-m_gamma * norm2);
		return exponential;
	}
	
	/// \brief evaluates \f$ k(x_1,x_2)\f$ and computes the intermediate value
	///
	/// Gaussian radial basis function kernel
	/// \f[ k(x_1, x_2) = \exp(-\gamma \cdot \| x_1 - x_2 \|^2) \f]
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result, State& state) const{
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		std::size_t sizeX1=batchX1.size1();
		std::size_t sizeX2=batchX2.size1();
		
		//configure state memory
		InternalState& s=state.toState<InternalState>();
		s.resize(sizeX1,sizeX2);

		//calculate kernel response
		noalias(s.norm2)=distanceSqr(batchX1,batchX2);
		noalias(s.expNorm)=exp(-m_gamma*s.norm2);
		result=s.expNorm;
	}
	
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result) const{
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		result = distanceSqr(batchX1,batchX2);
		noalias(result)=exp(-m_gamma*result);
	}
	
	void weightedParameterDerivative(
		ConstBatchInputReference batchX1, 
		ConstBatchInputReference batchX2, 
		RealMatrix const& coefficients,
		State const& state, 
		RealVector& gradient
	) const{
		std::size_t sizeX1=batchX1.size1();
		std::size_t sizeX2=batchX2.size1();
		InternalState const& s = state.toState<InternalState>();
		
		//internal checks
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		SIZE_CHECK(s.norm2.size1() == sizeX1);
		SIZE_CHECK(s.norm2.size2() == sizeX2);
		SIZE_CHECK(s.expNorm.size1() == sizeX1);
		SIZE_CHECK(s.expNorm.size2() == sizeX2);
		
		gradient.resize(1);
		gradient(0)= - sum(coefficients *s.expNorm * s.norm2);
		if(m_unconstrained){
			gradient *= m_gamma;
		}
	}
	void weightedInputDerivative( 
		ConstBatchInputReference batchX1, 
		ConstBatchInputReference batchX2, 
		RealMatrix const& coefficientsX2,
		State const& state,
		BatchInputType& gradient
	) const{
		std::size_t sizeX1=batchX1.size1();
		std::size_t sizeX2=batchX2.size1();
		InternalState const& s = state.toState<InternalState>();
		
		//internal checks
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		SIZE_CHECK(s.norm2.size1() == sizeX1);
		SIZE_CHECK(s.norm2.size2() == sizeX2);
		SIZE_CHECK(s.expNorm.size1() == sizeX1);
		SIZE_CHECK(s.expNorm.size2() == sizeX2);
		
		gradient.resize(sizeX1,batchX1.size2());
		RealMatrix W = coefficientsX2*s.expNorm;
		axpy_prod(W,batchX2,gradient);
		RealVector columnSum = sum_columns(coefficientsX2*s.expNorm);
		
		for(std::size_t i = 0; i != sizeX1; ++i){
			noalias(row(gradient,i)) -= columnSum(i) *  row(batchX1,i);
		}
		gradient*=2.0*m_gamma;
	}
	
	
	//~ /// \brief Evaluates \f$ \frac {\partial k(x_1,x_2)}{\partial \gamma}\f$ and \f$ \frac {\partial^2 k(x_1,x_2)}{\partial \gamma^2}\f$
	//~ ///
	//~ /// Gaussian radial basis function kernel
	//~ /// \f[ \frac {\partial k(x_1,x_2)}{\partial \gamma} = - \| x_1 - x_2 \|^2 \cdot k(x_1,x_2) \f]
	//~ /// \f[ \frac {\partial^2 k(x_1,x_2)}{\partial^2 \gamma^2} = \| x_1 - x_2 \|^4 \cdot k(x_1,x_2) \f]
	//~ void parameterDerivative(ConstInputReference x1, ConstInputReference x2, Intermediate const& intermediate, RealVector& gradient, RealMatrix& hessian) const{
		//~ SIZE_CHECK(x1.size() == x2.size());
		//~ SIZE_CHECK(intermediate.size() == numberOfIntermediateValues(x1,x2));
		//~ double norm2 = intermediate[0];
		//~ double exponential = intermediate[1];

		//~ gradient.resize(1);
		//~ hessian.resize(1, 1);
		//~ if (!m_unconstrained){
			//~ gradient(0) = -exponential * norm2;
			//~ hessian(0, 0) = -gradient(0) * norm2;
		//~ }
		//~ else{
			//~ gradient(0) = -exponential * norm2 * m_gamma;
			//~ hessian(0, 0) = -gradient(0) * norm2 * m_gamma;
		//~ }
	//~ }
	//~ /// \brief Evaluates \f$ \frac {\partial k(x_1,x_2)}{\partial x_1}\f$ and \f$ \frac {\partial^2 k(x_1,x_2)}{\partial x_1^2}\f$
	//~ ///
	//~ /// Gaussian radial basis function kernel
	//~ /// \f[ \frac {\partial k(x_1,x_2)}{\partial x_1} = -2 \gamma \left( x_1 - x_2 \right)\cdot k(x_1,x_2) \f]
	//~ /// \f[ \frac {\partial^2 k(x_1,x_2)}{\partial^2 x_1^2} =2 \gamma \left[ -k(x_1,x_2) \cdot \mathbb{I} - \frac {\partial k(x_1,x_2)}{\partial x_1} ( x_1 - x_2 )^T\right] \f]
	//~ void inputDerivative(const InputType& x1, const InputType& x2, Intermediate const& intermediate, InputType& gradient, InputMatrixType& hessian) const{
		//~ SIZE_CHECK(x1.size() == x2.size());
		//~ SIZE_CHECK(intermediate.size() == numberOfIntermediateValues(x1,x2));
		//~ double exponential = intermediate[1];
		//~ gradient.resize(x1.size());
		//~ noalias(gradient) = (2.0 * m_gamma * exponential) * (x2 - x1);
		//~ hessian.resize(x1.size(), x1.size());
		//~ noalias(hessian) = 2*m_gamma*outer_prod(gradient,x2 - x1)
						//~ - RealIdentityMatrix(x1.size())*2*m_gamma*exponential;
	//~ }

protected:
	double m_gamma;			///< kernel bandwidth parameter
	bool m_unconstrained;			///< use log storage
};

typedef GaussianRbfKernel<> DenseRbfKernel;
typedef GaussianRbfKernel<CompressedRealVector> CompressedRbfKernel;


}
#endif
