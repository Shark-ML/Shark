//===========================================================================
/*!
 * 
 *
 * \brief       Normalization of a kernel function.
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

#ifndef SHARK_MODELS_KERNELS_NORMALIZED_KERNEL_H
#define SHARK_MODELS_KERNELS_NORMALIZED_KERNEL_H


#include <shark/Models/Kernels/AbstractKernelFunction.h>

namespace shark {


/// \brief Normalized version of a kernel function
///
/// For a positive definite kernel k, the normalized kernel
/// \f[ \tilde k(x, y) := \frac{k(x, y)}{\sqrt{k(x, x) \cdot k(y, y)}} \f]
/// is again a positive definite kernel function.
template<class InputType=RealVector>
class NormalizedKernel : public AbstractKernelFunction<InputType>
{
private:
	typedef AbstractKernelFunction<InputType> base_type;
	
	struct InternalState: public State{
		RealMatrix kxy;
		RealVector kxx;
		RealVector kyy;
		
		boost::shared_ptr<State> stateKxy;
		std::vector<boost::shared_ptr<State> > stateKxx;
		std::vector<boost::shared_ptr<State> > stateKyy;
		
		void resize(std::size_t sizeX1,std::size_t sizeX2, AbstractKernelFunction<InputType>* base){
			kxy.resize(sizeX1,sizeX2);
			kxx.resize(sizeX1);
			kyy.resize(sizeX2);
			stateKxx.resize(sizeX1);
			stateKyy.resize(sizeX2);
			for(std::size_t i = 0; i != sizeX1;++i){
				stateKxx[i] = base->createState();
			} 
			for(std::size_t i = 0; i != sizeX2;++i){
				stateKyy[i] = base->createState();
			}
		}
	};
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstInputReference ConstInputReference;
	
	NormalizedKernel(AbstractKernelFunction<InputType>* base) : m_base(base){
		SHARK_ASSERT( base != NULL );
		this->m_features|=base_type::IS_NORMALIZED;
		if ( m_base->hasFirstParameterDerivative() ) 
			this->m_features|=base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		if ( m_base->hasFirstInputDerivative() ) 
			this->m_features|=base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NormalizedKernel<" + m_base->name() + ">"; }

	RealVector parameterVector() const{
		return m_base->parameterVector();
	}

	void setParameterVector(RealVector const& newParameters){
		m_base->setParameterVector(newParameters);
	}

	std::size_t numberOfParameters() const{
		return m_base->numberOfParameters();
	}
	
	///\brief creates the internal state of the kernel
	boost::shared_ptr<State> createState()const{
		InternalState* state = new InternalState();
		state->stateKxx = m_base->createState();
		state->stateKyy = m_base->createState();
		state->stateKxy = m_base->createState();
		return boost::shared_ptr<State>(state);
	}

	///evaluates \f$ k(x,y) \f$
	///
	/// calculates
	/// \f[ \tilde k(x, y) := \frac{k(x, y)}{\sqrt{k(x, x) \cdot k(y, y)}} \f]
	double eval(ConstInputReference x1, ConstInputReference x2) const{
		double val = m_base->eval(x1, x2);
		val /= std::sqrt(m_base->eval(x1, x1));
		val /= std::sqrt(m_base->eval(x2, x2));
		return val;
	}
	
	
	///evaluates \f$ k(x_i,y_j) \f$ for a batch of inputs x=(x...x_n) and x=(y_1...y_m)
	///
	/// calculates
	/// \f[ \tilde k(x_i,y_j) := \frac{k(x_i,y_j)}{\sqrt{k(x_i,x_i) \cdot k(y_j, y_j)}} \f]
	void eval(BatchInputType const& batchX1, BatchInputType const& batchX2, RealMatrix& result, State& state) const{
		InternalState const& s = state.toState<InternalState>();
		
		std::size_t sizeX1 = size(batchX1);
		std::size_t sizeX2 = size(batchX2);
		s.resize(sizeX1,sizeX2,m_base);
		result.resize(sizeX1,sizeX2);
		
		s.kxy = m_base->eval(batchX1, batchX2, *s.stateKxy);
		
		
		//possible very slow way to evaluate
		//we need to calculate k(x_i,x_i) and k(y_j,y_j) for every element.
		//we do it by coping the single element in a batch of size 1 and evaluating this. 
		//the following could be made easier with an interface like 
		//m_base->eval(batchX1,s.kxx,s.statekxx);
		BatchInputType singleBatch = Batch<InputType>::createBatch(1,get(batchX1,0));
		RealMatrix singleResult(1,1);
		for(std::size_t i = 0; i != sizeX1;++i){
			get(singleBatch,i) = get(batchX1,i);
			m_base.eval(singleBatch,singleBatch,singleResult,*s.stateKxx[i]);
			s.kxx[i] = singleResult(0,0);
		}
		
		for(std::size_t j = 0; j != sizeX2;++j){
			get(singleBatch,i) = get(batchX2,j);
			m_base.eval(singleBatch,singleBatch,singleResult,*s.stateKyy[j]);
			s.kyy[j] = singleResult(0,0);
		}
		RealVector sqrtKyy=sqrt(s.kyy);
		
		//finally calculate the result
		//r(i,j) = k(x_i,x_j)/sqrt(k(x_i,x_i))*sqrt(k(x_j,kx_j))
		noalias(result) = s.kxx / outer_prod(sqrt(s.kxx),sqrtKyy);
	}
	
	///evaluates \f$ k(x,y) \f$ for a batch of inputs
	///
	/// calculates
	/// \f[ \tilde k(x, y) := \frac{k(x, y)}{\sqrt{k(x, x) \cdot k(y, y)}} \f]
	void eval(BatchInputType const& batchX1, BatchInputType const& batchX2, RealMatrix& result) const{
		std::size_t sizeX1 = size(batchX1);
		std::size_t sizeX2 = size(batchX2);

		result = m_base->eval(batchX1, batchX2);
		
		//possible very slow way to evaluate
		//we need to calculate k(x_i,x_i) and k(y_j,y_j) for every element.
		//we do it by coping the single element in a batch of size 1 and evaluating this. 
		BatchInputType singleBatch = Batch<InputType>::createBatch(1,get(batchX1,0));
		RealVector sqrtKyy(sizeX2);
		RealMatrix singleResult(1,1);
		for(std::size_t j = 0; j != sizeX2;++j){
			get(singleBatch,i) = get(batchX2,j);
			m_base.eval(singleBatch,singleBatch,singleResult,*s.stateKyy[j]);
			sqrtKyy(j) = std::sqrt(singleResult(0,0));
		}
		
		
		for(std::size_t i = 0; i != sizeX1;++i){
			get(singleBatch,i) = get(batchX1,i);
			m_base.eval(singleBatch,singleBatch,singleResult,*s.stateKxx[i]);
			double sqrtKxx = std::sqrt(singleResult(0,0));
			noalias(row(result,i)) = sqrtKxx* result / sqrtKyy;
		}
	}

	/// calculates the weighted derivate w.r.t. the parameters \f$ w \f$
	///
	/// The derivative for a single element is calculated as follows:
	///\f[ \frac{\partial \tilde k_w(x, y)}{\partial w} = \frac{k_w'(x,y)}{\sqrt{k_w(x,x) k_w(y,y)}} - \frac{k_w(x,y) \left(k_w(y,y) k_w'(x,x)+k_w(x,x) k_w'(y,y)\right)}{2 (k_w(x,x) k_w(y,y))^{3/2}} \f]
	/// where \f$ k_w'(x, y) = \partial k_w(x, y) / \partial w \f$.
	void weightedParameterDerivative(
		BatchInputType const& batchX1, 
		BatchInputType const& batchX2, 
		RealMatrix const& coefficients,
		State const& state, 
		RealVector& gradient
	) const{
		ensure_size(gradient,numberOfParameters());
		InternalState const& s = state.toState<InternalState>();
		std::size_t sizeX1 = size(batchX1);
		std::size_t sizeX2 = size(batchX2);
		
		RealMatrix weights = coefficients / sqrt(s.kxx,s.kyy);
		
		m_base->weightedParameterDerivative(batchX1,batchX2,weights,s.stateKxy,gradient);
		
		noalias(weights) = weights * s.kxy;
		RealVector wx = sum_rows(weights) / (2.0 * s.kxx);
		RealVector wy = sum_columns(weights) / (2.0 * s.kyy);
		
		//the following mess could be made easier with an interface like 
		//m_base->weightedParameterDerivative(batchX1,wx,s.statekxx,subGradient);
		//m_base->weightedParameterDerivative(batchX2,wy,s.statekyy,subGradient);
		//(calculating the weighted parameter derivative of k(x_i,x_i) or (k(y_i,y_i)
		RealVector subGradient(gradient.size());
		BatchInputType singleBatch = Batch<InputType>::createBatch(1,get(batchX1,0));
		RealMatrix coeff(1,1);
		for(std::size_t i = 0; i != sizeX1; ++i){
			get(singleBatch,0) = get(batchX1,i);
			coeff(0,0) = wx(i);
			m_base->weightedParameterDerivative(singleBatch,singleBatch,coeff,s.stateKxx[i],subGradient);
			gradient -= subGradient;
		}
		for(std::size_t j = 0; j != sizeX2; ++j){
			get(singleBatch,0) = get(batchX2,j);
			coeff(0,0) = wy(j);
			m_base->weightedParameterDerivative(singleBatch,singleBatch,coeff,s.stateKyy[j],subGradient);
			gradient -= subGradient;
		}
	}

	/// Input derivative, calculated according to the equation:
	/// <br/>
	/// \f$ \frac{\partial k(x, y)}{\partial x}
	///     \frac{\sum_i \exp(w_i) \frac{\partial k_i(x, y)}{\partial x}}
	///          {\sum_i exp(w_i)} \f$
	/// and summed up over all elements of the second batch
	void weightedInputDerivative( 
		BatchInputType const& batchX1, 
		BatchInputType const& batchX2, 
		RealMatrix const& coefficientsX2,
		State const& state, 
		BatchInputType& gradient
	) const{
		InternalState const& s = state.toState<InternalState>();
		std::size_t sizeX1 = size(batchX1);
		std::size_t sizeX2 = size(batchX2);
		
		RealMatrix weights(sizeX1,sizeX2);
		for(std::size_t i = 0; i != sizeX1;++i){
			for(std::size_t j = 0; j != sizeX2; ++j){
				weights(i,j) = coefficientsX2(i,j)/(s.sqrtKxx[i]*s.sqrtKyy[j]);
			}
		}
		
		m_base->weightedInputDerivative(batchX1,batchX2,weights,s.stateKxy,gradient);
		
		noalias(weights) = weights * s.kxy;
		RealVector wx = sum_rows(weights);
		noalias(wx) = 0.5*wx / s.kxx;
		
		//the following mess could be made easier with an interface like 
		//m_base->weightedInputDerivative(batchX1,wx,s.statekxx,subGradient);
		//(calculating the weighted input derivative of k(x_i,x_i)
		RealVector subGradient(gradient.size());
		BatchInputType singleBatch = Batch<InputType>::createBatch(1,get(batchX1,0));
		RealMatrix coeff(1,1);
		for(std::size_t i = 0; i != sizeX1; ++i){
			get(singleBatch,0) = get(batchX1,i);
			coeff(0,0) = wx(i);
			m_base->weightedInputDerivative(singleBatch,singleBatch,coeff,s.stateKxx[i],subGradient);
			gradient -= subGradient;
		}
	}

protected:
	/// kernel to normalize
	AbstractKernelFunction<InputType>* m_base;
};

typedef NormalizedKernel<> DenseNormalizedKernel;
typedef NormalizedKernel<CompressedRealVector> CompressedNormalizedKernel;


}
#endif
