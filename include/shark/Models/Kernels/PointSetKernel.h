//===========================================================================
/*!
 * 
 *
 * \brief       Applies a kernel to two pointsets and comptues the average response
 * 
 * 
 *
 * \author      T.Glasmachers, O. Krause, M. Tuma
 * \date        2010, 2011
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

#ifndef SHARK_MODELS_KERNELS_POINT_SET_KERNEL_H
#define SHARK_MODELS_KERNELS_POINT_SET_KERNEL_H


#include <shark/Models/Kernels/AbstractKernelFunction.h>

namespace shark {


/// \brief Normalized version of a kernel function
///
/// For a positive definite kernel k, the normalized kernel
/// \f[ \tilde k(x, y) := \frac{k(x, y)}{\sqrt{k(x, x) \cdot k(y, y)}} \f]
/// is again a positive definite kernel function.
/// \ingroup kernels
template<class InputType=RealVector>
class PointSetKernel : public AbstractKernelFunction<typename Batch<InputType>::type >
{
private:
	typedef AbstractKernelFunction<typename Batch<InputType>::type> base_type;
	
	struct InternalState: public State{
		std::vector<boost::shared_ptr<State> > state;
		
		void resize(std::size_t sizeX1,std::size_t sizeX2, AbstractKernelFunction<InputType> const* base){
			state.resize(sizeX1 * sizeX2);
			
			for(std::size_t i = 0; i != state.size();++i){
				state[i] = base->createState();
			} 
		}
	};
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstBatchInputReference ConstBatchInputReference;
	typedef typename base_type::ConstInputReference ConstInputReference;
	
	PointSetKernel(AbstractKernelFunction<InputType>* base) : m_base(base){
		SHARK_ASSERT( base != NULL );
		if ( m_base->hasFirstParameterDerivative() ) 
			this->m_features|=base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
	}

	std::string name() const
	{ return "PointSetKernel<" + m_base->name() + ">"; }

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
		return boost::shared_ptr<State>(state);
	}

	///evaluates \f$ k(x,y) \f$
	///
	/// calculates
	/// \f[ \tilde k(x, y) := \frac{k(x, y)}{\sqrt{k(x, x) \cdot k(y, y)}} \f]
	double eval(ConstInputReference x1, ConstInputReference x2) const{
        RealMatrix response = (*m_base)(x1,x2);
        
        return sum(response)/(response.size1() * response.size2());
	}
	
	
	void eval(ConstBatchInputReference const& batchX1, ConstBatchInputReference const& batchX2, RealMatrix& result, State& state) const{
		InternalState& s = state.toState<InternalState>();
		
		std::size_t sizeX1 = batchSize(batchX1);
		std::size_t sizeX2 = batchSize(batchX2);
        s.resize(sizeX1,sizeX2,m_base);
		result.resize(sizeX1,sizeX2);
        RealMatrix response;
        for(std::size_t i = 0; i != sizeX1; ++i){
            for(std::size_t j = 0; j != sizeX2; ++j){
                m_base->eval(getBatchElement(batchX1,i),getBatchElement(batchX2,j),response,*s.state[i*sizeX2+j]);
                result(i,j) = sum(response)/(response.size1() * response.size2());
            }
        }
	}
    
    void eval(ConstBatchInputReference const& batchX1, ConstBatchInputReference const& batchX2, RealMatrix& result) const{
		std::size_t sizeX1 = batchSize(batchX1);
		std::size_t sizeX2 = batchSize(batchX2);
		result.resize(sizeX1,sizeX2);
        
        RealMatrix response;
        for(std::size_t i = 0; i != sizeX1; ++i){
            for(std::size_t j = 0; j != sizeX2; ++j){
                m_base->eval(getBatchElement(batchX1,i),getBatchElement(batchX2,j),response);
                result(i,j) = sum(response)/(response.size1() * response.size2());
            }
        }
	}

	void weightedParameterDerivative(
		ConstBatchInputReference const& batchX1, 
		ConstBatchInputReference const& batchX2, 
		RealMatrix const& coefficients,
		State const& state, 
		RealVector& gradient
	) const{
		gradient.resize(numberOfParameters());
		InternalState const& s = state.toState<InternalState>();
		std::size_t sizeX1 = batchSize(batchX1);
		std::size_t sizeX2 = batchSize(batchX2);
		
        for(std::size_t i = 0; i != sizeX1; ++i){
            for(std::size_t j = 0; j != sizeX2; ++j){
                auto x1 = getBatchElement(batchX1,i);
                auto x2 = getBatchElement(batchX2,j);
                std::size_t size1 = batchSize(x1);
                std::size_t size2 = batchSize(x2);
                RealMatrix setCoeff(size1,size2, coefficients(i,j)/(size1 * size2));
                RealVector grad;
                m_base->weightedParameterDerivative(x1,x2,setCoeff,*s.state[i*sizeX2+j],grad);
                noalias(gradient) += grad;
            }
        }
	}

protected:
	/// kernel to normalize
	AbstractKernelFunction<InputType>* m_base;
};

}
#endif
