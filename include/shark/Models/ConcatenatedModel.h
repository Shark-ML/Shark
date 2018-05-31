//===========================================================================
/*!
 * 
 *
 * \brief       concatenation of two models, with type erasure
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010-2011
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

#ifndef SHARK_MODEL_CONCATENATEDMODEL_H
#define SHARK_MODEL_CONCATENATEDMODEL_H

#include <shark/Models/AbstractModel.h>
#include <boost/scoped_ptr.hpp>
#include <boost/serialization/scoped_ptr.hpp>

namespace shark {

///\brief ConcatenatedModel concatenates two models such that the output of the first model is input to the second.
///
///Sometimes a series of models is needed to generate the desired output. For example when input data needs to be 
///normalized before it can be put into the trained model. In this case, the ConcatenatedModel can be used to 
///represent this series as one model. 
///The easiest way to do is is using the operator >> of AbstractModel:
///ConcatenatedModel<InputType,OutputType> model = model1>>model2;
///InputType must be the type of input model1 receives and model2 the output of model2. The output of model1 and input
///of model2 must match. Another way of construction is calling the constructor of ConcatenatedModel using the constructor:
/// ConcatenatedModel<InputType,OutputType> model (&modell,&model2);
///warning: model1 and model2 must outlive model. When they are destroyed first, behavior is undefined.
///
/// \ingroup models
template<class VectorType>
class ConcatenatedModel: public AbstractModel<VectorType, VectorType, VectorType> {
private:
	typedef AbstractModel<VectorType, VectorType, VectorType> base_type;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::ParameterVectorType ParameterVectorType;

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ConcatenatedModel"; }
	
	
	///\brief Returns the expected shape of the input
	Shape inputShape() const{
		return m_layers.front().model->inputShape();
	}
	///\brief Returns the shape of the output
	Shape outputShape() const{
		return m_layers.back().model->outputShape();
	}


	void add(AbstractModel<VectorType, VectorType, VectorType>* layer, bool optimize){
		m_layers.push_back({layer,optimize});
		enableModelOptimization(m_layers.size()-1, optimize);//recompute capabilities
	}
	
	///\brief sets whether the parameters of the index-th model should be optimized
	///
	/// If the model has non-differentiable submodels disabling those will make
	/// the whole model differentiable.
	/// Note that the models are ordered as model0 >> model1>> model2>>...
	void enableModelOptimization(std::size_t index, bool opt){
		SIZE_CHECK(index < m_layers.size());
		m_layers[index].optimize = opt;
		this->m_features.reset();
		bool inputDerivative = true;
		bool parameterDerivative = true;
		for(std::size_t k = 0; k != m_layers.size(); ++k){
			auto const& layer = m_layers[m_layers.size() - k -1];//we iterate backwards through the layers
			if( layer.optimize && (!layer.model->hasFirstParameterDerivative() || !inputDerivative)){
				parameterDerivative = false;
			}
			if( !layer.model->hasFirstInputDerivative()){
				inputDerivative = false;
			}
		}

		if (parameterDerivative){ 
			this->m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		}
		
		if (inputDerivative){ 
			this->m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		}
	
	}
	ParameterVectorType parameterVector() const {
		ParameterVectorType params(numberOfParameters());
		std::size_t pos = 0;
		for(auto layer: m_layers){
			if(!layer.optimize) continue;
			ParameterVectorType layerParams = layer.model->parameterVector();
			noalias(subrange(params,pos,pos+layerParams.size())) = layerParams;
			pos += layerParams.size();
		}
		return params;
	}

	void setParameterVector(ParameterVectorType const& newParameters) {
		std::size_t pos = 0;
		for(auto layer: m_layers){
			if(!layer.optimize) continue;
			ParameterVectorType layerParams = subrange(newParameters,pos,pos+layer.model->numberOfParameters());
			layer.model->setParameterVector(layerParams);
			pos += layerParams.size();
		}
	}

	std::size_t numberOfParameters() const{
		std::size_t numParams = 0;
		for(auto layer: m_layers){
			if(!layer.optimize) continue;
			numParams += layer.model->numberOfParameters();
		}
		return numParams;
	}
	
	boost::shared_ptr<State> createState()const{
		InternalState* state = new InternalState;
		for(std::size_t i = 0; i != m_layers.size(); ++i){
			state->state.push_back(m_layers[i].model->createState());
			state->intermediates.push_back(BatchOutputType());
		}
		return boost::shared_ptr<State>(state);
	}
	
	BatchOutputType const& hiddenResponses(State const& state, std::size_t index)const{
		InternalState const& s = state.toState<InternalState>();
		return s.intermediates[index];
	}
	
	State const& hiddenState(State const& state, std::size_t index)const{
		InternalState const& s = state.toState<InternalState>();
		return *s.state[index];
	}

	using base_type::eval;
	void eval(BatchInputType const& patterns, BatchOutputType& outputs)const {
		BatchOutputType intermediates;
		outputs = patterns;
		for(auto layer: m_layers){
			swap(intermediates,outputs);
			layer.model->eval(intermediates,outputs);
		}
	}
	void eval(BatchInputType const& patterns, BatchOutputType& outputs, State& state)const{
		InternalState& s = state.toState<InternalState>();
		outputs = patterns;
		for(std::size_t i = 0; i != m_layers.size(); ++i){
			if(i == 0)
				m_layers[i].model->eval(patterns,s.intermediates[i], *s.state[i]);
			else
				m_layers[i].model->eval(s.intermediates[i-1],s.intermediates[i], *s.state[i]);
		}
		outputs = s.intermediates.back();
	}
	
	void weightedParameterDerivative(
		BatchInputType const& patterns,
		BatchOutputType const & outputs,
		BatchOutputType const& coefficients,
		State const& state,
		ParameterVectorType& gradient
	)const{
		InternalState const& s = state.toState<InternalState>();
		BatchOutputType inputDerivativeLast;
		BatchOutputType inputDerivative = coefficients;
		gradient.resize(numberOfParameters());
		std::size_t paramEnd = gradient.size();
		for(std::size_t k = 0; k != m_layers.size(); ++k){
			std::size_t i = m_layers.size() - k -1;//we iterate backwards through the layers
			BatchInputType const* pInput = &patterns;
			if(i != 0)
				pInput = &s.intermediates[i-1];
			
			swap(inputDerivativeLast,inputDerivative);
			//if the current layer does not need to be optimized, we just check whether we have to compute the chain rule
			if(!m_layers[i].optimize || m_layers[i].model->numberOfParameters() == 0){
				if(i != 0) //check, if we are done, the input layer does not need to compute anything
					m_layers[i].model->weightedInputDerivative(*pInput,s.intermediates[i], inputDerivativeLast, *s.state[i], inputDerivative);
			}else{
				ParameterVectorType paramDerivative;
				if(i != 0){//if we are in an intermediates layer, compute chain rule
					m_layers[i].model->weightedDerivatives(*pInput,s.intermediates[i], inputDerivativeLast, *s.state[i], paramDerivative,inputDerivative);					
				}
				else{//lowest layer only needs to compute parameter derivative
					m_layers[i].model->weightedParameterDerivative(*pInput,s.intermediates[i], inputDerivativeLast, *s.state[i], paramDerivative);
				}
				noalias(subrange(gradient,paramEnd - paramDerivative.size(),paramEnd)) = paramDerivative;
				paramEnd -= paramDerivative.size();
			}
		}
	}

	void weightedInputDerivative(
		BatchInputType const& patterns,
		BatchOutputType const & outputs,
		BatchOutputType const& coefficients,
		State const& state, 
		BatchOutputType& derivatives
	)const{
		InternalState const& s = state.toState<InternalState>();
		BatchOutputType derivativeLast;
		derivatives = coefficients;
		for(std::size_t k = 0; k != m_layers.size(); ++k){
			std::size_t i = m_layers.size() - k -1;//we iterate backwards through the layers
			
			BatchInputType const* pInput = &patterns;
			if(i != 0)
				pInput = &s.intermediates[i-1];
			
			swap(derivativeLast,derivatives);
			m_layers[i].model->weightedInputDerivative(*pInput,s.intermediates[i], derivativeLast, *s.state[i], derivatives);
		}
	}

	virtual void weightedDerivatives(
		BatchInputType const & patterns,
		BatchOutputType const & outputs,
		BatchOutputType const & coefficients,
		State const& state,
		ParameterVectorType& gradient,
		BatchInputType& inputDerivative
	)const{
		InternalState const& s = state.toState<InternalState>();
		BatchOutputType inputDerivativeLast;
		inputDerivative = coefficients;
		gradient.resize(numberOfParameters());
		std::size_t paramEnd = gradient.size();
		for(std::size_t k = 0; k != m_layers.size(); ++k){
			std::size_t i = m_layers.size() - k -1;//we iterate backwards through the layers
			BatchInputType const* pInput = &patterns;
			if(i != 0)
				pInput = &s.intermediates[i-1];
			
			swap(inputDerivativeLast,inputDerivative);
			//if the current layer does not need to be optimized, we just check whether we have to compute the chain rule
			if(!m_layers[i].optimize || m_layers[i].model->numberOfParameters() == 0){
				m_layers[i].model->weightedInputDerivative(*pInput,s.intermediates[i], inputDerivativeLast, *s.state[i], inputDerivative);
			}else{
				ParameterVectorType paramDerivative;
				m_layers[i].model->weightedDerivatives(*pInput,s.intermediates[i], inputDerivativeLast, *s.state[i], paramDerivative,inputDerivative);
				noalias(subrange(gradient,paramEnd - paramDerivative.size(),paramEnd)) = paramDerivative;
				paramEnd -= paramDerivative.size();
			}
		}
	}

	/// From ISerializable
	void read( InArchive & archive ){
		for(auto layer: m_layers){
			archive >> *layer.model;
			archive >> layer.optimize;
		}
	}

	/// From ISerializable
	void write( OutArchive & archive ) const{
		for(auto layer: m_layers){
			archive << *layer.model;
			archive << layer.optimize;
		}
	}
private:
	struct Layer{
		AbstractModel<VectorType, VectorType, VectorType>* model;
		bool optimize;
	};
	std::vector<Layer> m_layers;
	
	struct InternalState: State{
		std::vector<boost::shared_ptr<State> > state;
		std::vector<BatchOutputType> intermediates;
	};
};



///\brief Connects two AbstractModels so that the output of the first model is the input of the second.
template<class VectorType>
ConcatenatedModel<VectorType>  operator>>(
	AbstractModel<VectorType,VectorType, VectorType>& firstModel,
	AbstractModel<VectorType,VectorType, VectorType>& secondModel
){
	ConcatenatedModel<VectorType> sequence;
	sequence.add(&firstModel, true);
	sequence.add(&secondModel, true);
	return sequence;
}

template<class VectorType>
ConcatenatedModel<VectorType>  operator>>(
	ConcatenatedModel<VectorType> const& firstModel,
	AbstractModel<VectorType,VectorType, VectorType>& secondModel
){
	ConcatenatedModel<VectorType> sequence = firstModel;
	sequence.add(&secondModel, true);
	return sequence;
}


}
#endif
