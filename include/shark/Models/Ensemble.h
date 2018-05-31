//===========================================================================
/*!
 * 
 *
 * \brief       Implements the Ensemble Model that can be used to merge predictions from weighted models
 * 
 * \author      O. Krause
 * \date        2018
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

#ifndef SHARK_MODELS_ENSEMBLE_H
#define SHARK_MODELS_ENSEMBLE_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/Classifier.h>
#include <type_traits>
namespace shark {
	
namespace detail{
template<class BaseModelType, class VectorType>
class EnsembleImpl: public AbstractModel<
	typename std::remove_pointer<BaseModelType>::type::InputType,
	VectorType,
	typename std::remove_pointer<BaseModelType>::type::ParameterVectorType
>{
public:
	typedef typename std::remove_pointer<BaseModelType>::type ModelType;
private:
	typedef AbstractModel<typename ModelType::InputType, VectorType, typename ModelType::ParameterVectorType> Base;

	// the following functions are returning a reference to the model
	// independent of whether a pointer to the model or the model itself 
	// is stored.
	ModelType& derefIfPtr(ModelType& model)const{
		return model;
	}
	ModelType const& derefIfPtr(ModelType const& model)const{
		return model;
	}
	ModelType& derefIfPtr(ModelType* model)const{
		return *model;
	}
	

	//implements the pooling operation which creates a vector from the model responses to the given patterns
	template<class T> struct tag{};

	template<class InputBatch, class T, class Device>
	void pool(InputBatch const& patterns, blas::matrix<T, blas::row_major, Device>& outputs, tag<blas::vector<T, Device> >)const{
		for(std::size_t i = 0; i != numberOfModels(); i++){
			noalias(outputs) += weight(i) * model(i)(patterns);
		}
		outputs /= sumOfWeights();
	}
	template<class InputBatch, class OutputBatch>
	void pool(InputBatch const& patterns, OutputBatch& outputs, tag<unsigned int>)const{
		blas::vector<unsigned int> responses;
		for(std::size_t i = 0; i != numberOfModels(); ++i){
			model(i).eval(patterns, responses);
			for(std::size_t p = 0; p != patterns.size1(); ++p){
				outputs(p,responses(p)) += weight(i);
			}
		}
		outputs /= sumOfWeights();
	}

	std::vector<BaseModelType> m_models;
	RealVector m_weights;
public:
	typedef typename Base::BatchInputType BatchInputType;
	typedef typename Base::BatchOutputType BatchOutputType;
	typedef typename Base::ParameterVectorType ParameterVectorType;

	ParameterVectorType parameterVector() const {
		return {};
	}
	void setParameterVector(ParameterVectorType const& param) {
		SHARK_ASSERT(param.size() == 0);
	}

	void addModel(BaseModelType const& model, double weight = 1.0){
		SHARK_RUNTIME_CHECK(weight > 0, "Weights must be positive");
		m_models.push_back(model);
		m_weights.push_back(weight);
	}
	
	/// \brief Removes all models from the ensemble
	void clearModels(){
		m_models.clear();
		m_weights.clear();
	}
	
	ModelType& model(std::size_t index){
		return derefIfPtr(m_models[index]);
	}
	
	ModelType const& model(std::size_t index)const{
		return derefIfPtr(m_models[index]);
	}
	
	/// \brief Returns the weight of the i-th model.
	double const& weight(std::size_t i)const{
		return m_weights[i];
	}
	
	/// \brief Returns the weight of the i-th model.
	double& weight(std::size_t i){
		return m_weights[i];
	}
	
	/// \brief Returns the total sum of weights used for averaging
	double sumOfWeights() const{
		return sum(m_weights);
	}
	
	/// \brief Returns the number of models.
	std::size_t numberOfModels()const{
		return m_models.size();
	}
	
	///\brief Returns the expected shape of the input
	Shape inputShape() const{
		return m_models.empty() ? Shape(): model(0).inputShape();
	}
	///\brief Returns the shape of the output
	Shape outputShape() const{
		return m_models.empty() ? Shape(): model(0).outputShape();
	}

	using Base::eval;
	void eval(BatchInputType const& patterns, BatchOutputType& outputs)const{
		outputs.resize(patterns.size1(), outputShape().numElements());
		outputs.clear();
		pool(patterns,outputs, tag<typename ModelType::OutputType>());
	}
	void eval(BatchInputType const& patterns, BatchOutputType& outputs, State&)const{
		eval(patterns,outputs);
	}
	
	void read(InArchive& archive){
		std::size_t numModels;
		archive >> numModels;
		m_models.resize(numModels);
		for(std::size_t i = 0; i != numModels; ++i){
			archive >> model(i);
		}
		archive >> m_weights;
	}
	void write(OutArchive& archive)const{
		std::size_t numModels = m_models.size();
		archive << numModels;
		for(std::size_t i = 0; i != numModels; ++i){
			archive << model(i);
		}
		archive << m_weights;
	}
};

//the following creates an ensemble base depending on whether the ensemble should be a classifier or not.

template<class ModelType, class OutputType>
struct EnsembleBase : public detail::EnsembleImpl<ModelType, OutputType>{
private:
	typedef typename std::remove_pointer<ModelType>::type::OutputType ModelOutputType;
protected:
	detail::EnsembleImpl<ModelType, OutputType>& impl(){ return *this;};
	detail::EnsembleImpl<ModelType, OutputType> const& impl() const{ return *this;};
};

//if the output type is unsigned int, this is a classifier
template<class BaseModelType>
struct EnsembleBase<BaseModelType, unsigned int>
: public Classifier<detail::EnsembleImpl<BaseModelType, typename std::remove_pointer<BaseModelType>::type::ParameterVectorType> >{
private:
	typedef typename std::remove_pointer<BaseModelType>::type::ParameterVectorType PoolingVectorType;
protected:
	detail::EnsembleImpl<BaseModelType, PoolingVectorType>& impl()
	{ return this->decisionFunction();}
	detail::EnsembleImpl<BaseModelType, PoolingVectorType> const& impl() const
	{ return this->decisionFunction();}
};

//if the OutputType is void, this is treated as choosing it as the OutputType of the model
template<class ModelType>
struct EnsembleBase<ModelType, void>
: public EnsembleBase<ModelType, typename std::remove_pointer<ModelType>::type::OutputType>{};
}

/// \brief Represents en weighted ensemble of models. 
///
/// In an ensemble, each model computes a response for an input independently. The responses are then pooled
/// to form a single label. The hope is that models in an ensemble do not produce the same type of errors
/// and thus the averaged response is more reliable. An example for this is AdaBoost, where a series
/// of weak models is trained and weighted to create one final prediction. 
///
/// There are two orthogonal aspects to consider in the Ensemble. The pooling function, which is chosen
/// based on the output type of the ensemble models, and the mapping of the output of the pooling function
/// to the model output.
/// 
/// If the ensemble consists of models returning vectors, pooling is implemented
/// using weighted averaging. If the models return class labels, those are first transformed
/// into a one-hot encoding before averaging. Thus the output can be interpreted
/// as the probability of a class label when picking a member of the emsemble randomly with probability 
/// proportional to its weights. 
///
/// The final mapping to the output is based on the OutputType template parameter, which by default
/// is the same as the output type of the model. If it is unsigned int, the Ensemble is treated as Classifier
/// with decision function being the result of the pooling function (i.e. the class with maximum response in
/// the weighted average is chosen). In this case, Essemble is derived from Classifier<>. 
/// Otherwise the weighted average is returned.
///
/// Note that there is a decision in algorihm design tot ake for classifiers:
/// We can either let each member of the Ensemble predict
/// a class-label and then pool the labels as described above, or we can create an ensemble of
/// decision functions and weight them into one decision function to produce the class-label.
/// Those approaches will lead to different results. For example if the underlying models
/// produce class probabilities, the class with the largest average probability
/// might not be the same as the class with most votes from the individual models.
///
/// Models are added using addModel.
/// The ModelType is allowed to be either a concrete model like LinearModel<>, in which
/// case a copy of each added model is stored. If the ModelType is a pointer, for example
/// AbstractModel<...>*, only pointers are stored and all added models
/// must outlive the lifetime of the ensemble. This also entails differences in serialization.
/// In the first case, the model can be serialized completely without any setup. In the second
/// case before deserializing, the models must be constructed and added.
///
/// \ingroup models
template<class ModelType, class OutputType  = void>
class Ensemble: public detail::EnsembleBase<ModelType, OutputType>{
public:
	std::string name() const
	{ return "Ensemble"; }
	
	/// \brief Adds a new model to the ensemble.
	///
	/// \param model the new model
	/// \param weight weight of the model. must be > 0
	void addModel(ModelType const& model, double weight = 1.0){
		this->impl().addModel(model,weight);
	}
	
	/// \brief Removes all models from the ensemble
	void clearModels(){
		this->impl().clearModels();
	}
	
	/// \brief Returns the number of models.
	std::size_t numberOfModels()const{
		return this->impl().numberOfModels();
	}
	
	/// \brief Returns a reference to the i-th model.
	///
	/// \param i model index.
	typename std::remove_pointer<ModelType>::type& model(std::size_t i){
		return this->impl().model(i);
	}
	/// \brief Returns a const reference to the i-th model.
	///
	/// \param i model index.
	typename std::remove_pointer<ModelType>::type const& model(std::size_t i)const{
		return this->impl().model(i);
	}
	
	/// \brief Returns the weight of the i-th model.
	///
	/// \param i model index.
	double const& weight(std::size_t i)const{
		return this->impl().weight(i);
	}
	
	/// \brief Returns the weight of the i-th model.
	///
	/// \param i model index.
	double& weight(std::size_t i){
		return this->impl().weight(i);
	}
	
	/// \brief Returns the total sum of weights used for averaging
	double sumOfWeights() const{
		return this->impl().sumOfWeights();
	}
	
};

}
#endif
