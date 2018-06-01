/*!
 * 
 *
 * \brief       error function for supervised learning
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_ERRORFUNCTION_H
#define SHARK_OBJECTIVEFUNCTIONS_ERRORFUNCTION_H


#include <shark/Models/AbstractModel.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/WeightedDataset.h>
#include "Impl/ErrorFunction.inl"

#include <boost/scoped_ptr.hpp>

namespace shark{

///
/// \brief Objective function for supervised learning
///
/// \par
/// An ErrorFunction object is an objective function for
/// learning the parameters of a model from data by means
/// of minimization of a cost function. The value of the
/// objective function is the cost of the model predictions
/// on the training data, given the targets.
/// \par
/// It supports mini-batch learning using an optional fourth argument to
/// The constructor. With mini-batch learning enabled, each iteration a random
/// batch is taken from the dataset. Thus the size of the minibatch is the size of the batches in
/// the datasets. Normalization ensures that batches of different sizes have approximately the same
/// magnitude of error and derivative.
///
///\par
/// It automatically infers the input und label type from the given dataset and the output type
/// of the model in the constructor and ensures that Model and loss match. Thus the user does
/// not need to provide the types as template parameters. 
/// \ingroup objfunctions
template<class SearchPointType = RealVector>
class ErrorFunction : public AbstractObjectiveFunction<SearchPointType, double>
{
private:
	typedef AbstractObjectiveFunction<SearchPointType, double> FunctionType;
public:
	typedef typename FunctionType::ResultType ResultType;
	typedef typename FunctionType::FirstOrderDerivative FirstOrderDerivative;

	template<class InputType, class LabelType, class OutputType>
	ErrorFunction(
		LabeledData<InputType, LabelType> const& dataset,
		AbstractModel<InputType,OutputType, SearchPointType>* model, 
		AbstractLoss<LabelType, OutputType>* loss,
		bool useMiniBatches = false
	){
		m_regularizer = nullptr;
		mp_wrapper.reset(new detail::ErrorFunctionImpl<InputType,LabelType,OutputType, SearchPointType>(dataset,model,loss, useMiniBatches));

		this -> m_features = mp_wrapper -> features();
	}
	template<class InputType, class LabelType, class OutputType>
	ErrorFunction(
		WeightedLabeledData<InputType, LabelType> const& dataset,
		AbstractModel<InputType,OutputType, SearchPointType>* model, 
		AbstractLoss<LabelType, OutputType>* loss
	){
		m_regularizer = nullptr;
		mp_wrapper.reset(new detail::WeightedErrorFunctionImpl<InputType,LabelType,OutputType, SearchPointType>(dataset,model,loss));
		this -> m_features = mp_wrapper -> features();
	}
	ErrorFunction(ErrorFunction const& op)
	:mp_wrapper(op.mp_wrapper->clone()){
		this -> m_features = mp_wrapper -> features();
	}
	ErrorFunction& operator=(ErrorFunction const& op){
		ErrorFunction copy(op);
		swap(copy.mp_wrapper,mp_wrapper);
		swap(copy.m_features, this->m_features);
		return *this;
	}

	std::string name() const
	{ return "ErrorFunction"; }
	
	void setRegularizer(double factor, FunctionType* regularizer){
		m_regularizer = regularizer;
		m_regularizationStrength = factor;
	}

	SearchPointType proposeStartingPoint()const {
		return mp_wrapper -> proposeStartingPoint();
	}
	std::size_t numberOfVariables()const{
		return mp_wrapper -> numberOfVariables();
	}
	
	void init(){
		mp_wrapper->setRng(this->mep_rng);
		mp_wrapper-> init();
	}

	double eval(SearchPointType const& input) const{
		++this->m_evaluationCounter;
		double value = mp_wrapper -> eval(input);
		if(m_regularizer)
			value += m_regularizationStrength * m_regularizer->eval(input);
		return value;
	}
	ResultType evalDerivative( SearchPointType const& input, FirstOrderDerivative & derivative ) const{
		++this->m_evaluationCounter;
		double value = mp_wrapper -> evalDerivative(input,derivative);
		if(m_regularizer){
			FirstOrderDerivative regularizerDerivative;
			value += m_regularizationStrength * m_regularizer->evalDerivative(input,regularizerDerivative);
			noalias(derivative) += m_regularizationStrength*regularizerDerivative;
		}
		return value;
	}
private:
	boost::scoped_ptr<detail::FunctionWrapperBase<SearchPointType> > mp_wrapper;
	FunctionType* m_regularizer;
	double m_regularizationStrength;
};

}

#endif
