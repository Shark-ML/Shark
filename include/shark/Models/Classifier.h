//===========================================================================
/*!
 * 
 *
 * \brief       Model for conversion of real valued output to class labels
 *
 * \author      T. Glasmachers, O.Krause
 * \date        2017
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

#ifndef SHARK_MODELS_CLASSIFIER_H
#define SHARK_MODELS_CLASSIFIER_H

#include <shark/Models/AbstractModel.h>
namespace shark {

///
/// \brief Conversion of real-valued or vector valued outputs to class labels
///
/// \par
/// The Classifier is a model converting the
/// real-valued vector output of an underlying decision function to a 
/// class label 0, ..., d-1 by means of an arg-max operation.
/// The class returns the argument of the maximal
/// input component as its output. This convertson is adjusted to
/// interpret the output of a linear model, a neural network or a support vector
/// machine for multi-category classification.
///
/// In the special case that d is 1, it is assumed that the model can be represented as
/// a 2 d vector with both components having the same value but opposite sign. 
/// In consequence, a positive output of the model is interpreted as class 1, a negative as class 0.
///
/// The underlying decision function is an arbitrary model. It should
/// be default constructable and it can be accessed using decisionFunction().
/// The parameters of the Classifier are the ones of the decision function.
///
/// Optionally the model allows to set bias values which are added on the predicted
/// values of the decision function. Thus adding positive weights on a class makes it
/// more likely to be predicted. In the binary case with a single output, a positive weight
/// makes class one more likely and a negative weight class 0.
///
/// \ingroup models
template<class Model>
class Classifier : public AbstractModel<
	typename Model::InputType,
	unsigned int,
	typename Model::ParameterVectorType
>{
private:
	typedef typename Model::BatchOutputType ModelBatchOutputType;
public:
	typedef Model DecisionFunctionType;
	typedef typename Model::InputType InputType;
	typedef unsigned int OutputType;
	typedef typename Batch<InputType>::type BatchInputType;
	typedef Batch<unsigned int>::type BatchOutputType;
	typedef typename Model::ParameterVectorType ParameterVectorType;

	Classifier(){}
	Classifier(Model const& decisionFunction)
	: m_decisionFunction(decisionFunction){}

	std::string name() const
	{ return "Classifier<"+m_decisionFunction.name()+">"; }
	
	ParameterVectorType parameterVector() const{
		return m_decisionFunction.parameterVector();
	}

	void setParameterVector(ParameterVectorType const& newParameters){
		m_decisionFunction.setParameterVector(newParameters);
	}

	std::size_t numberOfParameters() const{
		return m_decisionFunction.numberOfParameters();
	}
	
	///\brief Returns the expected shape of the input
	Shape inputShape() const{
		return m_decisionFunction.inputShape();
	}
	///\brief Returns the shape of the output
	///
	/// For the classifier, Shape is a number representing the number of classes.
	Shape outputShape() const{
		return m_decisionFunction.outputShape().flatten();
	}
	
	RealVector const& bias()const{
		return m_bias;
	}
	RealVector& bias(){
		return m_bias;
	}
	
	/// \brief Return the decision function
	Model const& decisionFunction()const{
		return m_decisionFunction;
	}
	
	/// \brief Return the decision function
	Model& decisionFunction(){
		return m_decisionFunction;
	}
	
	void eval(BatchInputType const& input, BatchOutputType& output)const{
		SIZE_CHECK(m_bias.empty() || m_decisionFunction.outputShape().numElements() == m_bias.size());
		ModelBatchOutputType modelResult;
		m_decisionFunction.eval(input,modelResult);
		std::size_t batchSize = modelResult.size1();
		output.resize(batchSize);
		if(modelResult.size2()== 1){
			double bias = m_bias.empty()? 0.0 : m_bias(0);
			for(std::size_t i = 0; i != batchSize; ++i){
				output(i) = modelResult(i,0) + bias > 0.0;
			}
		}
		else{
			for(std::size_t i = 0; i != batchSize; ++i){
				if(m_bias.empty())
					output(i) = static_cast<unsigned int>(arg_max(row(modelResult,i)));
				else
					output(i) = static_cast<unsigned int>(arg_max(row(modelResult,i) + m_bias));
			}
		}
	}
	void eval(BatchInputType const& input, BatchOutputType& output, State& state)const{
		eval(input,output);
	}
	
	void eval(InputType const & pattern, OutputType& output)const{
		SIZE_CHECK(m_bias.empty() || m_decisionFunction.outputShape().numElements() == m_bias.size());
		typename Model::OutputType modelResult;
		m_decisionFunction.eval(pattern,modelResult);
		if(m_bias.empty()){
			if(modelResult.size() == 1){
				double bias = m_bias.empty()? 0.0 : m_bias(0);
				output = modelResult(0) + bias > 0.0;
			}
			else{
				if(m_bias.empty())
					output = static_cast<unsigned int>(arg_max(modelResult));
				else
					output = static_cast<unsigned int>(arg_max(modelResult + m_bias));
			}
		}
	}
	
	/// From ISerializable
	void read(InArchive& archive){
		archive >> m_decisionFunction;
		archive >> m_bias;
	}
	/// From ISerializable
	void write(OutArchive& archive) const{
		archive << m_decisionFunction;
		archive << m_bias;
	}
	
private:
	Model m_decisionFunction;
	RealVector m_bias;
};

};
#endif
