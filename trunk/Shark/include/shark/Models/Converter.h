//===========================================================================
/*!
 *  \brief Format conversion models
 *
 *  \par
 *  This file provides a number of parameter-free models
 *  performing format conversions. The classes are intended
 *  to be used in two ways: First, they can be used to convert
 *  data stored in Set or Datasets objects to different formats.
 *  Second, they can be appended to another model by means of
 *  the ConcatenatedModel class.
 *
 *  \par
 *  The need for converter models arises in particular for
 *  classification problems. There are at least three encodings
 *  of class labels in common use. Assume there are d classes in
 *  the problem, then it is natural to use integers
 *  \f$ 0, \dots, d-1 \f$. Neural networks usually use a one-hot
 *  encoding, with a unit vector representing the class label.
 *  This encoding has the advantage that it naturally generalizes
 *  to encoding probabilities over class labels, and thus allows
 *  for objective functions like cross-entropy for model training.
 *  The third encoding in common use, both in support vector
 *  machines and neural networks, is a thresholded real value
 *  representing one out of d=2 classes. Within Shark we
 *  consistently use the data types unsigned int for the first
 *  and RealVector for the latter two cases, such that format
 *  conversions can focus on essential differences in encoding
 *  only. The models in this file allow for the most important
 *  conversions between these three encodings.
 *
 *
 *  \author  T. Glasmachers
 *  \date    2010
 *
 *  \par Copyright (c) 1999-2010:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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

#ifndef SHARK_ML_MODEL_CONVERTER_H
#define SHARK_ML_MODEL_CONVERTER_H

#include <shark/Models/AbstractModel.h>
namespace shark {


///
/// \brief Convertion of real-valued outputs to classes 0 or 1
///
/// \par
/// The ThresholdConverter is a parameter-free model converting its
/// real-valued input to a binary class label 0 or 1 by means of a
/// threshold operation. Values above the threshold result in class 1,
/// values equal to or below the threshold are converted to class 0.
/// Ths threshold takes a default value of zero, which is adjusted to
/// the case of a (linear or tanh) output neuron of a neural network,
/// and to a binary support vector machine.
///
class ThresholdConverter : public AbstractModel<RealVector, unsigned int>
{
public:
	ThresholdConverter(double threshold = 0.0);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ThresholdConverter"; }

	RealVector parameterVector() const;
	void setParameterVector(RealVector const& newParameters);
	std::size_t numberOfParameters() const;

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}
	void eval(BatchInputType const& patterns, BatchOutputType& outputs)const;
	void eval(BatchInputType const& patterns, BatchOutputType& outputs, State& state)const{
		eval(patterns,outputs);
	}
	using AbstractModel<RealVector,unsigned int>::eval;

protected:
	double m_threshold;
};

///
/// \brief Convertion of real-vector outputs to vectors of class labels 0 or 1
///
/// \par
/// The ThresholdVectorConverter is a parameter-free model converting its
/// real-valued inputs to binary class labels 0 or 1 by means of a
/// threshold operation. Values above the threshold result in class 1,
/// values equal to or below the threshold are converted to class 0.
/// Ths threshold takes a default value of zero, which is adjusted to
/// the case of a (linear or tanh) output neuron of a neural network,
/// and to a binary support vector machine.
///
class ThresholdVectorConverter : public AbstractModel<RealVector, RealVector>
{
public:
	ThresholdVectorConverter(double threshold = 0.0);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ThresholdVectorConverter"; }

	RealVector parameterVector() const;
	void setParameterVector(RealVector const& newParameters);
	std::size_t numberOfParameters() const;

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	void eval(BatchInputType const& patterns, BatchOutputType& outputs)const;
	void eval(BatchInputType const& patterns, BatchOutputType& outputs, State& state)const{
		eval(patterns,outputs);
	}
	using AbstractModel<RealVector,RealVector>::eval;

protected:
	double m_threshold;
};


///
/// \brief Convertion of real-valued outputs to classes
///
/// \par
/// The ArgMaxConverter is a model converting the
/// real-valued vector output of an underlying decision function to a 
/// class label 0, ..., d-1 by means of an arg-max operation.
/// The class returns the argument of the maximal
/// input component as its output. This convertion is adjusted to
/// interpret the output of a neural network or a support vector
/// machine for multi-category classification.
///
/// The underlying decision function is an arbitrary model. It should
/// be default constructable and it can be accessed using decisionFunction().
///
/// The parameters of the ArgMaxConverter are the ones of the decision function
template<class Model>
class ArgMaxConverter : public AbstractModel<typename Model::InputType, unsigned int>
{
private:
	typedef typename Model::BatchOutputType ModelBatchOutputType;
public:
	typedef typename Model::InputType InputType;
	typedef unsigned int OutputType;
	typedef typename Batch<InputType>::type BatchInputType;
	typedef Batch<unsigned int>::type BatchOutputType;

	ArgMaxConverter(){}

	std::string name() const
	{ return "ArgMaxConverter<"+m_decisionFunction.name()+">"; }
	
	RealVector parameterVector() const{
		return m_decisionFunction.parameterVector();
	}

	void setParameterVector(RealVector const& newParameters){
		m_decisionFunction.setParameterVector(newParameters);
	}

	std::size_t numberOfParameters() const{
		return m_decisionFunction.numberOfParameters();
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
		ModelBatchOutputType modelResult;
		m_decisionFunction.eval(input,modelResult);
		std::size_t batchSize = shark::size(modelResult);
		output.resize(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			output(i) = arg_max(get(modelResult,i));
		}
	}
	void eval(BatchInputType const& input, BatchOutputType& output, State& state)const{
		eval(input,output);
	}
	
	void eval(InputType const & pattern, OutputType& output)const{
		output = arg_max(m_decisionFunction(pattern));
	}
private:
	Model m_decisionFunction;
};

};
#endif
