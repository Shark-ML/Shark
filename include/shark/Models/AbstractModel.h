//===========================================================================
/*!
 * \author      T.Glasmachers, O. Krause
 * \date        2010
 * \file
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

#ifndef SHARK_MODELS_ABSTRACTMODEL_H
#define SHARK_MODELS_ABSTRACTMODEL_H

/// \defgroup models Models
///
/// \brief Model classes for statistical prediction.
///
/// Models in shark define the classes that can perform statistical predictions on supplied input data.
/// Models can have different types of inputs and outputs so that they can be sued for classification and regression tasks.
#include <shark/Core/Flags.h>
#include <shark/Core/IParameterizable.h>
#include <shark/Core/INameable.h>
#include <shark/Core/State.h>
#include <shark/Core/Shape.h>
#include <shark/Core/Random.h>
#include<shark/Data/Dataset.h>

namespace shark {

///\brief Base class for all Models.
///
/// \par
/// A model is one of the three fundaments of supervised learning: model, error measure
/// and an optimization algorithm.
/// It is a concept of a function which performs a mapping \f$ x \rightarrow f_w(x)\f$.
/// In contrast to an error function it has two sets of parameters:
/// The first is the current point to map \f$x\f$, the others are the internal model parameters \f$w\f$
/// which define the mapping.
/// Often a model is used to find an optimal mapping for a problem, for example a function which
/// best fits the points of a given dataset. Therefore, AbstractModel does not only offer
/// the mapping itself, but also a set of special derivatives with respect to \f$ x \f$ and \f$ w \f$.
/// Most of the time, only the derivative with respect to \f$ w \f$ is needed, but in some special problems,
/// like finding optimal stimuli or stacking models, also the input derivative is needed.
///
///\par Models are optimized for batch processing. This means, that instead of only one data point at a time, it can
/// evaluate a big set of inputs at the same time, using optimized routines for this task.
///
/// \par
/// The derivatives are weighted, which means that the derivatives of every single output are added together
/// weighted by coefficients (see #weightedParameterDerivative). This is an optimization for the chain rule
/// which is very efficient to calculate most of the time.
///
/// \par
/// It is allowed to store intermediate values during #eval and use them to speed up calculation of
/// derivatives. Therefore it must be guaranteed that eval() is called before calculating derivatives.
/// This is no restriction, since typical error measures need the mapping itself and not only the derivative.
///
/// \par
/// Models have names and can be serialised and have parameters. The type of the parameter vector
/// can be set as third argument. By default, this is RealVector.
/// \ingroup models
template<class InputTypeT, class OutputTypeT, class ParameterVectorType=RealVector>
class AbstractModel : public IParameterizable<ParameterVectorType>, public INameable, public ISerializable
{
public:
	/// \brief Defines the input type of the model.
	typedef InputTypeT InputType;
	/// \brief Defines the output type of the model.
	typedef OutputTypeT OutputType;
	/// \brief Defines the output type of the model compatible with standard functors
	typedef OutputType result_type;

	///\brief Defines the BaseType used by the model (this type). Useful for creating derived models
	typedef AbstractModel<InputTypeT,OutputTypeT,ParameterVectorType> ModelBaseType;

	/// \brief defines the batch type of the input type.
	///
	/// This could for example be std::vector<InputType> but for example for RealVector it could be RealMatrix
	typedef typename Batch<InputType>::type BatchInputType;
	/// \brief defines the batch type of the output type
	typedef typename Batch<OutputType>::type BatchOutputType;


	AbstractModel() { }

	virtual ~AbstractModel() { }

	enum Feature {
		HAS_FIRST_PARAMETER_DERIVATIVE  = 1,
		HAS_FIRST_INPUT_DERIVATIVE      = 4,
	};
	SHARK_FEATURE_INTERFACE;

	/// \brief Returns true when the first parameter derivative is implemented.
	bool hasFirstParameterDerivative()const{
		return m_features & HAS_FIRST_PARAMETER_DERIVATIVE;
	}
	/// \brief Returns true when the first input derivative is implemented.
	bool hasFirstInputDerivative()const{
		return m_features & HAS_FIRST_INPUT_DERIVATIVE;
	}
	
	///\brief Returns the expected shape of the input.
	virtual Shape inputShape() const = 0;
	///\brief Returns the shape of the output.
	virtual Shape outputShape() const = 0;
	
	///\brief Creates an internal state of the model.
	///
	///The state is needed when the derivatives are to be
	///calculated. Eval can store a state which is then reused to speed up
	///the calculations of the derivatives. This also allows eval to be
	///evaluated in parallel!
	virtual boost::shared_ptr<State> createState() const
	{
		if (hasFirstParameterDerivative()
		|| hasFirstInputDerivative())
		{
			throw SHARKEXCEPTION("[AbstractModel::createState] createState must be overridden by models with derivatives");
		}
		return boost::shared_ptr<State>(new EmptyState());
	}

	/// \brief From ISerializable, reads a model from an archive.
	virtual void read( InArchive & archive ){
		m_features.read(archive);
		ParameterVectorType p;
		archive & p;
		this->setParameterVector(p);
	}

	/// \brief writes a model to an archive
	///
	/// the default implementation just saves the parameters, not the structure!
	virtual void write( OutArchive & archive ) const{
		m_features.write(archive);
		ParameterVectorType p = this->parameterVector();
		archive & p;
	}

	/// \brief  Standard interface for evaluating the response of the model to a batch of patterns.
	///
	/// \param patterns the inputs of the model
	/// \param outputs the predictions or response of the model to every pattern
	virtual void eval(BatchInputType const & patterns, BatchOutputType& outputs) const{
		boost::shared_ptr<State> state = createState();
		eval(patterns,outputs,*state);
	}
	
	/// \brief  Standard interface for evaluating the response of the model to a batch of patterns.
	///
	/// \param patterns the inputs of the model
	/// \param outputs the predictions or response of the model to every pattern
	/// \param state intermediate results stored by eval which can be reused for derivative computation.
	virtual void eval(BatchInputType const & patterns, BatchOutputType& outputs, State& state) const = 0;

	/// \brief  Standard interface for evaluating the response of the model to a single pattern.
	///
	/// \param pattern the input of the model
	/// \param output the prediction or response of the model to the pattern
	virtual void eval(InputType const & pattern, OutputType& output)const{
		BatchInputType patternBatch=Batch<InputType>::createBatch(pattern);
		getBatchElement(patternBatch,0) = pattern;
		BatchOutputType outputBatch;
		eval(patternBatch,outputBatch);
		output = getBatchElement(outputBatch,0);
	}

	/// \brief Model evaluation as an operator for a whole dataset. This is a convenience function
	///
	/// \param patterns the input of the model
	/// \returns the responses of the model
	Data<OutputType> operator()(Data<InputType> const& patterns)const{
		return transform(patterns,*this);
	}

	/// \brief Model evaluation as an operator for a single pattern. This is a convenience function
	///
	/// \param pattern the input of the model
	/// \returns the response of the model
	OutputType operator()(InputType const & pattern)const{
		OutputType output;
		eval(pattern,output);
		return output;
	}

	/// \brief Model evaluation as an operator for a single pattern. This is a convenience function
	///
	/// \param patterns the input of the model
	/// \returns the response of the model
	BatchOutputType operator()(BatchInputType const & patterns)const{
		BatchOutputType output;
		eval(patterns,output);
		return output;
	}

	/// \brief calculates the weighted sum of derivatives w.r.t the parameters.
	///
	/// \param  pattern       the patterns to evaluate
	/// \param  coefficients  the coefficients which are used to calculate the weighted sum for every pattern
	/// \param  state intermediate results stored by eval to speed up calculations of the derivatives
	/// \param  derivative    the calculated derivative as sum over all derivates of all patterns
	virtual void weightedParameterDerivative(
		BatchInputType const & pattern,
		BatchOutputType const& outputs,
		BatchOutputType const & coefficients,
		State const& state,
		ParameterVectorType& derivative
	)const{
		SHARK_FEATURE_EXCEPTION(HAS_FIRST_PARAMETER_DERIVATIVE);
	}

	///\brief calculates the weighted sum of derivatives w.r.t the inputs
	///
	/// \param  pattern       the patterns to evaluate
	/// \param  coefficients  the coefficients which are used to calculate the weighted sum for every pattern
	/// \param state intermediate results stored by eval to sped up calculations of the derivatives
	/// \param  derivative    the calculated derivative for every pattern
	virtual void weightedInputDerivative(
		BatchInputType const & pattern,
		BatchOutputType const& outputs,
		BatchOutputType const & coefficients,
		State const& state,
		BatchInputType& derivative
	)const{
		SHARK_FEATURE_EXCEPTION(HAS_FIRST_INPUT_DERIVATIVE);
	}

	///\brief calculates weighted input and parameter derivative at the same time
	///
	/// Sometimes, both derivatives are needed at the same time. But sometimes, when calculating the
	/// weighted parameter derivative, the input derivative can be calculated for free. This is for example true for
	/// the feed-forward neural networks. However, there exists the obvious default implementation to just calculate
	/// the derivatives one after another.
	/// \param patterns       the patterns to evaluate
	/// \param coefficients  the coefficients which are used to calculate the weighted sum
	/// \param state intermediate results stored by eval to sped up calculations of the derivatives
	/// \param parameterDerivative  the calculated parameter derivative as sum over all derivates of all patterns
	/// \param inputDerivative    the calculated derivative for every pattern
	virtual void weightedDerivatives(
		BatchInputType const & patterns,
		BatchOutputType const& outputs,
		BatchOutputType const & coefficients,
		State const& state,
		ParameterVectorType& parameterDerivative,
		BatchInputType& inputDerivative
	)const{
		weightedParameterDerivative(patterns, outputs, coefficients,state,parameterDerivative);
		weightedInputDerivative(patterns, outputs, coefficients,state,inputDerivative);
	}
};


/**
 * \ingroup shark_globals
 *
 * @{
 */

/// \brief Initialize model parameters normally distributed.
///
/// \param model: model to be initialized
/// \param s: variance of mean-free normal distribution
template <class InputType, class OutputType, class ParameterVectorType>
void initRandomNormal(AbstractModel<InputType, OutputType, ParameterVectorType>& model, double s){
	typedef typename ParameterVectorType::value_type Float;
	typedef typename ParameterVectorType::device_type Device;
	auto weights = blas::normal(random::globalRng, model.numberOfParameters(), Float(0), Float(s), Device() );
	model.setParameterVector(weights);
}


/// \brief Initialize model parameters uniformly at random.
///
/// \param model model to be initialized
/// \param lower lower bound of initialization interval
/// \param upper upper bound of initialization interval
template <class InputType, class OutputType, class ParameterVectorType>
void initRandomUniform(AbstractModel<InputType, OutputType, ParameterVectorType>& model, double lower, double upper){
	typedef typename ParameterVectorType::value_type Float;
	typedef typename ParameterVectorType::device_type Device;
	auto weights = blas::uniform(random::globalRng, model.numberOfParameters(), Float(lower), Float(upper), Device() );
	model.setParameterVector(weights);
}

/** @}*/

namespace detail{
//Required for correct shape infering of transform
template<class I, class O, class V>
struct InferShape<AbstractModel<I,O,V> >{
	static Shape infer(AbstractModel<I,O,V> const& f){return f.outputShape();}
};

}

}


#endif
