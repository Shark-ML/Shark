//===========================================================================
/*!
*  \brief base class for all models, as well as a specialized differentiable model
*
*  \author  T.Glasmachers, O. Krause
*  \date    2010
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

#ifndef SHARK_MODELS_ABSTRACTMODEL_H
#define SHARK_MODELS_ABSTRACTMODEL_H

#include <shark/Core/Flags.h>
#include <shark/Core/IParameterizable.h>
#include <shark/Core/IConfigurable.h>
#include <shark/Core/INameable.h>
#include <shark/Core/State.h>
#include <shark/Rng/Normal.h>
#include<shark/Data/Dataset.h>

namespace shark {

///\brief Base class for all Models
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
/// Models have names, can be serialized, and configured.
template<class InputTypeT, class OutputTypeT>
class AbstractModel : public IParameterizable, public IConfigurable, public INameable, public ISerializable
{
public:
	/// \brief Defines the input type of the model.
	typedef InputTypeT InputType;
	/// \brief Defines the output type of the model.
	typedef OutputTypeT OutputType;
	typedef OutputType result_type;

	/// \brief defines the batch type of the input type.
	///
	/// This ould for example be std::vector<InputType> but for example for RealVector it could be RealMatrix
	typedef typename Batch<InputType>::type BatchInputType;
	/// \brief defines the batch type of the output type
	typedef typename Batch<OutputType>::type BatchOutputType;


	AbstractModel() { }

	virtual ~AbstractModel() { }

	enum Feature {
		HAS_FIRST_PARAMETER_DERIVATIVE  = 1,
		HAS_SECOND_PARAMETER_DERIVATIVE = 2,
		HAS_FIRST_INPUT_DERIVATIVE      = 4,
		HAS_SECOND_INPUT_DERIVATIVE     = 8,
		IS_SEQUENTIAL = 16
	};
	SHARK_FEATURE_INTERFACE;

	/// \brief Returns true when the first parameter derivative is implemented.
	bool hasFirstParameterDerivative()const{
		return m_features & HAS_FIRST_PARAMETER_DERIVATIVE;
	}
	/// \brief Returns true when the second parameter derivative is implemented.
	bool hasSecondParameterDerivative()const{
		return m_features & HAS_SECOND_PARAMETER_DERIVATIVE;
	}
	/// \brief Returns true when the first input derivative is implemented.
	bool hasFirstInputDerivative()const{
		return m_features & HAS_FIRST_INPUT_DERIVATIVE;
	}
	/// \brief Returns true when the second parameter derivative is implemented.
	bool hasSecondInputDerivative()const{
		return m_features & HAS_SECOND_INPUT_DERIVATIVE;
	}
	bool isSequential()const{
		return m_features & IS_SEQUENTIAL;
	}
	
	///\brief Creates an internal state of the model.
	///
	///The state is needed when the derivatives are to be
	///calculated. Eval can store a state which is then reused to speed up
	///the calculations of the derivatives. This also allows eval to be
	///evaluated in parallel!
	virtual boost::shared_ptr<State> createState() const
	{
		if (hasFirstParameterDerivative()
				|| hasFirstInputDerivative()
				|| hasSecondParameterDerivative()
				|| hasSecondInputDerivative())
		{
			throw SHARKEXCEPTION("[AbstractModel::createState] createState must be overridden by models with derivatives");
		}
		return boost::shared_ptr<State>(new EmptyState());
	}

	/// \brief From ISerializable, reads a model from an archive.
	virtual void read( InArchive & archive ){
		m_features.read(archive);
		RealVector p;
		archive & p;
		setParameterVector(p);
	}

	/// \brief writes a model to an archive
	///
	/// the default implementation just saves the parameters, not the structure!
	virtual void write( OutArchive & archive ) const{
		m_features.write(archive);
		RealVector p = parameterVector();
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
	/// \param patterns  inputs of the model
	/// \param outputs   predictions or response of the model to every pattern
	//  \param state     stores intermediate values during the computation of eval for later re-use, in particular for gradient computation
	virtual void eval(BatchInputType const & patterns, BatchOutputType& outputs, State& state) const = 0;

	/// \brief  Standard interface for evaluating the response of the model to a single pattern.
	///
	/// \param pattern the input of the model
	/// \param output the prediction or response of the model to the pattern
	virtual void eval(InputType const & pattern, OutputType& output)const{
		BatchInputType patternBatch=Batch<InputType>::createBatch(pattern);
		get(patternBatch,0) = pattern;
		BatchOutputType outputBatch;
		eval(patternBatch,outputBatch);
		output = get(outputBatch,0);
	}

	/// \brief Model evaluation as an operator for a whole dataset. This is a convenience function
	///
	/// \param patterns the input of the model
	/// \returns the responses of the model
	Data<OutputType> operator()(Data<InputType> const& patterns)const{
		int batches = (int) patterns.numberOfBatches();
		Data<OutputType> result(batches);
		SHARK_PARALLEL_FOR(int i = 0; i < batches; ++i)
			result.batch(i)= (*this)(patterns.batch(i));
		return result;
		//return transform(patterns,*this);//todo this leads to compiler errors.
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
	/// \param  derivative    the calculated derivative as sum over all derivates of all patterns
	virtual void weightedParameterDerivative(
		BatchInputType const & pattern,
		BatchOutputType const & coefficients,
		State const& state,
		RealVector& derivative
	)const{
		SHARK_FEATURE_EXCEPTION(HAS_FIRST_PARAMETER_DERIVATIVE);
	}

	/// \brief calculates the weighted sum of derivatives w.r.t the parameters
	///
	/// \param  pattern       the patterns to evaluate
	/// \param  coefficients  the coefficients which are used to calculate the weighted sum for every pattern
	/// \param  errorHessian  the second derivative of the error function for every pattern
	/// \param state intermediate results stored by eval to sped up calculations of the derivatives
	/// \param  derivative    the calculated derivative as sum over all derivates of all patterns
	/// \param  hessian       the calculated hessian as sum over all derivates of all patterns
	virtual void weightedParameterDerivative(
		BatchInputType const & pattern,
		BatchOutputType const & coefficients,
		Batch<RealMatrix>::type const & errorHessian,//maybe a batch of matrices is bad?,
		State const& state,
		RealVector& derivative,
		RealMatrix& hessian
	)const{
		SHARK_FEATURE_EXCEPTION(HAS_SECOND_PARAMETER_DERIVATIVE);
	}

	///\brief calculates the weighted sum of derivatives w.r.t the inputs
	///
	/// \param  pattern       the patterns to evaluate
	/// \param  coefficients  the coefficients which are used to calculate the weighted sum for every pattern
	/// \param state intermediate results stored by eval to sped up calculations of the derivatives
	/// \param  derivative    the calculated derivative for every pattern
	virtual void weightedInputDerivative(
		BatchInputType const & pattern,
		BatchOutputType const & coefficients,
		State const& state,
		BatchInputType& derivative
	)const{
		SHARK_FEATURE_EXCEPTION(HAS_FIRST_INPUT_DERIVATIVE);
	}

	///\brief calculates the weighted sum of derivatives w.r.t the inputs
	///
	/// \param pattern       the pattern to evaluate
	/// \param coefficients  the coefficients which are used to calculate the weighted sum
	/// \param  errorHessian  the second derivative of the error function for every pattern
	/// \param state intermediate results stored by eval to sped up calculations of the derivatives
	/// \param derivative      the calculated derivative for every pattern
	/// \param hessian       the calculated hessian for every pattern
	virtual void weightedInputDerivative(
		BatchInputType const & pattern,
		BatchOutputType const & coefficients,
		typename Batch<RealMatrix>::type const & errorHessian,
		State const& state,
		RealMatrix& derivative,
		Batch<RealMatrix>::type& hessian
	)const{
		SHARK_FEATURE_EXCEPTION(HAS_SECOND_INPUT_DERIVATIVE);
	}
	///\brief calculates weighted input and parameter derivative at the same time
	///
	/// Sometimes, both derivatives are needed at the same time. But sometimes, when calculating the
	/// weighted parameter derivative, the input derivative can be calculated for free. This is for example true for
	/// the feed-forward neural networks. However, there exists the obvious default implementation to just calculate
	/// the derivatives one after another.
	/// \param pattern       the pattern to evaluate
	/// \param coefficients  the coefficients which are used to calculate the weighted sum
	/// \param state intermediate results stored by eval to sped up calculations of the derivatives
	/// \param parameterDerivative  the calculated parameter derivative as sum over all derivates of all patterns
	/// \param inputDerivative    the calculated derivative for every pattern
	virtual void weightedDerivatives(
		BatchInputType const & patterns,
		BatchOutputType const & coefficients,
		State const& state,
		RealVector& parameterDerivative,
		BatchInputType& inputDerivative
	)const{
		weightedParameterDerivative(patterns,coefficients,state,parameterDerivative);
		weightedInputDerivative(patterns,coefficients,state,inputDerivative);
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
template <class InputType, class OutputType>
void initRandomNormal(AbstractModel<InputType, OutputType>& model, double s)
{
	Normal<> gauss(Rng::globalRng,0, s);
	RealVector weights(model.numberOfParameters());
	std::generate(weights.begin(), weights.end(), gauss);
	model.setParameterVector(weights);
}


/// \brief Initialize model parameters uniformly at random.
///
/// \param model: model to be initialized
/// \param l: lower bound of initialization interval
/// \param h: upper bound of initialization interval
template <class InputType, class OutputType>
void initRandomUniform(AbstractModel<InputType, OutputType>& model, double l, double h)
{
	Uniform<> uni(Rng::globalRng,l, h);
	RealVector weights(model.numberOfParameters());
	std::generate(weights.begin(), weights.end(), uni);
	model.setParameterVector(weights);
}

/** @}*/

}


#endif
