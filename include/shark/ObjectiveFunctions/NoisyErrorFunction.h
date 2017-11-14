/*!
 * 
 *
 * \brief       implements an error fucntion which only uses a random portion of the data for training
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_NOISYERRORFUNCTION_H
#define SHARK_OBJECTIVEFUNCTIONS_NOISYERRORFUNCTION_H

#include <shark/Models/AbstractModel.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Random.h>
#include "Impl/FunctionWrapperBase.h"

#include <boost/scoped_ptr.hpp>

namespace shark{

///\brief Error Function which only uses a random fraction of data.
///
///Conceptionally, this is the same as the normal ErrorFunction, with the only difference,
///that only a fraction of the training examples is chosen randomly out of the set and
///thus noise is introduced. This can be used to perform stochastic gradient
///descent or to introduce some noise to a problem.
///
/// Internally this is implemented by extracting a random batch from the dataset every time.
/// Thus this error function uses the batch sizes used to split the dataset.
class NoisyErrorFunction : public SingleObjectiveFunction
{
public:
	template<class InputType, class LabelType, class OutputType>
	NoisyErrorFunction(
		LabeledData<InputType,LabelType> const& dataset,
		AbstractModel<InputType,OutputType>* model,
		AbstractLoss<LabelType,OutputType>* loss
	);
	NoisyErrorFunction(NoisyErrorFunction const& op1);
	NoisyErrorFunction& operator = (NoisyErrorFunction const& op1);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NoisyErrorFunction"; }

	SearchPointType proposeStartingPoint()const{
		return mp_wrapper->proposeStartingPoint();
	}
	std::size_t numberOfVariables()const{
		return mp_wrapper->numberOfVariables();
	}
	
	void init(){
		mp_wrapper->setRng(this->mep_rng);
		mp_wrapper->init();
	}
	
	void setRegularizer(double factor, SingleObjectiveFunction* regularizer){
		m_regularizer = regularizer;
		m_regularizationStrength = factor;
	}

	double eval(RealVector const& input)const;
	ResultType evalDerivative( SearchPointType const& input, FirstOrderDerivative & derivative )const;
private:
	boost::scoped_ptr<detail::FunctionWrapperBase> mp_wrapper;
	
	SingleObjectiveFunction* m_regularizer;
	double m_regularizationStrength;

};
}
#endif
#include "Impl/NoisyErrorFunction.inl"
