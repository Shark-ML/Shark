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
 * \par Copyright 1995-2015 Shark Development Team
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_NOISYERRORFUNCTION_H
#define SHARK_OBJECTIVEFUNCTIONS_NOISYERRORFUNCTION_H

#include <shark/Models/AbstractModel.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Rng/DiscreteUniform.h>
#include "Impl/FunctionWrapperBase.h"

#include <boost/scoped_ptr.hpp>

namespace shark{

namespace detail{
///\brief Baseclass for the Typewrapper of the Noisy Error Function.
class NoisyErrorFunctionWrapperBase:public FunctionWrapperBase{
protected:
	size_t m_batchSize;
public:
	void setBatchSize(size_t batchSize){
		m_batchSize = batchSize;
	}
	size_t batchSize() const{
		return m_batchSize;
	}
};
}

///\brief Error Function which only uses a random fraction of data.
///
///Conceptionally, this is the same as the normal ErrorFunction, with the only difference,
///that only a fraction of the training examples is chosen randomly out of the set and
///thus noise is introduced. This can be used to perform stochastic gradient
///descent or to introduce some noise to a problem.
template<class InputType = RealVector, class LabelType = RealVector>
class NoisyErrorFunction : public SingleObjectiveFunction
{
public:
	typedef LabeledData<InputType,LabelType> DatasetType;
	template<class OutputType>
	NoisyErrorFunction(
		DatasetType const& dataset,
		AbstractModel<InputType,OutputType>* model,
		AbstractLoss<LabelType,OutputType>* loss,
		unsigned int batchSize=1
	);
	NoisyErrorFunction(const NoisyErrorFunction<InputType,LabelType>& op1);
	NoisyErrorFunction<InputType,LabelType>& operator = (const NoisyErrorFunction<InputType,LabelType>& op1);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NoisyErrorFunction"; }

	void setBatchSize(unsigned int batchSize);
	unsigned int batchSize() const;

	void updateFeatures();
	void configure( const PropertyTree & node );

	void proposeStartingPoint( SearchPointType & startingPoint)const;
	std::size_t numberOfVariables()const;
	
	void setRegularizer(double factor, SingleObjectiveFunction* regularizer){
		m_regularizer = regularizer;
		m_regularizationStrength = factor;
	}

	double eval(const RealVector & input)const;
	ResultType evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative )const;

	template<class I,class L>
	friend void swap(const NoisyErrorFunction<I,L>& op1, const NoisyErrorFunction<I,L>& op2);
private:
	boost::scoped_ptr<detail::NoisyErrorFunctionWrapperBase> mp_wrapper;
	
	SingleObjectiveFunction* m_regularizer;
	double m_regularizationStrength;

};
}
#endif
#include "Impl/NoisyErrorFunction.inl"
