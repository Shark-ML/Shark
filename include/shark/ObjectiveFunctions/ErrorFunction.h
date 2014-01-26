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
 * \par Copyright 1995-2014 Shark Development Team
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_ERRORFUNCTION_H
#define SHARK_OBJECTIVEFUNCTIONS_ERRORFUNCTION_H


#include <shark/Models/AbstractModel.h>
#include <shark/ObjectiveFunctions/AbstractCost.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>
#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>
#include "Impl/FunctionWrapperBase.h"

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
///
/// \par
/// The class detects automatically when an AbstractLoss is used 
/// as Costfunction. In this case, it uses faster algorithms 
/// for empirical risk minimization

template<class InputType = RealVector, class LabelType = RealVector>
class ErrorFunction : public SupervisedObjectiveFunction<InputType, LabelType>
{
public:
	typedef SupervisedObjectiveFunction<InputType,LabelType> base_type;
	typedef typename base_type::SearchPointType SearchPointType;
	typedef typename base_type::ResultType ResultType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;
	typedef typename base_type::SecondOrderDerivative SecondOrderDerivative;

	template<class OutputType>
	ErrorFunction(AbstractModel<InputType,OutputType>* model, AbstractCost<LabelType, OutputType>* cost);
	template<class OutputType>
	ErrorFunction(AbstractModel<InputType,OutputType>* model, AbstractCost<LabelType, OutputType>* cost, LabeledData<InputType, LabelType> const& dataset);
	ErrorFunction(const ErrorFunction& op);
	ErrorFunction<InputType,LabelType>& operator=(const ErrorFunction<InputType,LabelType>& op);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ErrorFunction"; }

	void updateFeatures();

	void configure(const PropertyTree & node);
	void setDataset(LabeledData<InputType, LabelType> const& dataset);

	void proposeStartingPoint(SearchPointType& startingPoint) const;
	std::size_t numberOfVariables()const;

	double eval(RealVector const& input) const;
	ResultType evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative ) const;
	ResultType evalDerivative( const SearchPointType & input, SecondOrderDerivative & derivative ) const;
	
	template<class I,class L>
	friend void swap(const ErrorFunction<I,L>& op1, const ErrorFunction<I,L>& op2);

private:
	boost::scoped_ptr<detail::FunctionWrapperBase<InputType,LabelType> > mp_wrapper;
};

}
#include "Impl/ErrorFunction.inl"
#endif
