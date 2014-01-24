//===========================================================================
/*!
 * 
 * \file        CombinedObjectiveFunction.h
 *
 * \brief       CombinedObjectiveFunction
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
//===========================================================================
#ifndef SHARK_OBJECTIVEFUNCTIONS_COMBINEDOBJECTIVEFUNCTION_H
#define SHARK_OBJECTIVEFUNCTIONS_COMBINEDOBJECTIVEFUNCTION_H


#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>

namespace shark {


///
/// \brief Linear combination of objective functions
///
/// \par
/// The CombinedObjectiveFunction is a linear combination of
/// objective functions. It assumed that the result type is
/// capable of forming linear combinations with real coefficients.
///
template <typename SearchSpaceType, typename ResultT>
class CombinedObjectiveFunction : public AbstractObjectiveFunction<SearchSpaceType, ResultT>
{
public:

	typedef AbstractObjectiveFunction<SearchSpaceType, ResultT> super;
	typedef AbstractObjectiveFunction<SearchSpaceType, ResultT> element;

	/// Constructor
	CombinedObjectiveFunction(){
		this->m_features|=super::HAS_FIRST_DERIVATIVE;
		this->m_features|=super::HAS_SECOND_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CombinedObjectiveFunction"; }


	/// Adds a new objective function with a
	/// weight of one to the linear combination.
	void add(element const& e){
		add(1.0, e);
	}

	/// Adds a new objective function with
	/// a weight to the linear combination.
	void add(double weight, element const& e)
	{
		SHARK_CHECK(weight >= 0.0, "[CombinedObjectiveFunction::add] weight must be non-negative");

		m_weight.push_back(weight);
		m_elements.push_back(&e);

		if (e.features().test(element::IS_CONSTRAINED_FEATURE)) this->m_features.set(super::IS_CONSTRAINED_FEATURE);
		if (! e.features().test(element::HAS_FIRST_DERIVATIVE)) this->m_features.reset(super::HAS_FIRST_DERIVATIVE);
		if (! e.features().test(element::HAS_SECOND_DERIVATIVE)) this->m_features.reset(super::HAS_SECOND_DERIVATIVE);
	}

	/// Tests whether a point in SearchSpace is feasible,
	/// e.g., whether the constraints are fulfilled.
	bool isFeasible( const typename super::SearchPointType & input) const {
		unsigned int i, ic = m_elements.size();
		for (i=0; i<ic; i++) if (! m_elements[i]->isFeasible(input)) return false;
		return true;
	}
	
	std::size_t numberOfVariables()const{
		//todo sthis will fail if SarchPointType != Vectorspace
		return m_elements.size() == 0? 0: m_elements[0]->numberOfVariables();
	}

	/// Evaluates the objective function.
	typename super::ResultType eval( const typename super::SearchPointType & input ) const
	{
		++this->m_evaluationCounter;
		unsigned int i, ic = m_elements.size();
		typename super::ResultType ret = m_weight[0] * m_elements[0]->eval(input);
		for (i=1; i<ic; i++) ret += m_weight[i] * m_elements[i]->eval(input);
		return ret;
	}

	/// Evaluates the objective function
	/// and calculates its gradient.
	typename super::ResultType evalDerivative( const typename super::SearchPointType & input, typename super::FirstOrderDerivative & derivative ) const {
		++this->m_evaluationCounter;
		SHARK_CHECK(this->m_features.test(super::HAS_FIRST_DERIVATIVE), "[CombinedObjectiveFunction::evalDerivative] At least one of the objective functions combined is not differentiable");
		typename super::FirstOrderDerivative der;
		unsigned int i, ic = m_elements.size();
		typename super::ResultType ret = m_weight[0] * m_elements[0]->evalDerivative(input, der);
		derivative = m_weight[0] * der;
		for (i=1; i<ic; i++)
		{
			ret += m_weight[i] * m_elements[i]->evalDerivative(input, der);
			derivative += m_weight[i] * der;
		}
		return ret;
	}

	/// Evaluates the objective function
	/// and calculates its gradient and
	/// its Hessian.
	typename super::ResultType evalDerivative( const typename super::SearchPointType & input, typename super::SecondOrderDerivative & derivative )const {
		SHARK_CHECK(this->m_features.test(super::HAS_SECOND_DERIVATIVE), "[CombinedObjectiveFunction::evalDerivative] At least one of the objective functions combined is not twice differentiable");
		typename super::SecondOrderDerivative der;
		unsigned int i, ic = m_elements.size();
		typename super::ResultType ret = m_weight[0] * m_elements[0]->evalDerivative(input, der);
		derivative.m_gradient = m_weight[0] * der.m_gradient;
		derivative.m_hessian = m_weight[0] * der.m_hessian;
		for (i=1; i<ic; i++)
		{
			ret += m_weight[i] * m_elements[i]->evalDerivative(input, der);
			derivative.m_gradient += m_weight[i] * der.m_gradient;
			derivative.m_hessian += m_weight[i] * der.m_hessian;
		}
		return ret;
	}

protected:
	/// list of weights
	std::vector<double> m_weight;

	/// list of "base" objective functions
	std::vector<const element*> m_elements;
};


}
#endif // SHARK_CORE_COBINEDOBJECTIVEFUNCTION_H
