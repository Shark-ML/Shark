//===========================================================================
/*!
 * 
 *
 * \brief       Regularizer
 * 
 * 
 *
 * \author      T. Glasmachers
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
//===========================================================================
#ifndef SHARK_OBJECTIVEFUNCTIONS_REGULARIZER_H
#define SHARK_OBJECTIVEFUNCTIONS_REGULARIZER_H


#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>

namespace shark {


///
/// \brief One-norm of the input as an objective function
///
/// \par
/// The OneNormRegularizer is intended to be used together with other
/// objective functions within a CombinedObjectiveFunction, in order to
/// obtain a more smooth and more sparse solution.
/// \ingroup objfunctions
template<class SearchPointType = RealVector>
class OneNormRegularizer : public AbstractObjectiveFunction< SearchPointType, double >
{
public:

	/// Constructor
	OneNormRegularizer(std::size_t numVariables = 0):m_numberOfVariables(numVariables){
		this->m_features |= this->HAS_FIRST_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "OneNormRegularizer"; }

	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}

	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
	}

	void setMask(SearchPointType const& mask){
		m_mask = mask;
	}
	SearchPointType const& mask() const{
		return m_mask;
	}
	/// Evaluates the objective function.
	double eval( SearchPointType const& input ) const{
		if(m_mask.empty()){
			return norm_1(input);
		}else{
			return norm_1(input * m_mask);
		}
	}

	/// Evaluates the objective function
	/// and calculates its gradient.
	double evalDerivative( SearchPointType const& input, SearchPointType& derivative ) const {
		SIZE_CHECK(m_mask.empty() || m_mask.size() == input.size());
		std::size_t ic = input.size();
		derivative.resize(ic);
		
		for (std::size_t i = 0; i != ic; i++){
			derivative(i) = boost::math::sign(input(i));
		}
		if(!m_mask.empty()){
			derivative *= m_mask;
		}
		return eval(input);
	}
private:
	SearchPointType m_mask;
	std::size_t m_numberOfVariables;
};


///
/// \brief Two-norm of the input as an objective function
///
/// \par
/// The TwoNormRegularizer is intended to be used together with other
/// objective functions within a CombinedObjectiveFunction, in order to
/// obtain a more smooth solution.
/// \ingroup objfunctions
template<class SearchPointType = RealVector>
class TwoNormRegularizer : public AbstractObjectiveFunction< SearchPointType, double >
{
public:
	typedef AbstractObjectiveFunction< SearchPointType, double > base_type;

	/// Constructor
	TwoNormRegularizer(std::size_t numVariables = 0):m_numberOfVariables(numVariables){
		this->m_features |=  base_type::HAS_FIRST_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "TwoNormRegularizer"; }

	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}

	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
	}
	
	void setMask(SearchPointType const& mask){
		m_mask = mask;
	}
	SearchPointType const& mask()const{
		return m_mask;
	}

	/// Evaluates the objective function.
	double eval( SearchPointType const& input ) const
	{ 
		if(m_mask.empty()){
			return 0.5*norm_sqr(input);
		}else{
			return 0.5 * sum(m_mask*sqr(input));
		}
	}

	/// Evaluates the objective function
	/// and calculates its gradient.
	double evalDerivative( SearchPointType const& input, SearchPointType & derivative ) const {
		if(m_mask.empty()){
			derivative = input;
		}else{
			derivative = m_mask * input;
		}
		return eval(input);
	}
private:
	std::size_t m_numberOfVariables;
	SearchPointType m_mask;
};


}
#endif // SHARK_CORE_REGULARIZER_H
