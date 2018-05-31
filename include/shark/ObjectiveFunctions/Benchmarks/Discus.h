/*!
 * 
 *
 * \brief       Convex quadratic benchmark function.
 * 
 *
 * \author      -
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_DISCUS_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_DISCUS_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Random.h>

namespace shark {namespace benchmarks{
/**
 * \brief Convex quadratic benchmark function.
 * \ingroup benchmarks
 */
struct Discus : public SingleObjectiveFunction {

	Discus(std::size_t numberOfVariables = 5,double alpha = 1.E-3) : m_alpha(alpha) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= HAS_FIRST_DERIVATIVE;
		m_numberOfVariables = numberOfVariables;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Discus"; }

	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}

	/// \brief Adjusts the number of variables if the function is scalable.
	/// \param [in] numberOfVariables The new dimension.
	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
	}

	SearchPointType proposeStartingPoint() const {
		RealVector x(numberOfVariables());

		for (std::size_t i = 0; i < x.size(); i++) {
			x(i) = random::uni(*mep_rng, 0,1);
		}
		return x;
	}

	double eval(SearchPointType const& p) const {
		m_evaluationCounter++;
		double sum =  sqr(p(0));
		for (std::size_t i = 1; i < p.size(); i++)
			sum += m_alpha * sqr(p(i));

		return sum;
	}
	double evalDerivative(SearchPointType const& p, FirstOrderDerivative & derivative ) const {
		derivative.resize(p.size());
		noalias(derivative) = (2 * m_alpha) * p;
		derivative(0) = 2 * p(0);
		return eval(p);
	}

	double alpha() const {
		return m_alpha;
	}

	void setAlpha(double alpha) {
		m_alpha = alpha;
	}

private:
	double m_alpha;
	std::size_t m_numberOfVariables;
};
}}

#endif
