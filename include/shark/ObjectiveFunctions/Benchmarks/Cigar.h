/*!
 * 
 *
 * \brief       Convex quadratic benchmark function with single dominant axis

 * 
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_CIGAR_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_CIGAR_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Random.h>

namespace shark {namespace benchmarks{
/**
 * \brief Convex quadratic benchmark function with single dominant axis
* \ingroup benchmarks
 */
struct Cigar : public SingleObjectiveFunction {

	Cigar(std::size_t numberOfVariables = 5, double alpha=1.E-3) : m_alpha(alpha) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= HAS_FIRST_DERIVATIVE;
		m_numberOfVariables = numberOfVariables;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Cigar"; }

	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}
	
	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
	}

	SearchPointType proposeStartingPoint() const {
		return blas::normal(random::globalRng(), numberOfVariables(), 0.0,1.0, blas::cpu_tag());
	}

	double eval(const SearchPointType &p) const {
		m_evaluationCounter++;
		return norm_sqr(p)+(m_alpha - 1.0)*sqr(p(0));
	}
	double evalDerivative(SearchPointType const& p, FirstOrderDerivative & derivative ) const {
		derivative.resize(p.size());
		noalias(derivative) = 2 * p;
		derivative(0) *= m_alpha;
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

#endif // SHARK_EA_CIGAR_H
