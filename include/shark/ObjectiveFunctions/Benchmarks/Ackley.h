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
 * <https://shark-ml.github.io/Shark/>
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_ACKLEY_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_ACKLEY_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Random.h>

namespace shark {namespace benchmarks{
/**
 * \brief Convex quadratic benchmark function with single dominant axis
 * \ingroup benchmarks
 */
struct Ackley : public SingleObjectiveFunction {
	Ackley(std::size_t numberOfVariables = 5) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_numberOfVariables = numberOfVariables;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Ackley"; }

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
		SearchPointType x;
		x.resize(m_numberOfVariables);

		for (std::size_t i = 0; i < x.size(); i++) {
			x(i) = random::uni(*mep_rng, -10, 10);
		}
		return x;
	}

	double eval(const SearchPointType &p) const {
		m_evaluationCounter++;

		const double A = 20.;
		const double B = 0.2;
		const double C = 2* M_PI;

		std::size_t n = p.size();
		double a = 0., b = 0.;

		for (std::size_t i = 0; i < n; ++i) {
			a += p(i) * p(i);
			b += cos(C * p(i));
		}

		return -A * std::exp(-B * std::sqrt(a / n)) - std::exp(b / n) + A + M_E;
	}
private:
	std::size_t m_numberOfVariables;
};

}}

#endif
