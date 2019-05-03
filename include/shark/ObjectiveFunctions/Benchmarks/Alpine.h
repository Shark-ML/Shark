/*!
 * 
 *
 * \brief       Rastrigin function
 * 
 *
 * \author      O.Krause
 * \date        2019
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_ALPINE_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_ALPINE_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Random.h>

namespace shark {namespace benchmarks{
/**
 * \brief Alpine No. 1, A heavily multi-modal benchmark function with spherical level-lines
 * 
 * The function definition is:
 *\f[
 * f(x) = \sum_i^d |x_i sin(x_i) + 0.1 * x_1|
 *\f]
 * The optimum lies at f(0)=0
 * M. Clerc, "The Swarm and the Queen, Towards a Deterministic and Adaptive Particle Swarm Optimization" 
 * IEEE Congress on Evolutionary Computation, Washington DC, USA, pp. 1951-1957, 1999.
 *
 *  \ingroup benchmarks
 */
struct Alpine : public SingleObjectiveFunction {
	
	Alpine(std::size_t numberOfVariables = 5):m_numberOfVariables(numberOfVariables) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= HAS_FIRST_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Alpine"; }

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

	double eval(SearchPointType const& x) const {
		SIZE_CHECK(x.size() == numberOfVariables());
		m_evaluationCounter++;
		return norm_1(x*sin(x)+0.1*x);
	}
	
	double evalDerivative(SearchPointType const& x, FirstOrderDerivative& derivative) const {
		SIZE_CHECK(x.size() == numberOfVariables());
		m_evaluationCounter++;
		derivative.resize(x.size());
		double result = 0.0;
		for(std::size_t i = 0; i != x.size(); ++i){
			double sx= std::sin(x(i));
			double vali = x(i) * sx +0.1 * x(i);
			double sign = vali > 0? 1.0 : -1.0;
			result += sign * vali;
			derivative(i) = sign * (sx + x(i) * cos(x(i)) + 0.1);
		}
		return result;
	}
private:
	std::size_t m_numberOfVariables;
};

}}

#endif
