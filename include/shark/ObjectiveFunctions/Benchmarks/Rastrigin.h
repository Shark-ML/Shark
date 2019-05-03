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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_RASTRIGIN_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_RASTRIGIN_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Random.h>

namespace shark {namespace benchmarks{
/**
 * \brief Heavily multi-modal benchmark function
 * 
 * Rastrigin is the sum of a simple quadratic function and a cosine-wave function which generates many local optima.
 * The global optimum lies at x=0 with function-value 0.
 * f(x)=|x|^2  + 10*sum_i^d (1-cos(2*pi*x_i))
 *  \ingroup benchmarks
 */
struct Rastrigin : public SingleObjectiveFunction {
	
	Rastrigin(std::size_t numberOfVariables = 5):m_numberOfVariables(numberOfVariables) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= HAS_FIRST_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Rastrigin"; }

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
		//for numeric stability, we compute (1-cos(2*pi*x))= 2*(sin(pi*x))**2
		return norm_sqr(x) + 20*sum(sqr(sin(M_PI*x)));
	}
	
	double evalDerivative(SearchPointType const& x, FirstOrderDerivative& derivative) const {
		SIZE_CHECK(x.size() == numberOfVariables());
		derivative.resize(x.size());
		double twopi=2*M_PI;
		noalias(derivative) = 2*x + twopi*10 * sin(twopi * x );
		return eval(x);
	}
private:
	std::size_t m_numberOfVariables;
};

}}

#endif
