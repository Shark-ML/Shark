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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_SALOMON_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_SALOMON_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Random.h>

namespace shark {namespace benchmarks{
/**
 * \brief Heavily multi-modal benchmark function with spherical level-lines
 * 
 * Salomon is the sum of the two-norm of the point plus a cosine of the same two-norm. this results in
 * a function with spherical level-lines. Since the two-norm is used instead of the squared two norm, the
 * gradient is not more informative when a point far away from the optimum is evaluated
 *\f[
 * f(x) = \frac{|x|}{10} + 1 - \cos(2\pi |x|)
 *\f]
 * R. Salomon, "Re-evaluating Genetic Algorithm Performance Under Coordinate Rotation of Benchmark Functions: 
 * A Survey of Some Theoretical and Practical Aspects of Genetic Algorithms," BioSystems, vol. 39, no. 3, pp. 263-278, 1996.
 *
 *  \ingroup benchmarks
 */
struct Salomon : public SingleObjectiveFunction {
	
	Salomon(std::size_t numberOfVariables = 5):m_numberOfVariables(numberOfVariables) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= HAS_FIRST_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Salomon"; }

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
		//for numeric stability, we compute (1-cos(2*pi*d))= 2*(sin(pi*d))**2
		double d = norm_sqr(x);
		return 0.1 * d+ 2*sqr(sin(M_PI*d));
	}
	
	double evalDerivative(SearchPointType const& x, FirstOrderDerivative& derivative) const {
		SIZE_CHECK(x.size() == numberOfVariables());
		m_evaluationCounter++;
		derivative.resize(x.size());
		double twopi=2*M_PI;
		double d = norm_sqr(x);
		//~ noalias(derivative) = (0.1 + twopi * sin(twopi * d )) * x/d;
		noalias(derivative) = 2*(0.1 + twopi * sin(twopi * d )) * x;
		return 0.1 * d + 2*sqr(sin(M_PI*d));
	}
private:
	std::size_t m_numberOfVariables;
};

}}

#endif
