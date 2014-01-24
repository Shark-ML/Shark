/*!
 * 
 * \file        Sympart.h
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_SYMPART_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_SYMPART_H

#include <shark/Core/AbstractBoxConstraintsProvider.h>
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Traits/ObjectiveFunctionTraits.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/LinAlg/rotations.h>

namespace shark {

/// \brief Real-valued benchmark function with two objectives
//todo: fix this
/*struct Sympart : public MultiObjectiveFunction {
	typedef AbstractMultiObjectiveFunction< VectorSpace<double> > super;
	typedef ResultTypeType ResultType;
	typedef super::SearchPointType SearchPointType;

	Sympart() : m_a(1000) {
		rName() = "Sympart";
		rNoObjectives() = 2;
	}

	void setNoObjectives(unsigned int numberOfObjectives) {
		(void) numberOfObjectives;
	}

	void setNumberOfVariables(unsigned int n) {
		rNumberOfVariables() = n;

		m_rotationMatrix.resize(numberOfVariables(), numberOfVariables());
		rLowerBound().resize(numberOfVariables());
		rLowerBound() = -20;
		rUpperBound().resize(numberOfVariables());
		rUpperBound() = 20;
	}

	void init() {
		m_rotationMatrix = blas::randomRotationMatrix(m_numberOfVariables);
	}

	ResultType eval(const SearchPointType &x) {
		rNoEvaluations()++;

		std::vector<double> value(2);

		SearchPointType point = m_rotationMatrix * x;
		// SearchPointType point( x );

		const double a = 1.;
		const double b = 10.;
		const double c = 8.;
		const double c2 = c + 2*a;

		double t1 = sgn(point(0)) * ::ceil((::fabs(point(0))- c2/2.) / c2);
		double t2 = sgn(point(1)) * ::ceil((::fabs(point(1))- b/2.) / b);

		double sum1 = 0.0, sum2 = 0.0;

		for (unsigned i = 0; i < numberOfVariables(); i++) {
			if (i % 2 == 0) {
				sum1 += Shark::sqr(point(i) + a - t1*c2);
				sum2 += Shark::sqr(point(i) - a - t1*c2);
			} else {
				sum1 += Shark::sqr(point(i) - t2*b);
				sum2 += Shark::sqr(point(i) - t2*b);
			}
		}

		value[0] = sum1;
		value[1] = sum2;

		return(value);
	}

	SearchPointType proposeStartingPoint() const {
		SearchPointType result(numberOfVariables());
		for (unsigned int i = 0; i < result.size(); i++)
			result(i) = Rng::uni(-20, 20);
		return(result);
	}

	bool isFeasible(const SearchPointType &v) const {
		return(true);
	}

	void closestFeasible(SearchPointType &v) {
		(void) v;
	}

	bool isConstrained() const {
		return(false);
	}
private:
	double m_a;
	RealMatrix m_rotationMatrix;
};*/

}
#endif
