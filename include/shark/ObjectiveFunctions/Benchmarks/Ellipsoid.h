/*!
 * 
 *
 * \brief       Convex quadratic benchmark function.
 * 
 *
 * \author      T.Voss
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_ELLIPSOID_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_ELLIPSOID_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Random.h>

namespace shark {namespace benchmarks{
/**
*  \brief Convex quadratic benchmark function
*
*  The eigenvalues of the Hessian of this convex quadratic benchmark
*  function are equally distributed on logarithmic scale.
* \ingroup benchmarks
*/
struct Ellipsoid : public SingleObjectiveFunction {
	Ellipsoid(size_t numberOfVariables = 5, double alpha=1E-3) : m_alpha(alpha) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= HAS_FIRST_DERIVATIVE;
		m_features |= HAS_SECOND_DERIVATIVE;
		setNumberOfVariables(numberOfVariables);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Ellipsoid"; }
	
	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}
	
	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
		m_D.resize(m_numberOfVariables);
		double sizeMinusOne=m_numberOfVariables - 1.;
		for( std::size_t i = 0; i < m_numberOfVariables; i++ ){
			m_D(i) = std::pow( m_alpha, i / sizeMinusOne );
		}
	}

	SearchPointType proposeStartingPoint() const {
		return blas::normal(random::globalRng(), numberOfVariables(), 0.0,1.0, blas::cpu_tag());
	}

	double eval( const SearchPointType & p ) const {
		SIZE_CHECK(p.size() == m_numberOfVariables);
		m_evaluationCounter++;
		return sum(sqr(p) * m_D);
	}

	double evalDerivative( const SearchPointType & p, FirstOrderDerivative & derivative ) const {
		derivative.resize(p.size());
		noalias(derivative) = 2 * m_D * p;
		return eval(p);
	}
	double evalDerivative(const SearchPointType &p, SecondOrderDerivative &derivative)const {
		std::size_t size=p.size();
		derivative.hessian.resize(size,size);
		derivative.hessian.clear();
		diag(derivative.hessian) = 2* m_D;
		return evalDerivative(p, derivative.gradient);
	}
private:
	std::size_t m_numberOfVariables;
	double m_alpha;
	RealVector m_D;
};

}}

#endif
