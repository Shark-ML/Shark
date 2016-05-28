//===========================================================================
/*!
 * 
 *
 * \brief       Generalized Rosenbrock benchmark function 
 * 
 * This non-convex benchmark function for real-valued optimization is
 * a generalization from two to multiple dimensions of a classic
 * function first proposed in:
 * 
 * H. H. Rosenbrock. An automatic method for finding the greatest or
 * least value of a function. The Computer Journal 3: 175-184, 1960
 * 
 * 
 *
 * \author      -
 * \date        -
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_ROSENBROCK_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_ROSENBROCK_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {
/*! \brief Generalized Rosenbrock benchmark function 
*
*  This non-convex benchmark function for real-valued optimization is a
*  generalization from two to multiple dimensions of a classic
*  function first proposed in:
*
*  H. H. Rosenbrock. An automatic method for finding the greatest or
*  least value of a function. The Computer Journal 3: 175-184,
*  1960  
*/
struct Rosenbrock : public SingleObjectiveFunction {

	/// \brief Constructs the problem
	///
	/// \param dimensions number of dimensions to optimize
	/// \param initialSpread spread of the initial starting point
	Rosenbrock(std::size_t dimensions=23, double initialSpread = 1.0)
	:m_numberOfVariables(dimensions), m_initialSpread(initialSpread) {
		m_features|=CAN_PROPOSE_STARTING_POINT;
		m_features|=HAS_FIRST_DERIVATIVE;
		m_features|=HAS_SECOND_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Rosenbrock"; }

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
		RealVector x(numberOfVariables());

		for (std::size_t i = 0; i < x.size(); i++) {
			x(i) = Rng::uni( 0, m_initialSpread );
		}
		return x;
	}

	double eval( const SearchPointType & p ) const {
		m_evaluationCounter++;

		double sum = 0;

		for( std::size_t i = 0; i < p.size()-1; i++ ) {
			sum += 100*sqr( p(i+1) - sqr( p( i ) ) ) +sqr( 1. - p( i ) );
		}

		return( sum );
	}

	virtual ResultType evalDerivative( const SearchPointType & p, FirstOrderDerivative & derivative )const {
		double result = eval(p);
		size_t size = p.size();
		derivative.resize(size);
		derivative(0) = 2*( p(0) - 1 ) - 400 * ( p(1) - sqr( p(0) ) ) * p(0);
		derivative(size-1) = 200 * ( p(size - 1) - sqr( p( size - 2 ) ) ) ;
		for(size_t i=1; i != size-1; ++i){
			derivative( i ) = 2 * ( p(i) - 1 ) - 400 * (p(i+1) - sqr( p(i) ) ) * p( i )+200 * ( p( i )- sqr( p(i-1) ) );
		}
		return result;

	}

	virtual ResultType evalDerivative( const SearchPointType & p, SecondOrderDerivative & derivative )const {
		double result = eval(p);
		size_t size = p.size();
		derivative.gradient.resize(size);
		derivative.hessian.resize(size,size);
		derivative.hessian.clear();

		derivative.gradient(0) = 2*( p(0) - 1 ) - 400 * ( p(1) - sqr( p(0) ) ) * p(0);
		derivative.gradient(size-1) = 200 * ( p(size - 1) - sqr( p( size - 2 ) ) ) ;

		derivative.hessian(0,0) = 2 - 400* (p(1) - 3*sqr(p(0))) ;
		derivative.hessian(0,1) = -400 * p(0) ;

		derivative.hessian(size-1,size-1) = 200;
		derivative.hessian(size-1,size-2) = -400 * p( size - 2 );

		for(size_t i=1; i != size-1; ++i){
			derivative.gradient( i ) = 2 * ( p(i) - 1 ) - 400 * (p(i+1) - sqr( p(i) ) ) * p( i )+200 * ( p( i )- sqr( p(i-1) ) );

			derivative.hessian(i,i) = 202 - 400 * ( p(i+1) - 3 * sqr(p(i)));
			derivative.hessian(i,i+1) = - 400 * ( p(i) );
			derivative.hessian(i,i-1) = - 400 * ( p(i-1) );

		}
		return result;
	}

private:
	std::size_t m_numberOfVariables;
	double m_initialSpread;
};

}

#endif
