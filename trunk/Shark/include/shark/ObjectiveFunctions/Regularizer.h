//===========================================================================
/*!
 *  \file Regularizer.h
 *
 *  \brief Regularizer
 *
 *  \author T. Glasmachers
 *  \date 2010-2011
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#ifndef SHARK_OBJECTIVEFUNCTIONS_REGULARIZER_H
#define SHARK_OBJECTIVEFUNCTIONS_REGULARIZER_H


#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

namespace shark {


///
/// \brief One-norm of the input as an objective function
///
/// \par
/// The OneNormRegularizer is intended to be used together with other
/// objective functions within a CombinedObjectiveFunction, in order to
/// obtain a more smooth and more sparse solution.
///
class OneNormRegularizer : public AbstractObjectiveFunction<VectorSpace<double>, double>
{
public:
 	typedef RealVector SearchPointType;
 	typedef double ResultType;

 	typedef TypedFirstOrderDerivative<RealVector> FirstOrderDerivative;
 	typedef TypedSecondOrderDerivative<RealVector,RealMatrix> SecondOrderDerivative;

	/// Constructor
	OneNormRegularizer()
	{
		m_name = "OneNormRegularizer";
		m_features|=HAS_FIRST_DERIVATIVE;
		m_features|=HAS_SECOND_DERIVATIVE;
	}

	void setMask(const RealVector& mask){
		m_mask = mask;
	}
	const RealVector& mask()const{
		return m_mask;
	}
	/// Evaluates the objective function.
	double eval( RealVector const& input ) const{
		if(m_mask.empty()){
			return norm_1(input);
		}
		else
		{
			return norm_1(element_prod(input,m_mask));
		}
	}

	/// Evaluates the objective function
	/// and calculates its gradient.
	double evalDerivative( RealVector const& input, FirstOrderDerivative & derivative ) const {
		unsigned int i, ic = input.size();
		derivative.m_gradient.resize(ic);
		if(m_mask.empty()){
			for (i=0; i<ic; i++){
				derivative.m_gradient(i) = boost::math::sign(input(i));
			}
		}
		else
		{
			SIZE_CHECK(m_mask.size() == input.size());
			for (i=0; i<ic; i++){
				derivative.m_gradient(i) = m_mask(i)*boost::math::sign(input(i));
			}
		}
		return eval(input);
	}
	double evalDerivative( RealVector const& input, SecondOrderDerivative & derivative ) const {
		unsigned int i, ic = input.size();
		derivative.m_gradient.resize(ic);
		derivative.m_hessian.resize(ic,ic);
		derivative.m_hessian.clear();
		if(m_mask.empty()){
			for (i=0; i<ic; i++){
				derivative.m_gradient(i) = boost::math::sign(input(i));
			}
		}
		else
		{
			SIZE_CHECK(m_mask.size() == input.size());
			for (i=0; i<ic; i++){
				derivative.m_gradient(i) = m_mask(i)*boost::math::sign(input(i));
			}
		}
		return eval(input);
	}
private:
	RealVector m_mask;
};


///
/// \brief Two-norm of the input as an objective function
///
/// \par
/// The TwoNormRegularizer is intended to be used together with other
/// objective functions within a CombinedObjectiveFunction, in order to
/// obtain a more smooth solution.
///
class TwoNormRegularizer : public AbstractObjectiveFunction<VectorSpace<double>, double>
{
public:
	typedef RealVector SearchPointType;
 	typedef double ResultType;

 	typedef TypedFirstOrderDerivative<RealVector> FirstOrderDerivative;
 	typedef TypedSecondOrderDerivative<RealVector,RealMatrix> SecondOrderDerivative;

	typedef AbstractObjectiveFunction<VectorSpace<double>, double> super;

	/// Constructor
	TwoNormRegularizer()
	{
		m_name = "TwoNormRegularizer";
		m_features|=HAS_FIRST_DERIVATIVE;
		m_features|=HAS_SECOND_DERIVATIVE;
	}

	/// Destructor
	virtual ~TwoNormRegularizer() {}


	/// Evaluates the objective function.
	virtual double eval( RealVector const& input ) const
	{ return 0.5 * normSqr(input); }

	/// Evaluates the objective function
	/// and calculates its gradient.
	virtual double evalDerivative( RealVector const& input, FirstOrderDerivative & derivative ) const {
		derivative.m_gradient = input;
		return 0.5 * normSqr(input);
	}

	/// Evaluates the objective function
	/// and calculates its gradient and
	/// its Hessian.
	virtual ResultType evalDerivative( const SearchPointType & input, SecondOrderDerivative & derivative )const {
		derivative.m_gradient = input;
		derivative.m_hessian = RealIdentityMatrix(input.size(),input.size());
		return 0.5 * normSqr(input);
	}
};


}
#endif // SHARK_CORE_REGULARIZER_H
