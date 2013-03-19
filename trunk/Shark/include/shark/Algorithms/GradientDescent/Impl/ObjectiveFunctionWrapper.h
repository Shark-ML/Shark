//===========================================================================
/*!
 *  \file ObjectiveFunctionWrapper.h
 *
 *  \brief ObjectiveFunctionWrapper
 *
 *  \author O. Krause
 *  \date 2010
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

#ifndef SHARK_ML_OPTIMIZER_IMPL_OBJECTIVEFUNCTIONWRAPPER_H
#define SHARK_ML_OPTIMIZER_IMPL_OBJECTIVEFUNCTIONWRAPPER_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>
namespace shark {
///\brief A wrapper which wraps the evaluation of a model, given a set of parameters
///It can be used as glue to use errorfunctions together with the linesearch algorithms of LinAlg
class ObjectiveFunctionDerivativeWrapper
{
public:
		typedef AbstractObjectiveFunction<VectorSpace<double>,double> ObjectiveFunction;
		typedef AbstractObjectiveFunction<VectorSpace<double>,double>::FirstOrderDerivative Derivative;
		ObjectiveFunctionDerivativeWrapper(){}
		ObjectiveFunction const*& function()
		{
			return m_function;
		}
		ObjectiveFunction const* function()const
		{
			return m_function;
		}
		double operator()(const RealVector& parameter,RealVector& derivative)const {
			return m_function->evalDerivative(parameter,derivative);
		}
		double operator()(const RealVector& parameter)const {
			return m_function->eval(parameter);
		}
protected:

		ObjectiveFunction const* m_function;

};
}
#endif
