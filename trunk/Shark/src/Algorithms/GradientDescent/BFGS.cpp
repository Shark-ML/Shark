/*!
 *  \file BFGS.h
 *
 *  \brief BFGS
 *
 *  The Broyden, Fletcher, Goldfarb, Shannon (BFGS) algorithm is a
 *  quasi-Newton method for unconstrained real-valued optimization.
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
#include <shark/Algorithms/GradientDescent/BFGS.h>

using namespace shark;

BFGS::BFGS()
{
	m_features |= REQUIRES_VALUE;
	m_features |= REQUIRES_FIRST_DERIVATIVE;

}
void BFGS::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint) {
	checkFeatures(objectiveFunction);

	m_linesearch.init(objectiveFunction);
	m_parameters = startingPoint.size();
	m_best.point=startingPoint;
	m_lastDerivative.resize(m_parameters, false);

	m_hessian.resize(m_parameters, m_parameters, false);
	m_hessian.clear();
	for (size_t i = 0; i < m_parameters; ++i)
	{
		m_hessian(i, i) = 1.;
	}
	
	//evaluate starting point
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivative);
	
	m_initialStepLength = 0.0;//1.0 as step length might be very wrong.
	for (size_t i = 0; i < m_derivative.size(); ++i)
		m_initialStepLength += std::abs(m_derivative(i));
	m_initialStepLength = std::min(1.0, 1.0 / m_initialStepLength);
	
	//swap instead of copy
	swap(m_lastDerivative,m_derivative);
}

void BFGS::step(const ObjectiveFunctionType& objectiveFunction) {
	RealVector s(m_lastDerivative.size());
	fast_prod(m_hessian,m_lastDerivative,s);
	s*=-1;
	
	RealVector newPoint = m_best.point;
	m_derivative = m_lastDerivative;
	m_linesearch(newPoint,m_best.value,s,m_derivative,m_initialStepLength);
	m_initialStepLength = 1.0; 

	RealVector gamma=m_derivative-m_lastDerivative;
	RealVector delta=newPoint-m_best.point;
	double d = inner_prod(gamma,delta);

	fast_prod(m_hessian,gamma,s);

	if (d < 1e-20)
	{
		m_hessian.clear();
		for (size_t i = 0; i < m_parameters; ++i)
		{
			m_hessian(i, i) = 1.;
		}
	}
	else
	{
		double scale=inner_prod(gamma,s);
		scale = (scale / d + 1) / d;

		//swap instead of copy
		swap(m_lastDerivative,m_derivative);

		for (size_t i = 0; i < m_parameters; ++i)
		{
			for (size_t j = 0; j < m_parameters; ++j)
				m_hessian(i, j) += scale * delta(i) * delta(j)
						- (s(i) * delta(j) + s(j) * delta(i)) / d;
		}
	}
	m_best.point = newPoint;
}

//from IConfigure
void BFGS::configure( const PropertyTree & node )
{
	PropertyTree::const_assoc_iterator it = node.find("linesearch");
	if(it!=node.not_found())
	{
		m_linesearch.configure(it->second);
	}
}

//from ISerializable
void BFGS::read( InArchive & archive )
{
	archive>>m_linesearch;
	archive>>m_parameters;
	archive>>m_lastDerivative;
	archive>>m_hessian;
	archive>>m_best.point;
	archive>>m_best.value;
}

void BFGS::write( OutArchive & archive ) const
{
	archive<<m_linesearch;
	archive<<m_parameters;
	archive<<m_lastDerivative;
	archive<<m_hessian;
	archive<<m_best.point;
	archive<<m_best.value;
}