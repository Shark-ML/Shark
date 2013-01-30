/*!
 *  \file CG.cpp
 *
 *  \brief CG
 *
 *  Conjugate-gradient method for unconstraint optimization.
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
#include <shark/Algorithms/GradientDescent/CG.h>

using namespace shark;

CG::CG()
{
	m_name="CG";
	m_features |= REQUIRES_VALUE;
	m_features |= REQUIRES_FIRST_DERIVATIVE;
}
void CG::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint) {
	checkFeatures(objectiveFunction);

	m_linesearch.init(objectiveFunction);
	m_best.point=startingPoint;

	m_dimension = startingPoint.size();
	m_numReset  = m_dimension;

	m_count = 0;
	
	//evaluate starting point and initialize cg iterations
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_pointDerivative);
	m_xi= m_h= m_g =-m_pointDerivative.m_gradient;
}
void CG::step(const ObjectiveFunctionType& objectiveFunction) {
	//search in the direction of the current conjugate gradient xi
	m_linesearch(m_best.point,m_best.value,m_xi,m_pointDerivative.m_gradient);

	//reevaluate the derivative at the new point
	m_best.value = objectiveFunction.evalDerivative(m_best.point, m_pointDerivative);



	//OK: what happens in the next 10 lines is a mystery to me
	//nevertheless i transfered all loops and calculations to ublas. I hope everything is correct
	m_xi = m_pointDerivative.m_gradient;
	double gg=normSqr(m_g);
	double dgg=inner_prod(m_xi+m_g,m_xi);

	if (gg == 0.0) return;		// new 10/2008: converged

	double gamma = dgg / gg;
	m_g=-m_xi;
	m_xi=m_g+gamma*m_h;
	m_h=m_xi;


	//when the search direction differs more than 90 degrees from the gradient
	//we are heading into the wrong direction and so we better reset
	gg = inner_prod(m_xi,-m_pointDerivative.m_gradient);
	if (gg <= 0.)
	{
		m_xi = m_h = m_g = -m_pointDerivative.m_gradient;
	}
	//after numReset conjugent gradient steps, we reset automatically to the original gradient.
	m_count++;
	if (m_count == m_numReset)
	{
		m_count = 0;
		m_xi= m_h = m_g = -m_pointDerivative.m_gradient;
	}

}

void CG::configure( const PropertyTree & node )
{
	PropertyTree::const_assoc_iterator it = node.find("linesearch");
	if(it!=node.not_found())
	{
		m_linesearch.configure(it->second);
	}
	m_numReset=node.get("numReset",m_dimension+1);
}
//from ISerializable
void CG::read( InArchive & archive )
{
	archive>>m_linesearch;
	archive>>m_pointDerivative.m_gradient;
	archive>>m_g;
	archive>>m_h;
	archive>>m_xi;
	archive>>m_dimension;
	archive>>m_numReset;
	archive>>m_count;
	archive>>m_best.point;
	archive>>m_best.value;
}

void CG::write( OutArchive & archive ) const
{
	archive<<m_linesearch;
	archive<<m_pointDerivative.m_gradient;
	archive<<m_g;
	archive<<m_h;
	archive<<m_xi;
	archive<<m_dimension;
	archive<<m_numReset;
	archive<<m_count;
	archive<<m_best.point;
	archive<<m_best.value;
}