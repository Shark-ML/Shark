/*!
 *  \file CG.cpp
 *
 *  \brief CG
 *
 *  Conjugate-gradient method for unconstraint optimization.
 *
 *  \author O. Krause
 *  \date 2013
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

void CG::initModel() {
	m_count = 0;
}
void CG::computeSearchDirection(){
	//after numReset conjugent gradient steps, we reset automatically to the original gradient.
	//this ensure numerical stability near the optimum.
	m_count++;
	if (m_count == m_dimension)
	{
		m_count = 0;
		m_searchDirection = -m_derivative;
		return;
	}
	
	//compute beta - see class documentation for the formula
	double gg = norm_sqr(m_derivative);
	double divisor = inner_prod(m_searchDirection,m_derivative - m_lastDerivative);
	//double divisor = norm_sqr(m_lastDerivative);
	if(gg == 0.0 || std::abs(divisor) <= 1.e-10*gg){
		m_count = 0;
		m_searchDirection -= m_derivative;
		return;
	}
	double beta = gg/divisor;
	
	//Update search direction
	m_searchDirection *= beta;
	m_searchDirection -= m_derivative;
}

//from ISerializable
void CG::read( InArchive & archive )
{
	AbstractLineSearchOptimizer::read(archive);
	archive>>m_count;
}

void CG::write( OutArchive & archive ) const
{
	AbstractLineSearchOptimizer::write(archive);
	archive <<m_count;
}