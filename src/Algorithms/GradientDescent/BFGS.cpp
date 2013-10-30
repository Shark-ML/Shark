/*!
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

void BFGS::initModel(){
	m_hessian.resize(m_dimension, m_dimension);
	m_hessian.clear();
	for (size_t i = 0; i < m_dimension; ++i)
	{
		m_hessian(i, i) = 1.;
	}
}
void BFGS::computeSearchDirection(){
	RealVector gamma = m_derivative - m_lastDerivative;
	RealVector delta = m_best.point - m_lastPoint;
	double d = inner_prod(gamma,delta);
	
	RealVector Hg(m_dimension,0.0);
	axpy_prod(m_hessian,gamma,Hg);
	
	//update hessian
	if (d < 1e-20)
	{
		initModel();
	}
	else
	{
		double scale=inner_prod(gamma,Hg);
		scale = (scale / d + 1) / d;
		
		m_hessian += scale * outer_prod(delta,delta) 
			  - (outer_prod(Hg,delta)+outer_prod(delta,Hg))/d;

	}
	
	//compute search direction
	axpy_prod(m_hessian,m_derivative,m_searchDirection);
	m_searchDirection *= -1;
}

//from ISerializable
void BFGS::read( InArchive & archive )
{
	AbstractLineSearchOptimizer::read(archive);
	archive>>m_hessian;
}

void BFGS::write( OutArchive & archive ) const
{
	AbstractLineSearchOptimizer::write(archive);
	archive<<m_hessian;
}