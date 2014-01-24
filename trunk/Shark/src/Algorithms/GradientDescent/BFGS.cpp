/*!
 * 
 * \file        BFGS.cpp
 *
 * \brief       BFGS
 * 
 * The Broyden, Fletcher, Goldfarb, Shannon (BFGS) algorithm is a
 * quasi-Newton method for unconstrained real-valued optimization.
 * 
 * 
 *
 * \author      O. Krause 
 * \date        2010
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