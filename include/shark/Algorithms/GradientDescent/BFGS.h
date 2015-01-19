//===========================================================================
/*!
 * 
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


#ifndef SHARK_ALGORITHMS_GRADIENTDESCENT_BFGS_H
#define SHARK_ALGORITHMS_GRADIENTDESCENT_BFGS_H

#include <shark/Algorithms/GradientDescent/AbstractLineSearchOptimizer.h>

namespace shark {

//! \brief Broyden, Fletcher, Goldfarb, Shannon algorithm for unconstraint optimization
class BFGS : public AbstractLineSearchOptimizer
{
protected:
	void initModel();
	void computeSearchDirection();
public:
	std::string name() const
	{ return "BFGS"; }

	//from ISerializable
	void read( InArchive & archive );
	void write( OutArchive & archive ) const;
protected:
	RealMatrix m_hessian;
};

}
#endif
