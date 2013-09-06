//===========================================================================
/*!
 *  \file CG.h
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
//===========================================================================

#ifndef SHARK_ML_OPTIMIZER_CG_H
#define SHARK_ML_OPTIMIZER_CG_H

#include <shark/Algorithms/GradientDescent/AbstractLineSearchOptimizer.h>

namespace shark {
/// \brief Conjugate-gradient method for unconstrained optimization
///
/// The next CG search Direction  p_{k+1} is computed using the current gradient g_k by
/// p_{k+1} = \beta p_k - g_k
/// where beta can be computed using different formulas
/// well known is the Fletcher - Reeves method:
/// \f$ \beta = ||g_k||2/ ||g_{k-1}||^2 \f$
/// we use
///  \f$ \beta = ||g_k||^2 /<p_k,g_k-g_{k-1}> \f$
/// which is formula 5.49 in Nocedal, Wright - Numerical Optimization.
/// This formula has better numerical properties than Fletcher-Reeves for non-quadratic functions
/// while ensuring a descent direction.
/// 
/// We implement restarting to ensure quadratic convergence near the optimum as well as numerical stability
class CG : public AbstractLineSearchOptimizer{
protected:
	void initModel();
	void computeSearchDirection();
public:
	std::string name() const
	{ return "CG"; }

	//from ISerializable
	void read( InArchive & archive );
	void write( OutArchive & archive ) const;
protected:
	unsigned m_count;
};

}

#endif
