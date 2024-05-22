//===========================================================================
/*!
 * 
 *
 * \brief       CG
 * 
 * Conjugate-gradient method for unconstraint optimization.
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <https://shark-ml.github.io/Shark/>
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

#ifndef SHARK_ML_OPTIMIZER_CG_H
#define SHARK_ML_OPTIMIZER_CG_H

#include <shark/Algorithms/GradientDescent/AbstractLineSearchOptimizer.h>

namespace shark {
/// \brief Conjugate-gradient method for unconstrained optimization
///
/// The next CG search Direction  p_{k+1} is computed using the current gradient g_k by
/// \f$ p_{k+1} = \beta p_k - g_k \f$
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
/// \ingroup gradientopt
template<class SearchPointType = RealVector>
class CG : public AbstractLineSearchOptimizer<SearchPointType>
{
public:
	typedef typename AbstractLineSearchOptimizer<SearchPointType>::ObjectiveFunctionType ObjectiveFunctionType;
protected:
	void initModel();
	void computeSearchDirection(ObjectiveFunctionType const& objectiveFunction);
public:
	std::string name() const
	{ return "CG"; }

	//from ISerializable
	void read( InArchive & archive );
	void write( OutArchive & archive ) const;
protected:
	unsigned m_count;
};

//implementation is included in the library
extern template class CG<RealVector>;
extern template class CG<FloatVector>;
#ifdef SHARK_USE_OPENCL
extern template class CG<RealGPUVector>;
extern template class CG<FloatGPUVector>;
#endif
}

#endif
