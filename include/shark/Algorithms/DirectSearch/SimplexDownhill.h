//===========================================================================
/*!
 *
 *
 * \brief       Nelder-Mead Simplex Downhill Method
 *
 *
 *
 * \author      T. Glasmachers
 * \date        2015
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

#ifndef SHARK_ALGORITHMS_SIMPLEXDOWNHILL_H
#define SHARK_ALGORITHMS_SIMPLEXDOWNHILL_H


#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <boost/serialization/vector.hpp>
#include <vector>


namespace shark {

///
/// \brief Simplex Downhill Method
///
/// \par
/// The Nelder-Mead Simplex Downhill Method is a deterministic direct
/// search method. It is known to perform quite well in low dimensions,
/// at least for local search.
///
/// \par
/// The implementation of the algorithm is along the lines of the
/// Wikipedia article
/// https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
/// \ingroup singledirect
class SimplexDownhill : public AbstractSingleObjectiveOptimizer<RealVector >
{
public:
	/// \brief Default Constructor.
	SimplexDownhill()
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SimplexDownhill"; }

	//from ISerializable
	virtual void read( InArchive & archive )
	{
		archive >> m_simplex;
		archive >> m_best.point;
		archive >> m_best.value;
	}

	virtual void write( OutArchive & archive ) const
	{
		archive << m_simplex;
		archive << m_best.point;
		archive << m_best.value;
	}

	/// \brief Initialization of the optimizer.
	///
	/// The initial simplex is created is a distance of about one around the proposed starting point.
	///
	virtual void init(ObjectiveFunctionType const& objectiveFunction, SearchPointType const& startingPoint)
	{
		checkFeatures(objectiveFunction);
		size_t dim = startingPoint.size();

		// create the initial simplex
		m_best.value = 1e100;
		m_simplex = std::vector<SolutionType>(dim + 1);
		for (size_t j=0; j<=dim; j++)
		{
			RealVector p(dim);
			for (size_t i=0; i<dim; i++) p(i) = startingPoint(i) + ((i == j) ? 1.0 : -0.5);
			m_simplex[j].point = p;
			m_simplex[j].value = objectiveFunction.eval(p);
			if (m_simplex[j].value < m_best.value) m_best = m_simplex[j];
		}
	}
	using AbstractSingleObjectiveOptimizer<RealVector>::init;

	/// \brief Step of the simplex algorithm.
	void step(ObjectiveFunctionType const& objectiveFunction)
	{
		size_t dim = m_simplex.size() - 1;

		// step of the simplex algorithm
		sort(m_simplex.begin(), m_simplex.end());
		SolutionType& best = m_simplex[0];
		SolutionType& worst = m_simplex[dim];

		// compute centroid
		RealVector x0(dim, 0.0);
		for (size_t j=0; j<dim; j++) x0 += m_simplex[j].point;
		x0 /= (double)dim;

		// reflection
		SolutionType xr;
		xr.point = 2.0 * x0 - worst.point;
		xr.value = objectiveFunction(xr.point);
		if (xr.value < m_best.value) m_best = xr;   // keep track of best point
		if (best.value <= xr.value && xr.value < m_simplex[dim-1].value)
		{
			// replace worst point with reflected point
			worst = xr;
		}
		else if (xr.value < best.value)
		{
			// expansion
			SolutionType xe;
			xe.point = 3.0 * x0 - 2.0 * worst.point;
			xe.value = objectiveFunction(xe.point);
			if (xe.value < m_best.value) m_best = xe;   // keep track of best point
			if (xe.value < xr.value)
			{
				// replace worst point with expanded point
				worst = xe;
			}
			else
			{
				// replace worst point with reflected point
				worst = xr;
			}
		}
		else
		{
			// contraction
			SolutionType xc;
			xc.point = 0.5 * x0 + 0.5 * worst.point;
			xc.value = objectiveFunction(xc.point);
			if (xc.value < m_best.value) m_best = xc;   // keep track of best point
			if (xc.value < worst.value)
			{
				// replace worst point with contracted point
				worst = xc;
			}
			else
			{
				// reduction
				for (size_t j=1; j<=dim; j++)
				{
					m_simplex[j].point = 0.5 * best.point + 0.5 * m_simplex[j].point;
					m_simplex[j].value = objectiveFunction(m_simplex[j].point);
					if (m_simplex[j].value < m_best.value) m_best = m_simplex[j];   // keep track of best point
				}
			}
		}
	}

	/// \brief Read access to the current simplex.
	std::vector<SolutionType> const& simplex()
	{ return m_simplex; }

protected:
	std::vector<SolutionType> m_simplex;       ///< \brief Current simplex (algorithm state).
};


}
#endif
