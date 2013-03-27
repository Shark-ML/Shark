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

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/Algorithms/GradientDescent/LineSearch.h>

namespace shark {
//! \brief Conjugate-gradient method for unconstraint optimization
class CG : public AbstractSingleObjectiveOptimizer<VectorSpace<double> >
{
public:
	CG();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CG"; }

	using AbstractSingleObjectiveOptimizer<VectorSpace<double > >::init;
	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint);
	void step(const ObjectiveFunctionType& objectiveFunction);
	void configure( const PropertyTree & node );
	//from ISerializable
	void read( InArchive & archive );
	void write( OutArchive & archive ) const;

	/// Access the type of line search (e.g., switch between zeroth and first order line search).
	const LineSearch& lineSearch() const
	{
		return m_linesearch;
	}
	/// Returns type of line search.
	LineSearch& lineSearch()
	{
		return m_linesearch;
	}
	/// Access the number of line searches (iterations) after which the search direction is reset to the gradient.
	const unsigned& reset() const
	{
		return m_numReset;
	}
	/// Returns the number of line searches (iterations) after which the search direction is reset to the gradient.
	unsigned& reset()
	{
		return m_numReset;
	}

protected:

	LineSearch m_linesearch;
	ObjectiveFunctionType::FirstOrderDerivative m_derivative;
	RealVector m_g;
	RealVector m_h;
	RealVector m_xi;

	size_t   m_dimension;
	unsigned m_numReset;
	unsigned m_count;
};

}

#endif
