//===========================================================================
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
//===========================================================================


#ifndef SHARK_ML_OPTIMIZER_BFGS_H
#define SHARK_ML_OPTIMIZER_BFGS_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/Algorithms/GradientDescent/LineSearch.h>

namespace shark {

//! \brief Broyden, Fletcher, Goldfarb, Shannon algorithm for unconstraint optimization
class BFGS : public AbstractSingleObjectiveOptimizer<VectorSpace<double> >
{
public:
	BFGS();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "BFGS"; }

	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint);
	using AbstractSingleObjectiveOptimizer<VectorSpace<double> >::init;

	void step(const ObjectiveFunctionType& objectiveFunction);

	//from IConfigure
	void configure( const PropertyTree & node );

	//from ISerializable
	void read( InArchive & archive );
	void write( OutArchive & archive ) const;


	//linesearch handling
	const LineSearch& lineSearch()const
	{
		return m_linesearch;
	}
	LineSearch& lineSearch()
	{
		return m_linesearch;
	}
protected:
	ObjectiveFunctionType::FirstOrderDerivative m_derivative;
	LineSearch m_linesearch;

	size_t     m_parameters;
	/// \brief the derivative of the function of the last evaluated point
	RealVector m_lastDerivative;
	// the approximated Hessian matrix
	RealMatrix m_hessian;

	double m_initialStepLength;

};

}
#endif
