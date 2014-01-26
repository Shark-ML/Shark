//===========================================================================
/*!
 * 
 *
 * \brief       IParameterizable interface
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2010-2011
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
//===========================================================================

#ifndef SHARK_CORE_IPARAMETERIZABLE_H
#define SHARK_CORE_IPARAMETERIZABLE_H


#include <shark/LinAlg/Base.h>


namespace shark {

/// \brief Top level interface for everything that holds parameters.
///
/// This interface is inherited by AbstractModel for unified
/// access to the parameters of models, but also by objective
/// functions and algorithms with hyper-parameters.
class IParameterizable {
public:
	virtual ~IParameterizable () { }

	/// Return the parameter vector.
	virtual RealVector parameterVector() const
	{
		return RealVector();
	}

	/// Set the parameter vector.
	virtual void setParameterVector(RealVector const& newParameters)
	{
		SHARK_ASSERT(newParameters.size() == 0);
	}

	/// Return the number of parameters.
	virtual std::size_t numberOfParameters() const {
		return parameterVector().size();
	}
};


}
#endif
