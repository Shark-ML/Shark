//===========================================================================
/*!
 *  \file IParameterizable.h
 *
 *  \brief IParameterizable interface
 *
 *  \author T. Glasmachers
 *  \date 2010-2011
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
	virtual RealVector parameterVector() const = 0;
	/// Set the parameter vector.
	virtual void setParameterVector(RealVector const& newParameters) = 0;
	/// Return the number of parameters.
	virtual std::size_t numberOfParameters() const {
		return parameterVector().size();
	}
};


}
#endif
