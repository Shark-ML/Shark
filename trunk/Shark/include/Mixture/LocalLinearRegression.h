//===========================================================================
/*!
 *  \file LocalLinearRegression.h
 *
 *  \brief Localized linear regression model
 *
 *  \author  Martin Kreutz
 *  \date    1995-01-01
 *
 *  \par Copyright (c) 1995,2002:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      Mixture
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Mixture. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
//===========================================================================

#ifndef __LOCALLINEARREGRESSION_H
#define __LOCALLINEARREGRESSION_H


#include <Mixture/CodeBook.h>
#include <Mixture/MixtureLinearRegression.h>


//!
//! \brief Abstract model of a parameterized mixture density
//!
class LocalLinearRegression : public CodeBook
{
public:
	LocalLinearRegression()
	{ };
	LocalLinearRegression(unsigned n);

	void reset();
	void clear();
	void update();
	void train(const Array< double >&,
			   const Array< double >&);
	void recall(const Array< double >&, Array< double >&);

	//
	// the resize operators must be overloaded in order to
	// properly re-size the array of local models
	//
	void resize(unsigned n, bool = false);
	void resize(unsigned n, unsigned d, bool = false);

	//private:
public:
	Array< MixtureLinearRegression > lr;
};


#endif /* !__LOCALLINEARREGRESSION_H */

