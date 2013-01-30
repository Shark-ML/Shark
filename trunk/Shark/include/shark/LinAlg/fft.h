//===========================================================================
/*!
 *  \file fft.h
 *
 *  \brief fast fourier transformation
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
#ifndef FFT_H
#define FFT_H

#include <cmath>
#include "Base.h"

namespace shark {

/**
* \ingroup shark_globals
* 
* @{
*/

//! Depending on the value of "isign" the "data" is
//! replaced by its discrete Fourier transform
//! or by its inverse discrete Fourier transform.
template<class MatrixT>
void fft(MatrixT & data, int isign);


//! Replaces the "data" by its discrete Fourier transform.
template<class MatrixT>
void fft(MatrixT & data);


//! Replaces the "data" by its inverse discrete Fourier transform.
template<class MatrixT>
void ifft(MatrixT & data);
	
/** @}*/
}
#endif
#include "Impl/fft.inl"
