//===========================================================================
/*!
 *  \file fft.h
 *
 *  \brief Functions for performing a fast fourier transformation
 *
 *  \par Copyright (c) 1998-2003:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      LinAlg
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of LinAlg. This library is free software;
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


#include <cmath>
#include <complex>
#include <SharkDefs.h>
#include <Array/Array.h>



//! Depending on the value of "isign" the "data" is
//! replaced by its discrete Fourier transform
//! or by its inverse discrete Fourier transform.
void fft(Array< std::complex< double > > & data, int isign);


//! Replaces the "data" by its discrete Fourier transform.
void fft(Array< std::complex< double > > & data);


//! Replaces the "data" by its inverse discrete Fourier transform.
void ifft(Array< std::complex< double > > & data);

