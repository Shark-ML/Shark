//===========================================================================
/*!
 *  \file fft.cpp
 *
 *  \brief Functions for performing a fast fourier transformation
 *
 *  The Fourier transformation is often used in the technical sector, e.g.
 *  in the communications engineering. <br>
 *  In general a physical process can be described either in the time
 *  domain, by the value of some quantity \f$h\f$ as a function of time
 *  \f$t\f$, e.g. \f$h(t)\f$, or else in the frequency domain,
 *  where the process is specified by giving its amplitude \f$H\f$
 *  (generally a complex number indicating phase also) as a function
 *  of frequency \f$f\f$, that is \f$H(f)\f$, with
 *  \f$- \infty < f < \infty\f$. It is often useful to think of
 *  \f$h(t)\f$ and \f$H(f)\f$ as being two different representations
 *  of the same function. You can go back and forth between these
 *  two representations by means of the Fourier transform equations
 *  \f$ H(f) = \int_{- \infty}^{\infty} h(t)e^{2\pi i f t} dt\f$ and
 *  \f$h(t) = \int_{- \infty}^{\infty} H(f)e^{-2\pi i f t} df\f$. <br>
 *  If the function \f$f\f$ is given by a finite number \f$N\f$ of sample
 *  points \f$h_k\f$, then the discrete Fourier transform of this function
 *  can be calculated by
 *  \f$H_n \equiv \sum_{k=0}^{N-1} h_k e^{2 \pi i k n / N}\f$. <br>
 *  The discrete inverse Fourier transform is then given by
 *  \f$h_k = \frac{1}{N} \sum_{n=0}^{N-1} H_n e^{-2 \pi i k n / N}\f$.
 *  <br>
 *  The discrete Fourier transformation has the calculation time
 *  of \f$O(N^2)\f$, that can be critical for great \f$N\f$. <br>
 *  The "fast" Fourier transformation uses the fact, that a discrete
 *  Fourier transform of length \f$N\f$ can be rewritten as the
 *  sum of two discrete Fourier transforms, each of length \f$\frac{N}{2}\f$,
 *  where one of the two is formed from the even-numbered points
 *  and the other from the odd-numbered points. Then using this
 *  splitting recursively we finally have transforms of length 1,
 *  which are just the identity operation.
 *  By this way the calculation time
 *  can be decreased to \f$O(N log_2 N)\f$. To use this algorithm,
 *  the number \f$N\f$ of sample points must be an integer power of two.
 *  The following example will show how the functions in this file
 *  are used: <br>
 *
 *
 *  You can find, compile and run this example in
 *  "$SHARK_ROOTDIR/LinAlg/examples".
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Copyright (c) 1998-2000:
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


#include "LinAlg/fft.h"


//===========================================================================
/*!
 *  \brief Depending on the value of "isign" the "data" is
 *         replaced by its discrete Fourier transform
 *         or by its inverse discrete Fourier transform.
 *
 *  When \em isign is set to "-1", then
 *  \f$H_n \equiv \sum_{k=0}^{N-1} data_k e^{2 \pi i k n / N}\f$
 *  is calculated. <br>
 *  When \em isign is set to "1", then
 *  \f$h_k = \frac{1}{N} \sum_{n=0}^{N-1} data_k e^{-2 \pi i k n / N}\f$
 *  is calculated. <br>
 *  Please notice, that the number of sample points in \em data
 *  must be an integer power of 2. If there are not enough
 *  sample points, create the data array with a size equal to the
 *  next higher power of 2 and fill the last empty positions of
 *  the array with zero values.
 *
 *  \param data  The original sample data (in the time or frequency
 *               domain), that will be replaced by its corresponding
 *               data of the other domain.
 *  \param isign Determines the type of transformation. <br>
 *               "-1" = a discrete Fourier transform is performed, <br>
 *               "1" = an inverse Fourier transform is performed
 *  \return none.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void fft(Array< std::complex< double > > & data, int isign)
{
	int    i1, i2, i3, i2rev, ip1, ip2, ip3, ifp1, ifp2;
	int    ibit, idim, k1, k2, n, nprev, nrem;
	double theta, wtemp, twopi;
	std::complex< double > temp, w, wp;

	twopi = 2.0 * M_PI;

	nprev = 1;

	for (idim = data.ndim(); idim--;) {
		n = data.dim(idim);
		nrem = data.nelem() / (n * nprev);
		ip1 = nprev << 1;
		ip2 = ip1 * n;
		ip3 = ip2 * nrem;
		i2rev = 1;
		for (i2 = 1; i2 <= ip2; i2 += ip1) {
			if (i2 < i2rev) {
				for (i1 = i2; i1 <= i2 + ip1 - 2; i1 += 2) {
					for (i3 = i1 - 1; i3 < ip3; i3 += ip2) {
						std::swap
						(
							data.elem(i3 / 2),
							data.elem((i2rev + i3 - i2) / 2)
						);
					}
				}
			}

			ibit = ip2 >> 1;
			while ((ibit >= ip1) && (i2rev > ibit)) {
				i2rev -= ibit;
				ibit >>= 1;
			}
			i2rev += ibit;
		}
		ifp1 = ip1;
		while (ifp1 < ip2) {
			ifp2  = ifp1 << 1;
			theta = isign * twopi / (ifp2 / ip1);
			wtemp = sin(0.5 * theta);
			wp = std::complex< double >(-2.0 * wtemp * wtemp, sin(theta));
			w  = 1;
			for (i3 = 1; i3 <= ifp1; i3 += ip1) {
				for (i1 = i3; i1 <= i3 + ip1 - 2; i1 += 2) {
					for (i2 = i1 - 1; i2 < ip3; i2 += ifp2) {
						k1 = i2 / 2;
						k2 = k1 + ifp1 / 2;
						temp = data.elem(k2) * w;
						data.elem(k2)  = data.elem(k1) - temp;
						data.elem(k1) += temp;
					}
				}

				w += w * wp;
			}
			ifp1 = ifp2;
		}
		nprev *= n;
	}
}

//===========================================================================
/*!
 *  \brief Replaces the "data" by its inverse discrete Fourier transform.
 *
 *  \f$h_k = \frac{1}{N} \sum_{n=0}^{N-1} data_k e^{-2 \pi i k n / N}\f$
 *  is calculated. <br>
 *  Please notice, that the number of sample points in \em data
 *  must be an integer power of 2. If there are not enough
 *  sample points, create the data array with a size equal to the
 *  next higher power of 2 and fill the last empty positions of
 *  the array with zero values.
 *
 *  \param data The original sample data of the frequency domain, that will be
 *              replaced by its corresponding data of the time domain.
 *  \return none.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void ifft(Array< std::complex< double > > & data)
{
	fft(data, 1);
	for (unsigned i = 0; i < data.nelem(); ++i)
		data.elem(i) /= double(data.nelem());
}

//===========================================================================
/*!
 *  \brief Replaces the "data" by its discrete Fourier transform.
 *
 *  \f$H_n \equiv \sum_{k=0}^{N-1} data_k e^{2 \pi i k n / N}\f$
 *  is calculated. <br>
 *  Please notice, that the number of sample points in \em data
 *  must be an integer power of 2. If there are not enough
 *  sample points, create the data array with a size equal to the
 *  next higher power of 2 and fill the last empty positions of
 *  the array with zero values.
 *
 *  \param data The original sample data of the time domain, that will be
 *              replaced by its corresponding data of the frequency domain.
 *  \return none.
 *
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void fft(Array< std::complex< double > > & data)
{
	fft(data, -1);
}
