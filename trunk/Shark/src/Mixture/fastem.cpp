//===========================================================================
/*!
 *  \file fastem.cpp
 *
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
 *      Mixture
 *
 *
 *
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
 *
 *
 */
//===========================================================================




static double fastem(unsigned size, unsigned dim, unsigned num,
					 double* a, double* m, double* v, const double* x)
{
	unsigned d, i, j, k, l;
	double   ll = 0;

	if (size == 1) {
		for (i = dim; i--; m[ i ] = v[ i ] = 0.);

		for (k = num, l = num * dim - 1; k--;)
			for (i = dim; i--;) {
				v[ i ] += x[ l ] * x[ l ];
				m[ i ] += x[ l-- ];
			}

		a[ 0 ] = 1.;

		for (i = dim; i--;) {
			m[ i ] /= num;
			v[ i ]  = v[ i ] / num - m[ i ] * m[ i ];
		}
	}
	else {
		double  px, t, marg, varg;
		double* mem = new double[ size * 2 *(dim + 1)];
		double* pi  = mem;
		double* na  = mem + size;
		double* nm  = mem + size * 2;
		double* nv  = mem + size * (2 + dim);

		//
		// set all variables to zero
		//
		for (i = size * (2 * dim + 1); i--; na[ i ] = 0.);

		//
		// loop over all data vectors
		//
		for (k = num; k--;) {
			for (px = 0., i = size, j = size * dim - 1; i--;) {
				marg = 0.;
				varg = 1.;

				for (d = dim, l = (k + 1) * dim - 1; d--;) {
					t = x[ l-- ] - m[ j ];
					marg -= t * t / v[ j ];
					varg *= v[ j-- ];
				}

				if (! finite(marg) || ! finite(varg))
					pi[ i ] = 0.;
				else {
					if (varg < MIN_VAL)
						varg = MIN_VAL;
					px += (pi[ i ] = exp(marg / 2)
									 / sqrt(varg) * a[ i ]);
				}
			}

			if (px < MIN_VAL)
				px = MIN_VAL;

			ll += log(px);

			for (i = size, j = size * dim - 1; i--;) {
				na[ i ] += (pi[ i ] /= px);
				for (d = dim, l = (k + 1) * dim - 1; d--;) {
					nv[ j   ] += pi[ i ] * x[ l ] * x[ l ];
					nm[ j-- ] += pi[ i ] * x[ l-- ];
				}
			}
		}

		for (i = size, j = size * dim - 1; i--;)
			if ((a[ i ] = na[ i ] / num) > MIN_VAL)
				for (k = dim; k--; j--) {
					m[ j ] = nm[ j ] / na[ i ];
					v[ j ] = nv[ j ] / na[ i ] - m[ j ] * m[ j ];
				}
			else
				j -= dim;

		delete[ ] mem;
	}

	return ll;
}

//===========================================================================

