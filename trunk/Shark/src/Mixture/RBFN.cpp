//===========================================================================
/*!
 *  \file RBFN.cpp
 *
 *  \brief Radial Basis Function Network
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
 */
//===========================================================================

#include <SharkDefs.h>
#include <Array/ArrayOp.h>
#include <LinAlg/LinAlg.h>
#include <Mixture/MixtureLinearRegression.h>
#include <Rng/GlobalRng.h>
#include <Mixture/RBFN.h>

//===========================================================================

#define MIN_VAL 1e-100
#define MAX_VAL 1e+100


RBFN::RBFN(unsigned nInputs,
		   unsigned nOutputs,
		   unsigned nCenters)
		: MixtureOfGaussians(nCenters, nInputs),
		A(nOutputs, nCenters),
		b(nOutputs)
{
	unsigned i;

	for (i = A.nelem(); i--; A.elem(i) = uni());
	for (i = b.nelem(); i--; b.elem(i) = uni());
}

//===========================================================================

void RBFN::setParams(const Array< double >& w)
{
	SIZE_CHECK(w.nelem() == b.nelem() +
			   A.nelem() +
			   m.nelem() +
			   v.nelem())

	unsigned i, k;

	for (k = i = 0; i < b.nelem(); ++i, ++k) {
		b.elem(i) = w(k);
	}
	for (i = 0; i < A.nelem(); ++i, ++k) {
		A.elem(i) = w(k);
	}
	for (i = 0; i < m.nelem(); ++i, ++k) {
		m.elem(i) = w(k);
	}
	for (i = 0; i < v.nelem(); ++i, ++k) {
		v.elem(i) = Shark::max(Shark::sqr(w(k)), MIN_VAL);
	}
}

void RBFN::getParams(Array< double >& w) const
{
	unsigned i, k;

	w.resize(b.nelem() +
			 A.nelem() +
			 m.nelem() +
			 v.nelem(), false);

	for (k = i = 0; i < b.nelem(); ++i, ++k) {
		w(k) = b.elem(i);
	}
	for (i = 0; i < A.nelem(); ++i, ++k) {
		w(k) = A.elem(i);
	}
	for (i = 0; i < m.nelem(); ++i, ++k) {
		w(k) = m.elem(i);
	}
	for (i = 0; i < v.nelem(); ++i, ++k) {
		w(k) = sqrt(Shark::max(v.elem(i), MIN_VAL));
	}
}

//===========================================================================

void RBFN::firstLayer(const Array< double >& x, Array< double >& ex) const
{
	SIZE_CHECK(x.ndim() == 1 && x.dim(0) == dim())

	unsigned i, j;
	double   marg;

	ex.resize(size(), false);

	for (i = 0; i < size(); i++) {
		for (marg = 0, j = dim(); j--;) {
			marg -= Shark::sqr(x(j) - m(i, j)) / Shark::max(v(i, j), MIN_VAL);
		}
		ex(i) = exp(marg / 2);
	}
}

//===========================================================================

double RBFN::gradientMSE(const Array< double >& in,
						 const Array< double >& out,
						 Array< double >& de)
{

	unsigned i, j, k, l;
	double s, t, db, mse;
	Array< double > e;
	Array< double > d(size());

	mse = 0.;

	de.resize(b.nelem() +
			  A.nelem() +
			  m.nelem() +
			  v.nelem(), false);

	if (in.ndim() == 2) {
		de = 0.;

		for (l = 0; l < in.dim(0); ++l) {
			firstLayer(in[ l ], e);

			d = 0.;

			for (j = 0; j < odim(); ++j) {
				db = b(j) - out(l, j);
				for (k = 0; k < size(); ++k) {
					db += A(j, k) * e(k);
				}
				mse += db * db;
				de(j) += (db *= 2);

				for (i = j * size() + b.nelem(), k = 0; k < size();
						++k, ++i) {
					de(i) += db * e(k);
					d(k) += db * e(k) * A(j, k);
				}
			}

			for (j = b.nelem() + A.nelem(), k = 0; k < size(); ++k) {
				for (i = 0; i < dim(); ++i, ++j) {
					s = sqrt(Shark::max(v(k, i), MIN_VAL));
					t = in(l, i) - m(k, i);

					de(j) += d(k) * t  / (s * s);
					de(j + m.nelem()) += d(k) * t * t / (s * s * s);
				}
			}
		}


		de /= (double) in.dim(0);   // added by M. Huesken, 19.10.2001

		return mse / in.dim(0);
	}
	else {
		firstLayer(in, e);

		d = 0.;

		for (j = 0; j < odim(); ++j) {
			db = b(j) - out(j);
			for (k = 0; k < size(); ++k) {
				db += A(j, k) * e(k);
			}
			de(j) = (db *= 2);
			mse += db * db;

			for (i = j * size() + b.nelem(), k = 0; k < size(); ++k, ++i) {
				d(k) += (de(i) = db * e(k)) * A(j, k);
			}
		}

		for (j = b.nelem() + A.nelem(), k = 0; k < size(); ++k) {
			for (i = 0; i < dim(); ++i, ++j) {
				s = sqrt(Shark::max(v(k, i), MIN_VAL));
				t = in(i) - m(k, i);

				de(j + m.nelem()) = (de(j) = d(k) * t / s * s) * t / s;
			}
		}

		return mse;
	}
}

//===========================================================================

void RBFN::gradientOut(const Array< double >& in, Array< double >& dw)
{
	unsigned dwi, i, j, k, o;
	double s, t, Ae;
	Array< double > e;

	dw.resize(odim(),
			  b.nelem() +
			  A.nelem() +
			  m.nelem() +
			  v.nelem(), false);

	firstLayer(in, e);

	for (o = 0; o < odim(); ++o) {
		for (dwi = j = 0; j < odim(); ++j, ++dwi) {
			dw(o, dwi) = o == j ? 1. : 0.;
		}

		for (j = 0; j < odim(); ++j) {
			for (k = 0; k < size(); ++k, ++dwi) {
				dw(o, dwi) = o == j ? e(k) : 0.;
			}
		}

		for (k = 0; k < size(); ++k) {
			Ae = A(o, k) * e(k);

			for (i = 0; i < dim(); ++i, ++dwi) {
				s  = sqrt(v(k, i));
				t  = in(i) - m(k, i);

				dw(o, dwi) = Ae * t / Shark::max(s * s, MIN_VAL);
				dw(o, dwi + m.nelem()) = Ae * t * t / Shark::max(s * s * s, MIN_VAL);
			}
		}
	}
}

//===========================================================================

void RBFN::estimateFisherInformation(const Array< double >& input,
									 const Array< double >& output,
									 Array< double >& A)
{
	SIZE_CHECK(input.ndim() == 2)

	unsigned i;
	Array< double > dw;

	for (i = 0; i < input.dim(0); ++i) {
		gradientOut(input[ i ], dw);
		dw.resize(dw.nelem(), true);
		if (i == 0) {
			A = outerProduct(dw, dw);
		}
		else {
			A += outerProduct(dw, dw);
		}
	}

	A /= mse(input, output);
}

//===========================================================================

void RBFN::estimateInvFisher
(
	const Array< double >& input,
	const Array< double >& output,
	Array< double >& invA,
	Array< double >& transInvA,
	double &S2
)
{
	SIZE_CHECK(input.ndim() == 2)

	unsigned i;
	Array< double > A;
	Array< double > dw;
	Array< double > dwMean;

	for (i = 0; i < input.dim(0); ++i) {
		gradientOut(input[ i ], dw);
		dw.resize(dw.nelem(), true);
		if (i == 0) {
			A = outerProduct(dw, dw);
			dwMean = dw;
		}
		else {
			A += outerProduct(dw, dw);
			dwMean += dw;
		}
	}

	S2 = mse(input, output);
	A /= S2;

	invA = invert(A);

	//
	// inner_product yields the same results as matrix_product (not
	// implemented yet) since all matrices are symmetric
	//
	transInvA = innerProduct(invA,
							 innerProduct(outerProduct(dwMean, dwMean), invA));
}

//===========================================================================

double RBFN::overallVariance(const Array< double >& input,
							 const Array< double >& output)
{
	SIZE_CHECK(input.ndim() == 2)

	unsigned i;
	Array< double > A;
	Array< double > invA;
	Array< double > invAdwMean;
	Array< double > dw;
	Array< double > dwMean;

	for (i = 0; i < input.dim(0); ++i) {
		gradientOut(input[ i ], dw);
		dw.resize(dw.nelem(), true);
		if (i == 0) {
			A = outerProduct(dw, dw);
			dwMean = dw;
		}
		else {
			A += outerProduct(dw, dw);
			dwMean += dw;
		}
	}

	A /= mse(input, output);
	invA = invert(A);
	invAdwMean = innerProduct(invA, outerProduct(dwMean, dwMean));

	return scalarProduct(dwMean, innerProduct(invA, dwMean))
		   + trace(invAdwMean);
}

//===========================================================================

double RBFN::estimateVariance(const Array< double >& input,
							  const Array< double >& invA)
{
	Array< double > dw;
	gradientOut(input, dw);
	dw.resize(dw.nelem(), true);
	return scalarProduct(dw, innerProduct(invA, dw));
}

//===========================================================================

//
// cf. David A. Cohn: Neural Network Exploration Using Optimal Experimental
//     Design. A.I. Memo No. 1491 (via ftp://publications.ai.mit.edu)
//
double RBFN::estimateVarianceChange(const Array< double >& input,
									const Array< double >& invA,
									const Array< double >& transInvA,
									double S2)
{
	Array< double > dw;
	gradientOut(input, dw);
	dw.resize(dw.nelem(), true);
	return scalarProduct(dw, innerProduct(transInvA, dw))
		   / (S2 + scalarProduct(dw, innerProduct(invA, dw)));
}

//===========================================================================

void RBFN::initialize(const Array< double >& x, const Array< double >& y)
{
	//
	// initialize centers and widths of Gaussians
	//
	MixtureOfGaussians::initialize(x);
	MixtureOfGaussians::kmc(x);

	//
	// train linear weights with least squares method
	//
	A.resize(A.dim(0), MixtureOfGaussians::dim(), false);
	train_linear(x, y);
}

//===========================================================================

void RBFN::initialize_linear(double min, double max)
{
	unsigned i;

	for (i = b.nelem(); i--;) {
		b.elem(i) = uni(min, max);
	}
	for (i = A.nelem(); i--;) {
		A.elem(i) = uni(min, max);
	}
}

//===========================================================================

void RBFN::train_linear(const Array< double >& x,
						const Array< double >& y)
{
	SIZE_CHECK(x.ndim() <= 2)

	Array< double > e;
	MixtureLinearRegression lr;

	if (x.ndim() == 2) {
		for (unsigned i = 0; i < x.dim(0); ++i) {
			firstLayer(x[ i ], e);
			lr.train(e, y[ i ]);
		}
	}
	else {
		firstLayer(x, e);
		lr.train(e, y);
	}

	A = lr.A();
	b = lr.b();
}

//===========================================================================

void RBFN::insertRBFData(const Array< double >& input,
						 const Array< double >& output,
						 const Array< double >& minInput,
						 const Array< double >& maxInput)
{
	unsigned i, j;
	Array< double > newA(output.dim(1));
	Array< double > newm(input .dim(1));
	Array< double > newv(input .dim(1));

	//
	// if either minimum or maximum is not specified then compute it with
	// help of the given data
	//
	if (minInput.nelem() == 0 || maxInput.nelem() == 0) {
		Array< double > minIn, maxIn;
		minIn = maxIn = input[ 0 ];
		for (i = 0; i < input.dim(0); ++i) {
			for (j = 0; j < input.dim(0); ++j) {
				if (input(i, j) < minIn(j)) minIn(j) = input(i, j);
				if (input(i, j) > maxIn(j)) maxIn(j) = input(i, j);
			}
		}
		//
		// sample a random mean vector
		//
		for (j = 0; j < newm.nelem(); ++j) {
			newm(j) = Rng::uni(minIn(j), maxIn(j));
		}
	}
	else {
		//
		// sample a random mean vector
		//
		for (j = 0; j < newm.nelem(); ++j) {
			newm(j) = Rng::uni(minInput(j), maxInput(j));
		}
	}

	if (size()) {
		//
		// find the distance to the nearest neighbor
		//
		double d, dist = ::sqrDistance(m[ 0 ], newm);
		for (i = 1; i < size(); ++i) {
			d = ::sqrDistance(m[ i ], newm);
			if (d < dist) {
				dist = d;
			}
		}

		//
		// set the variance randomly scattered around the minimum distance
		//
		for (j = 0; j < dim(); ++j) {
			newv(j) = uni(dist * 0.9, dist * 1.1);
		}

		//
		// set the linear weight
		//
		RBFN tmp(dim(), 1, 1);
		double sum;
		Array< double > oldout, newout;
		newA = 1;
		tmp.insertRBF(newA, newm, newv);
		sum  = 0;
		newA = 0;
		for (j = 0; j < input.dim(0); ++j) {
			recall(input[ j ], oldout);
			tmp.recall(input[ j ], newout);
			newA += (output[ j ] - oldout) * newout(0);
			sum  += Shark::sqr(newout(0));
		}
		newA /= sum;
	}
	else {
		//
		// do something (sensible?)
		//
		newA = 1;
		newv = 1;
	}

	insertRBF(newA, newm, newv);
}

//===========================================================================

void RBFN::insertRBF(const Array< double >& newA,
					 const Array< double >& newm,
					 const Array< double >& newv)
{
	unsigned i, j;
	Array< double > oldA(A);

	A.resize(odim(), size() + 1, false);

	for (i = 0; i < odim(); ++i) {
		for (j = 0; j < size(); ++j) {
			A(i, j) = oldA(i, j);
		}
	}

	for (i = 0; i < odim(); ++i) {
		A(i, size()) = newA(i);
	}

	insertKernel(1, newm, newv);
}

void RBFN::deleteRBF(unsigned k)
{
	RANGE_CHECK(k < size())

	unsigned i, j;
	Array< double > oldA(A);

	A.resize(odim(), size() - 1, false);

	for (i = 0; i < odim(); ++i) {
		for (j = 0; j < k; ++j) {
			A(i, j) = oldA(i, j);
		}
		for (j = k + 1; j < size(); ++j) {
			A(i, j - 1) = oldA(i, j);
		}
	}

	deleteKernel(k);
}

//===========================================================================

void RBFN::resize(unsigned ninput, unsigned noutput, unsigned num, bool copy)
{
	A.resize(noutput, num, copy);
	b.resize(noutput, copy);
	MixtureOfGaussians::resize(num, ninput, copy);
}

void RBFN::resize(unsigned num, bool copy)
{
	resize(dim(), odim(), num, copy);
}

//===========================================================================

/*
void RBFN::train_ran( const Array< double >& x,
                      const Array< double >& y,
                      double epsilon, double delta )
{
    SIZE_CHECK( x.ndim( ) <= 2 )

    if( x.ndim( ) == 2 )
        for( unsigned i = 0; i < x.dim( 0 ); ++i )
            train_ran( x[ i ], y[ i ], epsilon, delta );
    else {
        Array< double > var( dim( ) );
        if( size( ) == 0 ) {
            var = Shark::max( delta, MIN_VAL );
            insertRBF( y, x, var );
            b = 0;
        } else {
            double d;
            err = 0;
            gradient( x, y );
            if( err > epsilon &&
                ( d = euclidian_distance( x, m[ nearest( x ) ] ) ) > delta ) {
                dm = 0;
                ds = 0;
                dA = 0;
                dB = 0;

                Array< double > yy;
                recall( x, yy );
                var = Shark::max( d, MIN_VAL );
                insertRBF( y - yy, x, var );
            } else
                update( );
        }
    }
}
*/

//===========================================================================

void RBFN::recall(const Array< double >& x, Array< double >& y) const
{
	SIZE_CHECK(x.ndim() <= 2)

	Array< double > e;

	if (x.ndim() == 2) {
		y.resize(x.dim(0), b.dim(0), false);
		for (unsigned i = 0; i < x.dim(0); ++i) {
			firstLayer(x[ i ], e);
			y[ i ] = innerProduct(A, e) + b;
		}
	}
	else {
		firstLayer(x, e);
		y = innerProduct(A, e) + b;
	}
}

//===========================================================================

double RBFN::curvature()
{
	double curve = 0;

	for (unsigned j = 0; j < odim(); ++j) {
		for (unsigned i = 0; i < size(); ++i) {
			a(i) = A(j, i);
		}
		curve += MixtureOfGaussians::curvature();
	}

	return curve;
}

void RBFN::gradientCurve(Array< double >& de)
{
	unsigned i, j, k;
	Array< double > da, dm, ds;

	de.resize(b.nelem() +
			  A.nelem() +
			  m.nelem() +
			  v.nelem(), false);
	de = 0.;

	for (j = 0; j < odim(); ++j) {
		for (i = 0; i < size(); ++i) {
			a(i) = A(j, i);
		}
		MixtureOfGaussians::dCurveams(da, dm, ds);

		for (k = b.nelem() + j * size(), i = 0; i < da.nelem(); ++i, ++k) {
			de(k) += da(i);
		}

		for (k = b.nelem() + A.nelem(), i = 0; i < dm.nelem(); ++i, ++k) {
			de(k) += dm(i);
		}

		for (i = 0; i < ds.nelem(); ++i, ++k) {
			de(k) += ds(i);
		}
	}
}

//===========================================================================

