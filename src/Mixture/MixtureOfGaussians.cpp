//===========================================================================
/*!
 *  \file MixtureOfGaussians.cpp
 *
 *  \brief Core class implementing a sum of Gaussians
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

#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <SharkDefs.h>
#include <Array/ArrayOp.h>
#include <Rng/GlobalRng.h>
#include <Mixture/MixtureOfGaussians.h>

#ifdef __STDC__
extern "C" double erf(double) throw();
#endif

#ifdef _WIN32 // !!!
#define erf( x ) tanh( x )
#endif

#define MIN_VAL 1e-100
#define MAX_VAL 1e+100
#define LOG_MIN_VAL -230.259
#define LOG_MAX_VAL +230.259
#define SQRT_MAX_VAL 1e+50


using namespace std;

static inline double clip(double x, double minx, double maxx)
{
	return Shark::min(Shark::max(x, minx), maxx);
}

//===========================================================================

bool MixtureOfGaussians::isfinite() const
{
	/* !!!
	  unsigned i;

	  for( i = a.nelem( ); i--; )
	      if( ! finite( a.elem( i ) ) )
	   return false;

	  for( i = m.nelem( ); i--; )
	      if( ! finite( m.elem( i ) ) || ! finite( v.elem( i ) ) )
	   return false;
	*/
	return true;
}
//===========================================================================

void MixtureOfGaussians::join(const Array< double >& x,
							  const Array< double >& y,
							  Array< double >& z)
{
	unsigned i, j, k;

	if (x.ndim() == 2) {
		z.resize(x.dim(0), x.dim(1) + y.dim(1), false);

		for (i = x.dim(0); i--;) {
			for (j = k = 0; j < x.dim(1); ++j, ++k)
				z(i, k) = x(i, j);
			for (j = 0; j < y.dim(1); ++j, ++k)
				z(i, k) = y(i, j);
		}
	}
	else {
		z.resize(x.nelem() + y.nelem(), false);

		for (k = i = 0; i < x.nelem(); ++i, ++k)
			z(k) = x.elem(i);
		for (i = 0; i < y.nelem(); ++i, ++k)
			z(k) = y.elem(i);
	}
}

//===========================================================================

bool MixtureOfGaussians::operator == (const MixtureOfGaussians& mix) const
{
	return ((a == mix.a) && (m == mix.m) && (v == mix.v));
}

//===========================================================================

//
// force x to be (significantly) greater than zero
//
#define force_gt0( x ) \
{ \
    if( x < MIN_VAL ) { \
        x = MIN_VAL; \
	cerr << "cut-off in " << __FILE__ << ":" << __LINE__ << endl; \
    } \
}

double MixtureOfGaussians::q(const Array< double >& x,
							 const MixtureOfGaussians& old) const
{
	/*
	  SIZE_CHECK( x.ndim( ) == 2 && x.dim( 1 ) == dim ( ) &&
	size( ) == old.size( ) &&
	dim ( ) == old.dim ( ) )
	*/

	unsigned i, k;
	double   q, px;
	Array< double > pi(size());

	//
	// loop over all data vectors
	//
	for (q = 0., k = x.dim(0); k--;) {
		//
		// can be cached ( size( ) * x.dim( 0 ) ) !!!
		//
		for (px = 0., i = size(); i--;)
			px += (pi(i) = old.p(x[ k ], i) * old.a(i));
		pi /= Shark::max(px, MIN_VAL) ;
		for (i = size(); i--;)
			q += log(Shark::max(a(i) * p(x[ k ], i), MIN_VAL)) * pi(i);
	}

	return q;
}

//===========================================================================

void MixtureOfGaussians::dq(const Array< double >& x,
							const MixtureOfGaussians& old,
							Array< double >& da,
							Array< double >& dm,
							Array< double >& ds) const
{
	/*
	  SIZE_CHECK( x.ndim( ) == 2 && x.dim( 1 ) == dim ( ) &&
	size( ) == old.size( ) &&
	dim ( ) == old.dim ( ) )
	*/

	unsigned i, j, k;
	double   px;
	Array< double > pi(size());

	da.resize(size(), false);
	dm.resize(size(), dim(), false);
	ds.resize(size(), dim(), false);
	da = 0.;
	dm = 0.;
	ds = 0.;

	//
	// loop over all data vectors
	//
	for (k = x.dim(0); k--;) {
		for (px = 0., i = size(); i--;)
			px += (pi(i) = old.p(x[ k ], i) * old.a(i));
		pi /= Shark::max(px, MIN_VAL);
		for (i = 0; i < size(); ++i) {
			da(i)    += pi(i) - a(i);
			for (j = 0; j < dim(); ++j) {
				dm(i, j) += pi(i) * (x(k, j) - m(i, j))
							/ v(i, j);
				ds(i, j) += pi(i) * (Shark::sqr(x(k, j) - m(i, j))
									 / v(i, j) - 1)
							/ sqrt(v(i, j));
			}
		}
	}
}

//===========================================================================

void MixtureOfGaussians::firstLayer(const Array< double >& x,
									Array< double >& ex) const
{
	double norm = 0;

	ex.resize(size(), false);

	for (unsigned i = size(); i--;)
		norm += (ex(i) = prior(i) * p(x, i));

	ex /= norm;
}

//===========================================================================

/*
void MixtureOfGaussians::gradientMSE( )
{
    //POINTER_CHECK( _x != NULL && _y != NULL )

    unsigned i, k, l;
    Array< double > e;
    ArrayReference< double > x, y;
    double s, t, f, df, dfe, dfeu;

    for( l = 0; ; ++l ) {
        if( _x->ndim( ) == 2 ) {
	    x.copyReference( ( *_x )[ l ] );
	    y.copyReference( ( *_y )[ l ] );
	} else {
	    x.copyReference( *_x );
	    y.copyReference( *_y );
	}

	//SIZE_CHECK( y.nelem( ) == 1 )
	firstLayer( x, e );

	f = 0;
	for( i = 0; i < size( ); ++i )
	    f += m( i, dim( )-1 ) * e( i );
	df = f - y( 0 );

	for( k = 0; k < size( ); ++k ) {
	    dfe  = df * e( k );
	    dfeu = dfe * ( m( k, dim( )-1 ) - f );
	    dm( k, dim( )-1 ) += dfe;
	    db( k           ) += dfeu;
	    for( i = 0; i < dim( )-1; ++i ) {
	        s  = sqrt( v( k, i ) );
		t  = x( i ) - m( k, i );

		dm( k, i ) += dfeu * t / ( s*s );
		ds( k, i ) += dfeu * ( ( t*t ) / ( s*s*s ) - 1 / s );
	    }
	}

	if( _x->ndim( ) == 1 || l+1 >= _x->dim( 0 ) )
	    break;
    }
}

//===========================================================================

void MixtureOfGaussians::gradientLL( )
{
  //POINTER_CHECK( _x != NULL )

    unsigned i, j, l;
    ArrayReference< double > x;
    Array< double > pi( size( ) );
    double s, t, px;

    for( l = 0; ; ++l ) {
        if( _x->ndim( ) == 2 )
	    x.copyReference( ( *_x )[ l ] );
	else
	    x.copyReference( *_x );

	for( px = 0., i = size( ); i--; )
	    px += ( pi( i ) = p( x, i ) * a( i ) );
	pi /= Shark::max( px, MIN_VAL );

	for( i = 0; i < size( ); ++i ) {
	    db( i ) -= pi( i ) - prior( i );
	    for( j = 0; j < dim( ); ++j ) {
	        s  = sqrt( v( i, j ) );
		t  = x( j ) - m( i, j );

		dm( i, j ) -= pi( i ) * t / ( s*s );
		ds( i, j ) -= pi( i ) * ( ( t*t ) / ( s*s*s ) - 1 / s );
	    }
	}

	if( _x->ndim( ) == 1 || l+1 >= _x->dim( 0 ) )
	    break;
    }
}
*/

//===========================================================================

MixtureOfGaussians::MixtureOfGaussians(unsigned num,
									   unsigned dim)
		: MixtureModel< double >(num),
		CodeBook(num, dim),
		v(num, dim)
{
	if (num) {
		a = 1. / num;
		v = 1.;
	}
}

//===========================================================================

void MixtureOfGaussians::resize(unsigned n, bool copy)
{
	MixtureModel< double >::resize(n, copy);
	CodeBook::resize(n, dim(), copy);
	v.resize(n, dim(), copy);
}

//===========================================================================

void MixtureOfGaussians::resize(unsigned n, unsigned d, bool copy)
{
	MixtureModel< double >::resize(n, copy);
	CodeBook::resize(n, d, copy);
	v.resize(n, d, copy);
}

//===========================================================================

unsigned MixtureOfGaussians::removeDuplicates(double eps, bool norm)
{
	unsigned i, j, k, n;
	Array< unsigned > numinst(size());
	MixtureOfGaussians mix(*this);

	numinst = 1;
	for (n = i = 0; i < size(); ++i)
		for (j = i; j--;)
			if (2 * fabs((a(i) - a(j)) / (a(i) + a(j))) <= eps) {
				for (k = 0; k < dim(); ++k)
					if ((2 * fabs((m(i, k) - m(j, k)) /
								  (m(i, k) + m(j, k))) > eps) ||
							(2 * fabs((v(i, k) - v(j, k)) /
									  (v(i, k) + v(j, k))) > eps))
						break;

				if (k == dim()) {
					numinst(i) += numinst(j);
					numinst(j)  = 0;
					++n;
					break;
				}
			}

	resize(size() - n, dim(), false);

	for (i = j = 0; i < mix.size(); ++i)
		if (numinst(i) > 0) {
			a(j) = mix.a(i) * numinst(i);
			for (k = 0; k < mix.dim(); ++k) {
				m(j, k) = mix.m(i, k);
				v(j, k) = mix.v(i, k);
			}
			++j;
		}

	if (norm) normalize();

	return n;
}

//===========================================================================

unsigned MixtureOfGaussians::removeMinVar(double minv, bool norm)
{
	unsigned i, j, k, n;
	Array< bool > ok(size());
	MixtureOfGaussians mix(*this);

	ok = true;

	for (n = i = 0; i < size(); ++i)
		for (j = 0; j < dim(); ++j)
			if (mix.v(i, j) <= minv) {
				++n;
				ok(i) = false;
				break;
			}

	resize(size() - n, dim(), false);

	for (n = i = k = 0; i < size(); ++i)
		if (ok(i)) {
			a(k) = mix.a(i);
			for (j = 0; j < dim(); ++j) {
				m(k, j) = mix.m(i, j);
				v(k, j) = mix.v(i, j);
			}
			++k;
		}

	if (norm) normalize();

	return n;
}

//===========================================================================

void MixtureOfGaussians::append(const MixtureOfGaussians& mix, bool norm)
{
	//SIZE_CHECK( size( ) == mix.size( ) && dim( ) == mix.dim( ) )

	unsigned i, j, k;
	Array< double > na(size() + mix.size());
	Array< double > nm(size() + mix.size(), dim());
	Array< double > nv(size() + mix.size(), dim());

	for (i = 0; i < size(); ++i) {
		na(i) = a(i);
		for (j = dim(); j--;) {
			nm(i, j) = m(i, j);
			nv(i, j) = v(i, j);
		}
	}

	for (k = 0; k < mix.size(); ++i, ++k) {
		na(i) = mix.a(k);
		for (j = dim(); j--;) {
			nm(i, j) = mix.m(k, j);
			nv(i, j) = mix.v(k, j);
		}
	}

	a = na;
	m = nm;
	v = nv;

	if (norm) normalize();
}

//===========================================================================

void MixtureOfGaussians::insertKernel(double                 _a,
									  const Array< double >& _m,
									  const Array< double >& _v,
									  bool norm)
{
	unsigned i, j;
	Array< double > na(size() + 1);
	Array< double > nm(size() + 1, dim());
	Array< double > nv(size() + 1, dim());

	for (i = size(); i--;) {
		na(i) = a(i);
		for (j = dim(); j--;) {
			nm(i, j) = m(i, j);
			nv(i, j) = v(i, j);
		}
	}

	na(size()) = _a;
	for (j = dim(); j--;) {
		nm(size(), j) = _m(j);
		nv(size(), j) = _v(j);
	}

	a = na;
	m = nm;
	v = nv;

	if (norm) normalize();
}

//===========================================================================

void MixtureOfGaussians::insertKernel(const Array< double >& min,
									  const Array< double >& max, bool norm)
{
	double aa;
	Array< double > mu(min.nelem());
	Array< double > s2(min.nelem());

	//
	// sample a random mean vector
	//
	for (unsigned j = 0; j < mu.nelem(); ++j)
		mu(j) = uni(min(j), max(j));

	if (size()) {
		//
		// find the distance to the nearest neighbor
		//
		double d, dist = ::sqrDistance(m[ 0 ], mu);
		for (unsigned i = 1; i < size(); ++i) {
			d = ::sqrDistance(m[ i ], mu);
			if (d < dist) dist = d;
		}

		//
		// set the variance randomly scattered around the minimum distance
		//
		for (unsigned j = 0; j < dim(); ++ j)
			s2(j) = uni(dist * 0.9, dist * 1.1);

		//
		// choose a random prior
		//
		aa = uni(1. / (size() * 0.9), 1. / (size() * 1.1));
	}
	else {
		//
		// do something (sensible?)
		//
		s2 = 1;
		aa = 1;
	}

	insertKernel(aa, mu, s2, norm);
}

void MixtureOfGaussians::insertKernel(bool norm)
{
	Array< double > min(dim());
	Array< double > max(dim());

	if (size()) {
		//
		// find minimum and maximum of kernel means
		//
		min = max = m[ 0 ];
		for (unsigned i = 1; i < size(); ++i)
			for (unsigned j = 0; j < dim(); ++j) {
				if (m(i, j) < min(j)) min(j) = m(i, j);
				if (m(i, j) > max(j)) max(j) = m(i, j);
			}

		//
		// expand range by 10 per cent
		//
		for (unsigned j = 0; j < dim(); ++j) {
			min(j) = min(j) - 0.1 * fabs(min(j));
			max(j) = max(j) + 0.1 * fabs(max(j));
		}
	}
	else {
		//
		// know nothing, do something
		//
		min = -1;
		max = 1;
	}

	insertKernel(min, max, norm);
}

//===========================================================================

void MixtureOfGaussians::deleteKernel(unsigned k, bool norm)
{
	//SIZE_CHECK( size( ) >= 1 )

	unsigned i, j;
	Array< double > na(size() - 1);
	Array< double > nm(size() - 1, dim());
	Array< double > nv(size() - 1, dim());

	for (i = 0; i < k; ++i) {
		na(i) = a(i);
		for (j = dim(); j--;) {
			nm(i, j) = m(i, j);
			nv(i, j) = v(i, j);
		}
	}

	for (i = k + 1; i < size(); ++i) {
		na(i - 1) = a(i);
		for (j = dim(); j--;) {
			nm(i - 1, j) = m(i, j);
			nv(i - 1, j) = v(i, j);
		}
	}

	a = na;
	m = nm;
	v = nv;

	if (norm) normalize();
}

//===========================================================================

void MixtureOfGaussians::deleteKernel(unsigned k, unsigned l, bool norm)
{
	if (k == l) {
		deleteKernel(k, norm);
		return;
	}

	//SIZE_CHECK( size( ) >= 2 )

	unsigned i, j;
	Array< double > na(size() - 2);
	Array< double > nm(size() - 2, dim());
	Array< double > nv(size() - 2, dim());

	if (k > l) std::swap(k, l);

	for (i = 0; i < k; ++i) {
		na(i) = a(i);
		for (j = dim(); j--;) {
			nm(i, j) = m(i, j);
			nv(i, j) = v(i, j);
		}
	}

	for (i = k + 1; i < l; ++i) {
		na(i - 1) = a(i);
		for (j = dim(); j--;) {
			nm(i - 1, j) = m(i, j);
			nv(i - 1, j) = v(i, j);
		}
	}

	for (i = l + 1; i < size(); ++i) {
		na(i - 2) = a(i);
		for (j = dim(); j--;) {
			nm(i - 2, j) = m(i, j);
			nv(i - 2, j) = v(i, j);
		}
	}

	a = na;
	m = nm;
	v = nv;

	if (norm) normalize();
}

//===========================================================================

double MixtureOfGaussians::overlap(unsigned i, unsigned j) const
{
	double marg = 0.;

	for (unsigned k = 0; k < dim(); k++)
		marg -= Shark::sqr(m(j, k) - m(i, k)) / (v(i, k) + v(j, k));

	return exp(marg / 2);
}

//===========================================================================

unsigned MixtureOfGaussians::maxOverlap(unsigned i) const
{
	unsigned j, k = i;
	double   o, p = 0;

	for (j = 0; j < size(); j++)
		if (i != j && (o = overlap(i, j)) > p) {
			p = o;
			k = j;
		}

	return k;
}

//===========================================================================

double MixtureOfGaussians::volume(unsigned i) const
{
	double vol = 1;

	for (unsigned j = 0; j < dim(); ++j)
		vol *= v(i, j);

	return vol;
}

//===========================================================================

unsigned MixtureOfGaussians::minVolume() const
{
	unsigned i, j, k;
	double   vol;
	double   minvol;

	for (minvol = volume(k = i = 0); i < size(); ++i) {
		for (vol = 1, j = 0; j < dim(); ++j)
			vol *= v(i, j);
		if (vol < minvol) {
			vol = minvol;
			k = i;
		}
	}

	return k;
}

//===========================================================================

unsigned MixtureOfGaussians::maxVolume() const
{
	unsigned i, j, k;
	double   vol;
	double   maxvol;

	for (maxvol = 0, k = i = 0; i < size(); ++i) {
		for (vol = 1, j = 0; j < dim(); ++j)
			vol *= v(i, j);
		if (vol > maxvol) {
			vol = maxvol;
			k = i;
		}
	}

	return k;
}

//===========================================================================

/*
void MixtureOfGaussians::splitDataset( const Array< double >& x,
				       vector< Array< double > >& z )
{
    unsigned i;
    vector< unsigned > cnt( size( ) );
    vector< unsigned > ind( x.dim( 0 ) );

    z.resize( size( ), false );

    for( i = cnt.size( ); i--; )
        cnt[ i ] = 0;
    for( i = ind.size( ); i--; )
        cnt[ ind[ i ] = sampleModel( x[ i ] ) ]++;

    for( i = cnt.size( ); i--; )
        z[ i ].resize( cnt[ i ], dim( ), false );

    for( i = cnt.size( ); i--; )
        cnt[ i ] = 0;
    for( i = ind.size( ); i--; )
        z[ ind[ i ] ][ cnt[ ind[ i ] ]++ ] = x[ i ];
}
*/

void MixtureOfGaussians::splitDataset(const Array< double >& x,
									  vector< Array< double > > & z)
{
	//SIZE_CHECK( x.ndim( ) == 2 && x.dim( 1 ) == dim( ) )

	unsigned i, j;
	Array< unsigned > cnt(size());
	Array< unsigned > ind(x.dim(0));

	z.resize(size());

	for (i = cnt.nelem(); i--;)
		cnt(i) = 0;
	for (i = ind.nelem(); i--;)
		cnt(ind(i) = sampleModel(x[ i ]))++;

	for (i = cnt.nelem(); i--;)
		z[ i ].resize(cnt(i), dim(), false);

	for (i = cnt.nelem(); i--;)
		cnt(i) = 0;
	for (i = ind.nelem(); i--;) {
		Array< double >& zii = z[ind(i)];
// 		const double* xptr = x.elemvec() + i * dim();
// 		double* zptr = z[ ind(i)].elemvec() + cnt(ind(i))++ * dim();
		ArrayReference<double> xptr = x[i];
		ArrayReference<double> zptr = zii[cnt(ind(i))++];
		for (j = dim(); j--;)
// 			zptr[ j ] = xptr[ j ];
			zptr(j) = xptr(j);
	}
}

//===========================================================================

void MixtureOfGaussians::merge(unsigned i, unsigned j,
							   vector< Array< double > > & z)
{
	const unsigned MinNumber = 3;

	//
	// if both datasets z[ i ] and z[ j ] contain too few samples
	// the likelihood estimation becomes unstable and an alternative
	// method will be used
	//
	if (z[ i ].dim(0) + z[ j ].dim(0) < MinNumber) {
		if (z[ i ].dim(0) < z[ j ].dim(0))
			deleteKernel(i, true);
		else
			deleteKernel(j, true);
		return;
	}

	unsigned k, l;
	double   mergea;
	Array< double > mergeset(z[ i ].dim(0) + z[ j ].dim(0), dim());
	MixtureOfGaussians gmMerge(1, dim());

	//
	// merge data sets
	//
	for (k = l = 0; l < z[ i ].dim(0);)
		mergeset[ k++ ] = z[ i ][ l++ ];
	for (l = 0; l < z[ j ].dim(0);)
		mergeset[ k++ ] = z[ j ][ l++ ];

	//
	// initialize the mixture model
	//
	gmMerge.initialize(mergeset);
	gmMerge.kmc(mergeset, 1e-6, 1000);
	gmMerge.em(mergeset, 1e-6, 1000);

	//
	// replace the two original kernels by the new kernel
	// (note that model remains normalized)
	//
	mergea = a(i) + a(j);
	deleteKernel(i, j, false);
	insertKernel(mergea, gmMerge.m[ 0 ], gmMerge.v[ 0 ], false);
}

//===========================================================================

void MixtureOfGaussians::split(unsigned i,
							   vector< Array< double > > & z)
{
	const unsigned MinNumber = 3;

	//
	// if datasets z[ i ] contains too few samples
	// the likelihood estimation becomes unstable and an alternative
	// method will be used
	//
	if (z[ i ].dim(0) < MinNumber) {
		return;
	}

	double splita;
	MixtureOfGaussians gmSplit(2, dim());

	//
	// initialize the mixture model
	//
	gmSplit.initialize(z[ i ]);
	gmSplit.kmc(z[ i ], 1e-6, 1000);
	gmSplit.em(z[ i ], 1e-6, 1000);

	//
	// replace the original kernel by the new two kernels
	// (note that model remains normalized)
	//
	splita = a(i);
	deleteKernel(i, false);
	insertKernel(gmSplit.a(0) * splita, gmSplit.m[ 0 ], gmSplit.v[ 0 ], false);
	insertKernel(gmSplit.a(1) * splita, gmSplit.m[ 1 ], gmSplit.v[ 1 ], false);
}

//===========================================================================

double MixtureOfGaussians::p(const Array< double >& x, unsigned i) const
{
	//SIZE_CHECK( x.ndim( ) == 1 && x.nelem( ) <= dim( ) )

	double marg = 0.;
	double varg = 1.;

	for (unsigned j = x.dim(0); j--;) {
		//marg -= Shark::sqr( x.elem( j ) - m( i, j ) ) / Shark::max( v( i, j ), MIN_VAL );
		marg -= Shark::sqr(x.elem(j) - m(i, j)) / v(i, j);
		//
		// multiplication with 2 Pi in the inner loop is faster than
		// a call to pow( ... ) (for moderately small x.dim( 0 ) )
		//
		//if( varg < SQRT_MAX_VAL && v( i, j ) < SQRT_MAX_VAL )
		varg *= v(i, j) * M_2PI;
	}

	/* !!!
	if( ! finite( marg ) || ! finite( varg ) )
	    return 0;
	else */ if (varg < MIN_VAL)
		return exp(marg / 2) / MIN_VAL;
	else
		return exp(marg / 2) / sqrt(varg);
}

/*
double MixtureOfGaussians::p( const Array< double >& x, unsigned i ) const
{
  //SIZE_CHECK( x.ndim( ) == 1 && x.nelem( ) <= dim( ) )

    const double* xptr = x.elemvec( );
    const double* mptr = m.elemvec( ) + i * dim( );
    const double* vptr = v.elemvec( ) + i * dim( );
    double marg = 0.;
    double varg = 1.;

    for( unsigned j = x.dim( 0 ); j--; ) {
	marg -= Shark::sqr( xptr[ j ] - mptr[ j ] ) / vptr[ j ];
	varg *= vptr[ j ] * M_2PI;
    }

    if( ! finite( marg ) || ! finite( varg ) )
        return 0;
    else if( varg < MIN_VAL )
        return exp( marg / 2 ) / MIN_VAL;
    else
        return exp( marg / 2 ) / sqrt( varg );
}
*/

//===========================================================================

void MixtureOfGaussians::estimateRadii()
{
	unsigned i, k;
	double   d, dist;

	//
	// estimate radii (squared euclidian distance to the nearest neighbour)
	//
	for (i = 0; i < size(); ++i) {
		dist = 0.;

		for (k = i + 1; k < size(); ++k) {
			d = ::sqrDistance(m[ i ], m[ k ]);
			if (dist == 0. || d < dist) dist = d;
		}

		if (dist == 0.) dist = 1.;

		v[ i ] = dist;
	}

	a = size() ? 1. / size() : 1;
}

void MixtureOfGaussians::kmc(const Array< double >& x,
							 double prec, unsigned maxiter)
{
	if (size() == 1)
		em(x);
	else {
		CodeBook::kmc(x, prec, maxiter);
		estimateRadii();
	}
}

void MixtureOfGaussians::kmc(const Array< double >& x,
							 const Array< double >& y,
							 double prec, unsigned maxiter)
{
	Array< double > z;
	join(x, y, z);
	kmc(z, prec, maxiter);
}

//===========================================================================

void MixtureOfGaussians::initialize(const Array< double >& x)
{
	//SIZE_CHECK( x.ndim( ) == 2 )

	unsigned i, j;
	Array< double > min(x[ 0 ]);
	Array< double > max(x[ 0 ]);
	Array< double > mu(x.dim(1));
	Array< double > ms(x.dim(1));

	mu = 0.;
	ms = 0.;

	for (i = x.dim(0); i--;) {
		for (j = x.dim(1); j--;) {
			mu(j) += x(i, j);
			ms(j) += Shark::sqr(x(i, j));
			if (x(i, j) < min(j)) min(j) = x(i, j);
			if (x(i, j) > max(j)) max(j) = x(i, j);
		}
	}

	for (j = x.dim(1); j--;) {
		mu(j) /= x.dim(0);
		ms(j)  = ms(j) / x.dim(0) - Shark::sqr(mu(j));
	}
	/*
	#ifndef NDEBUG
	    for( i = 0; i < ms.nelem( ); ++i )
	        ASSERT( ms.elem( i ) >= 0 )
	#endif
	*/

	a = 1. / size();

	for (i = size(); i--;) {
		for (j = min.dim(0); j--;) {
			m(i, j) = uni(min(j), max(j));
			v(i, j) = uni(ms(j) / 2, ms(j) * 2);
		}
	}
}

//===========================================================================

void MixtureOfGaussians::initialize
(
	const Array< double >& x,
	const Array< double >& y
)
{
	Array< double > z;
	join(x, y, z);
	initialize(z);
}

//===========================================================================

double MixtureOfGaussians::entropy2()
{
	unsigned i, j, k;
	double   t;
	Array< double > x(dim());
	Array< double > g(dim());
	Array< double > s(sqrt(v));

	for (t = 0., i = 0; i < monteCarloTrials; ++i) {
		for (j = dim(); j--;) {
			g(j) = gauss();
		}
		for (k = size(); k--;) {
			for (j = dim(); j--;) {
				x(j) = m(k, j) + g(j) * s(k, j);
			}
			t += a(k) * log(MixtureModel< double >::p(x));
		}
	}

	return -t / monteCarloTrials;
}

//===========================================================================

double MixtureOfGaussians::Renyi() const
{
	double u, s, sum = 0;
	unsigned i, j, k;

	for (i = 0; i < size(); ++i) {
		for (s = 1, k = 0; k < dim(); ++k)
			s *= v(i, k);
		sum += a(i) * a(i) / (pow(2 * SqrtPI, int(dim())) * sqrt(s));
	}

	for (i = 0; i < size(); ++i) {
		for (j = 0; j < i; ++j) {
			for (u = 0, s = 1, k = 0; k < dim(); ++k) {
				u += Shark::sqr(m(i, k) - m(j, k)) / (v(i, k) + v(j, k));
				s *= v(i, k) + v(j, k);
			}
			sum += a(i) * a(j) / (pow(Sqrt2PI, int(dim())) * sqrt(s))
				   * exp(-u / 2);
		}
	}

	return -log(sum);
}

//===========================================================================

double MixtureOfGaussians::kernelEntropy(const Array< double >& x) const
{
	double h = modelEntropy(x);

	return size() > 1 ? h / log(double(size())) : 1;
}

//===========================================================================

double MixtureOfGaussians::kernelEntropy(const Array< double >& x,
		const Array< double >& y) const
{
	//SIZE_CHECK( x.ndim( ) == 2 )

	if (y.nelem() == 0)
		return kernelEntropy(x);
	else {
		Array< double > z;
		join(x, y, z);

		return kernelEntropy(z);
	}
}

//===========================================================================

double MixtureOfGaussians::sqrDistance(const MixtureOfGaussians& mix) const
{
	//SIZE_CHECK( dim( ) == mix.dim( ) )

	unsigned i, j, k;
	double   uij, sij;
	double   sum = 0;

	for (i = 0; i < size(); ++i)
		for (j = 0; j < size(); ++j) {
			for (uij = 0, sij = 1, k = 0; k < dim(); ++k) {
				uij += Shark::sqr(m(i, k) - m(j, k)) / (v(i, k) + v(j, k));
				sij *= v(i, k) + v(j, k);
			}
			sum += a(i) * a(j) * exp(-uij / 2)
				   / (pow(Sqrt2PI, int(dim())) * sqrt(sij));
		}

	for (i = 0; i < size(); ++i)
		for (j = 0; j < mix.size(); ++j) {
			for (uij = 0, sij = 1, k = 0; k < dim(); ++k) {
				uij += Shark::sqr(m(i, k) - mix.m(j, k)) / (v(i, k) + mix.v(j, k));
				sij *= v(i, k) + mix.v(j, k);
			}
			sum -= 2 * a(i) * mix.a(j) * exp(-uij / 2)
				   / (pow(Sqrt2PI, int(dim())) * sqrt(sij));
		}

	for (i = 0; i < mix.size(); ++i)
		for (j = 0; j < mix.size(); ++j) {
			for (uij = 0, sij = 1, k = 0; k < mix.dim(); ++k) {
				uij += Shark::sqr(mix.m(i, k) - mix.m(j, k))
					   / (mix.v(i, k) + mix.v(j, k));
				sij *= mix.v(i, k) + mix.v(j, k);
			}
			sum += mix.a(i) * mix.a(j) * exp(-uij / 2)
				   / (pow(Sqrt2PI, int(dim())) * sqrt(sij));
		}

	return sum;
}

//===========================================================================

void MixtureOfGaussians::bsom_update
(
	const Array< double >& x,
	double alpha,
	double beta
)
{
	unsigned i, j;
	double   px;
	Array< double > pi(size());

	for (px = 0., i = size(); i--;)
		px += (pi(i) = pow(p(x, i) * a(i), beta));

	px = Shark::max(px, MIN_VAL);

	for (i = size(); i--;) {
		pi(i) = alpha * (pi(i) / px);
		a(i) += pi(i) - alpha * a(i);
		for (j = dim(); j--;) {
			m(i, j) += pi(i) * (x(j) - m(i, j));
			v(i, j) += pi(i) * (Shark::sqr(x(j) - m(i, j)) - v(i, j));
		}
	}
}

//===========================================================================
//
// Bayesian SOM for Gaussian mixtures
// according to Hujun Yin and Nigel M. Allison
//
void MixtureOfGaussians::bsom(const Array< double >& x,
							  double   alphai,
							  double   alphaf,
							  double   betai,
							  double   betaf,
							  unsigned tmax)
{
	for (unsigned t = 1; t <= tmax; ++t) {
		bsom_update
		(
			x[ Rng::discrete(0, x.dim(0) - 1)],
			alphai * pow(alphaf / alphai, double(t) / tmax),
			betai  * pow(betaf  / betai,  double(t) / tmax)
		);
	}
}

//===========================================================================

Array< double > MixtureOfGaussians::operator()()
{
	unsigned i, j;
	Array< double > x(dim());

	//
	// select gaussian
	//
	i = sampleModel();

	//
	// sample from selected gaussian
	//
	for (j = dim(); j--;) {
		x(j) = gauss(m(i, j), v(i, j));
	}

	return x;
}

//===========================================================================

static bool inHyperCube
(
	const Array< double >& x,
	const Array< double >& bound1,
	const Array< double >& bound2
)
{
	//SIZE_CHECK( x.samedim( bound1 ) && x.samedim( bound2 ) )
	//SIZE_CHECK( x.ndim( ) == 1 )

	for (unsigned i = 0; i < x.nelem(); ++i)
		if (x.elem(i) < Shark::min(bound1.elem(i), bound2.elem(i)) ||
				x.elem(i) > Shark::max(bound1.elem(i), bound2.elem(i)))
			return false;

	return true;
}

//===========================================================================

void MixtureOfGaussians::crossover(MixtureOfGaussians& mate, bool norm)
{
	//SIZE_CHECK( dim( ) == mate.dim( ) )

	unsigned i, j;

	//
	// min and max values
	//
	Array< double > min(dim());
	Array< double > max(dim());

	//
	// random points in [min,max] which define the hypercube
	//
	Array< double > start(dim());
	Array< double > stop(dim());

	//
	// local copies of the parents
	//
	MixtureOfGaussians mom(*this);
	MixtureOfGaussians dad(mate);

	//
	// find min and max values
	//
	min = max = mom.m[ 0 ];
	for (i = 0; i < mom.size(); ++i)
		for (j = 0; j < mom.dim(); ++j) {
			if (mom.m(i, j) < min(j)) min(j) = mom.m(i, j);
			if (mom.m(i, j) > max(j)) max(j) = mom.m(i, j);
		}
	for (i = 0; i < dad.size(); ++i)
		for (j = 0; j < dad.dim(); ++j) {
			if (dad.m(i, j) < min(j)) min(j) = dad.m(i, j);
			if (dad.m(i, j) > max(j)) max(j) = dad.m(i, j);
		}

	//
	// sample two random points
	//
	for (j = 0; j < mom.dim(); ++j) {
		start(j) = uni(min(j), max(j));
		stop(j) = uni(min(j), max(j));
	}

	//
	// clear all kernels in offsprings
	//
	resize(0, false);
	mate.resize(0, false);

	//
	// exchange elements lying in the hypercube [start,stop]
	//
	for (i = 0; i < mom.size(); ++i)
		if (inHyperCube(mom.m[ i ], start, stop))
			insertKernel(mom.a(i), mom.m[ i ], mom.v[ i ], false);
		else
			mate.insertKernel(mom.a(i), mom.m[ i ], mom.v[ i ], false);
	for (i = 0; i < dad.size(); ++i)
		if (inHyperCube(dad.m[ i ], start, stop))
			mate.insertKernel(dad.a(i), dad.m[ i ], dad.v[ i ], false);
		else
			insertKernel(dad.a(i), dad.m[ i ], dad.v[ i ], false);

	//
	// if one of the offsprings has not received any element then
	// choose a random one
	//
	if (size() == 0) {
		for (i = 0; i < max.dim(0); ++i)
			max(i) = Shark::sqr(max(i) - min(i));
		insertKernel(1, start, max, false);
	}
	if (mate.size() == 0) {
		for (i = 0; i < max.dim(0); ++i)
			max(i) = Shark::sqr(max(i) - min(i));
		mate.insertKernel(1, start, max, false);
	}

	if (norm) {
		normalize();
		mate.normalize();
	}
}

//===========================================================================

void MixtureOfGaussians::deleteSmallest(bool norm)
{
	deleteKernel(minVolume(), norm);
}

//===========================================================================

void MixtureOfGaussians::deleteRandom(bool norm)
{
	deleteKernel(unsigned(uni(0, size())), norm);
}

//===========================================================================

void MixtureOfGaussians::insertRandom(bool norm)
{
	double newa;
	Array< double > minm(dim());
	Array< double > maxm(dim());
	Array< double > newm(dim());
	Array< double > maxv(dim());
	Array< double > newv(dim());

	minm = maxm = m[ 0 ];
	maxv = v[ 0 ];

	for (unsigned i = 1; i < size(); ++i) {
		for (unsigned j = 0; j < dim(); ++j) {
			if (m(i, j) < minm(j)) minm(j) = m(i, j);
			if (m(i, j) > maxm(j)) maxm(j) = m(i, j);
			if (v(i, j) > maxv(j)) maxv(j) = v(i, j);
		}
	}

	newa = uni(0.5, 2.0) / size();

	for (unsigned j = 0; j < dim(); ++j) {
		newm(j) = uni(minm(j) / 1.5, maxm(j) * 1.5);
		newv(j) = uni(maxv(j) / 2.0, maxv(j) * 2.0);
	}

	insertKernel(newa, newm, newv, norm);
}

//===========================================================================

Array< double > MixtureOfGaussians::max() const
{
	Array< double > maxx;
	double mp, maxp = 0;

	for (unsigned i = 0; i < size(); ++i)
		if ((mp = MixtureModel< double >::p(m[ i ])) > maxp) {
			maxx = m[ i ];
			maxp = mp;
		}

	return maxx;
}

//===========================================================================

MixtureOfGaussians MixtureOfGaussians::
condDensity(const Array< double >& x) const
{
	//SIZE_CHECK( x.ndim( ) == 1 && x.nelem( ) <= dim( ) )

	double px = 0.;

	MixtureOfGaussians condD(size(), dim() - x.nelem());

	for (unsigned i = size(); i--;) {
		px += (condD.a(i) = p(x, i) * a(i));
		for (unsigned j = 0, k = x.nelem(); j < condD.dim(); ++j, ++k) {
			condD.m(i, j) = m(i, k);
			condD.v(i, j) = v(i, k);
		}
	}

	condD.a /= Shark::max(px, MIN_VAL);

	return condD;
}

//===========================================================================

void MixtureOfGaussians::deleteInput(unsigned k)
{
	//RANGE_CHECK( k < dim( ) )
	//if( k >= dim( ) ) return;

	unsigned i, j;
	MixtureOfGaussians orig(*this);

	unsigned dim1 = dim() - 1;
	CodeBook::resize(size(), dim1, false);
	v        .resize(size(), dim1, false);

	for (i = size(); i--;) {
		for (j = 0; j < k; ++j) {
			m(i, j) = orig.m(i, j);
			v(i, j) = orig.v(i, j);
		}
		for (; j < dim1; ++j) {
			m(i, j) = orig.m(i, j + 1);
			v(i, j) = orig.v(i, j + 1);
		}
	}
}

//===========================================================================

MixtureOfGaussians MixtureOfGaussians::
marginalDensity(const Array< unsigned >& idx) const
{
	MixtureOfGaussians marginalD(size(), idx.nelem());

	for (unsigned i = size(); i--;) {
		marginalD.a(i) = a(i);
		for (unsigned j = 0; j < idx.nelem(); ++j) {
			marginalD.m(i, j) = m(i, idx(j));
			marginalD.v(i, j) = v(i, idx(j));
		}
	}

	return marginalD;
}

MixtureOfGaussians MixtureOfGaussians::
marginalDensity(unsigned i) const
{
	Array< unsigned > idx(1);
	idx(0) = i;
	return marginalDensity(idx);
}

MixtureOfGaussians MixtureOfGaussians::
marginalDensity(unsigned i, unsigned j) const
{
	Array< unsigned > idx(2);
	idx(0) = i;
	idx(1) = j;
	return marginalDensity(idx);
}

MixtureOfGaussians MixtureOfGaussians::marginalDensity
(
	unsigned i,
	unsigned j,
	unsigned k
) const
{
	Array< unsigned > idx(3);
	idx(0) = i;
	idx(1) = j;
	idx(2) = k;
	return marginalDensity(idx);
}

/*
void MixtureOfGaussians::dumpIt( )
{
    double enti, entj, entij;

    entj = marginalDensity( dim( ) - 1 ).entropy( );
    for( unsigned i = 0; i < dim( ); ++i ) {
        enti  = marginalDensity( i ).entropy( );
        entij = marginalDensity( i, dim( ) - 1 ).entropy( );
	cout << i << '\t' << enti << '\t' << entij << '\t'
	     << ( enti + entj - entij ) << endl;
    }
}
*/

//===========================================================================

void MixtureOfGaussians::recall
(
	const Array< double >&	x,
	Array< double >&		y
) const
{
	//SIZE_CHECK( x.ndim( ) <= 2 )

	if (x.ndim() == 2) {
		Array< double > y0;
		y0 = condExpectation(x[ 0 ]);
		y.resize(x.dim(0), y0.nelem(), false);
		y[ 0 ] = y0;
		for (unsigned i = 1; i < x.dim(0); ++i)
			y[ i ] = condExpectation(x[ i ]);
	}
	else
		y = condExpectation(x);
}

//===========================================================================

Array< double > MixtureOfGaussians::condExpectation
(
	const Array< double >& x
) const
{
	//SIZE_CHECK( x.ndim( ) == 1 && x.nelem( ) <= dim( ) )

	unsigned i, j, k;
	double   px, pi;
	Array< double > y(dim() - x.nelem());

	px = 0.;
	y  = 0.;

	for (i = size(); i--;) {
		px += (pi = p(x, i) * a(i));
		for (j = 0, k = x.nelem(); j < y.nelem(); ++j, ++k)
			y(j) += m(i, k) * pi;
	}

	return y / Shark::max(px, MIN_VAL);
}

//===========================================================================
//
// bedingter Erwartungswert von linear transformierten Variablen
// z.Zt. wird nur der skalare Wert von E(x_{n}|x_1,...,x_{n-1})
// zurueckgeliefert
//
Array< double > MixtureOfGaussians::condExpectation
(
	const Array< double >& x,
	const Array< double >& A,
	const Array< double >& b
) const
{
	/*
	  SIZE_CHECK( x.ndim( ) == 1 && A.ndim( ) == 2 && b.ndim( ) == 1 )
	  SIZE_CHECK( A.dim( 0 ) == dim( ) &&
	A.dim( 1 ) == x.dim( 0 )+1 &&
	b.dim( 0 ) == dim( ) )
	*/

	unsigned i, j, k;
	double   c1, c2, c3, Ax, numer, denom, v, w;
	Array< double > y(1);

	numer = denom = 0;
	for (i = 0; i < size(); ++i) {
		c1 = c2 = c3 = 0;
		v = 1;
		for (j = 0; j < dim(); ++j) {
			Ax = 0;
			for (k = 0; k < x.dim(0); ++k)
				Ax += A(j, k) * x(k);
			Ax += b(j) - mean(i, j);
			w  = var(i, j);
			c1 += Shark::sqr(A(j, k)) / w;
			c2 += A(j, k) * Ax / w;
			c3 += Ax * Ax / w;
			v *= w;
		}

		w = prior(i) * exp(clip((c2 * c2 / c1 - c3) / 2, LOG_MIN_VAL, LOG_MAX_VAL)) / (sqrt(v * c1));
		numer += w * (-c2) / c1;
		denom += w;
	}

	y(0) = denom > 0 ? numer / denom : 1;
	return y;
}

//===========================================================================

Array< double > MixtureOfGaussians::condVariance
(
	const Array< double >& x
) const
{
	//SIZE_CHECK( x.ndim( ) == 1 && x.nelem( ) <= dim( ) )

	unsigned i, j, k;
	double   px, pi;
	Array< double > ey(condExpectation(x));
	Array< double > y(ey.nelem());

	px = 0.;
	y  = 0.;

	for (i = size(); i--;) {
		px += (pi = p(x, i) * a(i));
		for (j = 0, k = x.nelem(); j < y.nelem(); ++j, ++k)
			y(j) += (Shark::sqr(ey(j) - m(i, k)) + v(i, k)) * pi;
	}

	return y / Shark::max(px, MIN_VAL);
}

//===========================================================================

double MixtureOfGaussians::RenyiLikelihood(const Array< double >& x) const
{
	//SIZE_CHECK( x.ndim( ) == 1 || x.ndim( ) == 2 )

	if (x.ndim() == 1)
		return log(MixtureModel< double >::p(x));
	else {
		double psum = 0;

		for (unsigned i = 0; i < x.dim(0); ++i)
			psum += MixtureModel< double >::p(x[ i ]);

		return log(psum / x.dim(0));
	}
}

//===========================================================================

double MixtureOfGaussians::RenyiLikelihood(const Array< double >& x,
		const Array< double >& y) const
{
	//SIZE_CHECK( x.ndim( ) <= 2 )

	if (y.nelem() == 0)
		return RenyiLikelihood(x);
	else {
		Array< double > z;
		join(x, y, z);

		return RenyiLikelihood(z);
	}
}

//===========================================================================

double MixtureOfGaussians::jointLogLikelihood(const Array< double >& x,
		const Array< double >& y) const
{
	//SIZE_CHECK( x.ndim( ) <= 2 )

	if (y.nelem() == 0)
		return logLikelihood(x);
	else {
		Array< double > z;
		join(x, y, z);

		return logLikelihood(z);
	}
}

//===========================================================================

double MixtureOfGaussians::condLogLikelihood(const Array< double >& y,
		const Array< double >& x) const
{
	return jointLogLikelihood(x, y) - logLikelihood(x);
}

//===========================================================================

static void fmtMathematica(ostream& os, double x)
{
	ostringstream s;
	s.precision(16);
	s << x;
	string::size_type pos = s.str().find('e', 0);
	if (pos == string::npos)
	{
		os << s.str();
	}
	else
	{
		os << s.str().substr(0, pos) << " 10^(" << s.str().substr(pos + 1) << ")";
	}
}

void MixtureOfGaussians::writeMathematica(ostream& os) const
{
	unsigned i, j;

	os << "Clear[a,m,v,mix]\n\n";

	os << "a={";
	for (i = 0; i < size(); ++i) {
		if (i > 0)
			os << ',';
		fmtMathematica(os, a(i));
	}
	os << "}\n";

	os << "m={";
	for (i = 0; i < size(); ++i) {
		if (i > 0)
			os << ',';
		os << '{';
		for (j = 0; j < dim(); ++j) {
			if (j > 0)
				os << ',';
			fmtMathematica(os, m(i, j));
		}
		os << '}';
	}
	os << "}\n";

	os << "v={";
	for (i = 0; i < size(); ++i) {
		if (i > 0)
			os << ',';
		os << '{';
		for (j = 0; j < dim(); ++j) {
			if (j > 0)
				os << ',';
			fmtMathematica(os, v(i, j));
		}
		os << '}';
	}
	os << "}\n\n";

	os << "mix[ x_, a_, m_, v_ ] := \\\n"
	"  Sum[ a[[i]] Exp[ -Sum[ ( x[[j]] - m[[i,j]] )^2 / v[[i,j]], \\\n"
	"       {j,1,Length[m[[i]]]} ] / 2 ], {i,1,Length[a]} ] ;";

	if (dim() == 2)
		os << "\n\n"
		"Plot3D[ mix[ { x1, x2 }, a, m, v ], {x1,-3,3}, {x2,-3,3}, \\\n"
		"        PlotPoints->20, Axes->False, Boxed->False ]";

	os << endl;
}

//===========================================================================
//
// split a kernel which receives the most input data when sampled
//
void MixtureOfGaussians::splitBroadest(const Array< double >& x)
{
	unsigned j, k;
	vector< Array< double > > z;
	splitDataset(x, z);
	for (j = 1, k = 0; j < z.size(); ++j)
		if (z[ j ].nelem() > z[ k ].nelem())
			k = j;
	split(k, z);
}

void MixtureOfGaussians::splitBroadest(const Array< double >& x,
									   const Array< double >& y)
{
	if (y.nelem() == 0)
		splitBroadest(x);
	else {
		Array< double > z;
		join(x, y, z);
		splitBroadest(z);
	}
}

//===========================================================================
//
// split a randomly selected kernel
//
void MixtureOfGaussians::splitRandom(const Array< double >& x)
{
	vector< Array< double > > z;
	splitDataset(x, z);
	split(sampleModelUniform(), z);
}

void MixtureOfGaussians::splitRandom(const Array< double >& x,
									 const Array< double >& y)
{
	if (y.nelem() == 0)
		splitRandom(x);
	else {
		Array< double > z;
		join(x, y, z);
		splitRandom(z);
	}
}

//===========================================================================
//
// merge a kernel
//
void MixtureOfGaussians::mergeSmallest(const Array< double >& x)
{
	unsigned j, k;
	vector< Array< double > > z;
	splitDataset(x, z);
	for (j = 1, k = 0; j < z.size(); ++j)
		if (z[ j ].nelem() < z[ k ].nelem())
			k = j;
	merge(k, maxOverlap(k), z);
}

void MixtureOfGaussians::mergeSmallest(const Array< double >& x,
									   const Array< double >& y)
{
	if (y.nelem() == 0)
		mergeSmallest(x);
	else {
		Array< double > z;
		join(x, y, z);
		mergeSmallest(z);
	}
}

//===========================================================================
//
// merge a randomly selected kernel
//
void MixtureOfGaussians::mergeRandom(const Array< double >& x)
{
	unsigned k;
	vector< Array< double > > z;
	splitDataset(x, z);
	k = sampleModelUniform();
	merge(k, maxOverlap(k), z);
}

void MixtureOfGaussians::mergeRandom(const Array< double >& x,
									 const Array< double >& y)
{
	if (y.nelem() == 0)
		mergeRandom(x);
	else {
		Array< double > z;
		join(x, y, z);
		mergeRandom(z);
	}
}

//===========================================================================

void MixtureOfGaussians::em_mask(const Array< double >& x,
								 const Array< bool >& amask,
								 const Array< bool >& mmask,
								 const Array< bool >& vmask,
								 double   precision,
								 unsigned maxIteration,
								 double   minVariance)
{
	//SIZE_CHECK( x.ndim( ) == 2 )

	unsigned i, j, k, iteration;
	double   px, last, ll = -1e30;

	Array< double > pi(size());
	Array< double > newA(size());
	Array< double > newM(size(), dim());
	Array< double > newV(size(), dim());

	for (iteration = 0; iteration < maxIteration; ++iteration) {
		last = ll;
		ll   = 0.;
		newA = 0.;
		newM = 0.;
		newV = 0.;

		//
		// loop over all data vectors
		//
		for (k = x.dim(0); k--;) {
			for (px = 0., i = size(); i--;)
				px += (pi(i) = p(x[ k ], i) * a(i));
			pi /= Shark::max(px, MIN_VAL);
			ll += log(px);

			for (i = size(); i--;) {
				newA(i) += pi(i);
				for (j = dim(); j--;) {
					newM(i, j) += pi(i) * x(k, j);
					newV(i, j) += pi(i) * Shark::sqr(x(k, j));
				}
			}
		}

		for (i = size(); i--;) {
			for (k = dim(); k--;)
				if (newA(i) > MIN_VAL) {
					if (mmask(i, k))
						m(i, k) = newM(i, k) / newA(i);
					if (vmask(i, k)) {
						v(i, k) = newV(i, k) / newA(i)
								  - Shark::sqr(newM(i, k) / newA(i));
						if (v(i, k) < minVariance)
							v(i, k) = minVariance;
					}
				}

			if (amask(i))
				a(i) = newA(i) / x.dim(0);
		}

		if (2 *(ll - last) / fabs(ll + last) < precision)
			break;
	}

	normalize();
}

//===========================================================================

/*
#include <fastem.cpp>

void MixtureOfGaussians::em( const Array< double >& x,
                             double precision,
			     unsigned maxIteration,
			     double minVariance )
{
  //SIZE_CHECK( x.ndim( ) == 2 )

    unsigned iteration;
    double   last, ll = -1e30;

    if( size( ) == 1 ) {
        fastem( size( ), dim( ), x.dim( 0 ),
		a.elemvec( ), m.elemvec( ), v.elemvec( ), x.elemvec( ) );

	iteration = 0;
    } else {
	for( iteration = 0; iteration < maxIteration; ++iteration ) {
	    last = ll;
	    ll   = fastem( size( ), dim( ), x.dim( 0 ),
			   a.elemvec( ), m.elemvec( ), v.elemvec( ),
			   x.elemvec( ) );

	    if( 2 * ( ll - last ) / fabs( ll + last ) < precision )
	        break;
	}

	normalize( );
    }
}
*/

//===========================================================================

//
// epsilon: acceleration according to Redner and Walker 1984
//

//
// beta: deterministic annealing according to Ueda and Nakano 1998
//

bool MixtureOfGaussians::em
(
	const Array< double >& x,
	double   precision,
	unsigned maxIteration,
	double   minVariance,
	double   epsilon,
	double   beta
)
{
	//SIZE_CHECK( x.ndim( ) == 2 )

	unsigned i, j, k, iteration;
	double   px, last, ll = -1e30;

	if (size() == 1) {
		Array< double > newM(dim());
		Array< double > newV(dim());

		for (i = dim(); i--;) {
			newM(i) = 0.;
			newV(i) = 0.;
		}

		for (k = x.dim(0); k--;) {
			for (i = dim(); i--;) {
				newM(i) += x(k, i);
				newV(i) += Shark::sqr(x(k, i));
			}
		}

		a(0) = 1.;
		for (i = dim(); i--;) {
			m(0, i) = newM(i) / x.dim(0);
			v(0, i) = newV(i) / x.dim(0) - Shark::sqr(m(0, i));
		}
		return true;
	}
	else {
		Array< double > pi(size());
		Array< double > newA(size());
		Array< double > newM(size(), dim());
		Array< double > newV(size(), dim());

		for (iteration = 0; iteration < maxIteration; ++iteration) {
			last = ll;
			ll   = 0.;

			newA = 0.;
			newM = 0.;
			newV = 0.;

			//
			// loop over all data vectors
			//
			for (k = x.dim(0); k--;) {
				if (beta < 1) {
					for (px = 0., i = size(); i--;) {
						px += (pi(i) = pow(p(x[ k ], i) * a(i), beta));
					}
				}
				else {
					for (px = 0., i = size(); i--;) {
						px += (pi(i) = p(x[ k ], i) * a(i));
					}
				}

				ll += log(px = Shark::max(px, MIN_VAL));

				for (i = size(); i--;) {
					newA(i) += (pi(i) /= px);
					for (j = dim(); j--;) {
						newM(i, j) += pi(i) * x(k, j);
						newV(i, j) += pi(i) * Shark::sqr(x(k, j));
					}
				}
			}

			if (epsilon > 1) {
				m *= 1 - epsilon;
				v *= 1 - epsilon;
				for (i = size(); i--;) {
					for (k = dim(); k--;) {
						if (newA(i) > MIN_VAL) {
							m(i, k) += (newM(i, k) / newA(i)) * epsilon;
							v(i, k) += (newV(i, k) / newA(i)
										- Shark::sqr(m(i, k))) * epsilon;
							if (v(i, k) < minVariance)
								v(i, k) = minVariance;
						}
					}
				}

				a *= 1 - epsilon;
				a += (newA / double(x.dim(0))) * epsilon;
			}
			else {
				for (i = size(); i--;) {
					a(i) = newA(i) / x.dim(0);
					for (j = dim(); j--;) {
						if (newA(i) > MIN_VAL) {
							m(i, j) = newM(i, j) / newA(i);
							v(i, j) = Shark::max(newV(i, j) / newA(i)
											   - Shark::sqr(m(i, j)), minVariance);
						}
					}
				}
			}

			if (2 *(ll - last) / fabs(ll + last) < precision) {
				normalize();
				return true;
			}
		}

		normalize();
		return false;
	}
}

//===========================================================================

void MixtureOfGaussians::noisy_em
(
	const Array< double >& x,
	const Array< double >& vx,
	unsigned numSamples,
	double   precision,
	unsigned maxIteration,
	double   minVariance,
	double   epsilon,
	double   beta
)
{
	//SIZE_CHECK( x.ndim( ) == 2 )

	unsigned i, j, k, iteration;
	double   px, last, ll = -1e30;

	/*
	if( size( ) == 1 ) {
	Array< double > newM( dim( ) );
	Array< double > newV( dim( ) );

	    for( i = dim( ); i--; ) {
	 newM( i ) = 0.;
	 newV( i ) = 0.;
	    }

	    for( k = x.dim( 0 ); k--; )
	        for( i = dim( ); i--; ) {
	     newM( i ) += x( k, i );
	     newV( i ) += Shark::sqr( x( k, i ) );
	        }

	a( 0 ) = 1.;
	    for( i = dim( ); i--; ) {
	        m( 0, i ) = newM( i ) / x.dim( 0 );
	        v( 0, i ) = newV( i ) / x.dim( 0 ) - Shark::sqr( m( 0, i ) );
	    }
	} else*/ {
		Array< double > pi(size());
		Array< double > newA(size());
		Array< double > newM(size(), dim());
		Array< double > newV(size(), dim());

		for (iteration = 0; iteration < maxIteration; ++iteration) {
			last = ll;
			ll   = 0.;

			newA = 0.;
			newM = 0.;
			newV = 0.;

			//
			// loop over all data vectors
			//
			for (k = x.dim(0); k--;) {
				if (beta < 1)
					for (px = 0., i = size(); i--;)
						px += (pi(i) = pow(p(x[ k ], i) * a(i), beta));
				else
					for (px = 0., i = size(); i--;)
						px += (pi(i) = p(x[ k ], i) * a(i));

				ll += log(px = Shark::max(px, MIN_VAL));

				for (i = size(); i--;) {
					newA(i) += (pi(i) /= px);
					for (j = dim(); j--;) {
						newM(i, j) += pi(i) * x(k, j);
						newV(i, j) += pi(i) * Shark::sqr(x(k, j));
					}
				}
			}

			if (epsilon > 1) {
				m *= 1 - epsilon;
				v *= 1 - epsilon;
				for (i = size(); i--;)
					for (k = dim(); k--;)
						if (newA(i) > MIN_VAL) {
							m(i, k) += (newM(i, k) / newA(i)) * epsilon;
							v(i, k) += (newV(i, k) / newA(i)
										- Shark::sqr(m(i, k))) * epsilon;
							if (v(i, k) < minVariance)
								v(i, k) = minVariance;
						}

				a *= 1 - epsilon;
				a += (newA / double(x.dim(0))) * epsilon;
			}
			else {
				for (i = size(); i--;) {
					a(i) = newA(i) / x.dim(0);
					for (j = dim(); j--;)
						if (newA(i) > MIN_VAL) {
							m(i, j) = newM(i, j) / newA(i);
							v(i, j) = Shark::max(newV(i, j) / newA(i)
											   - Shark::sqr(m(i, j)), minVariance);
						}
				}
			}

			if (2 *(ll - last) / fabs(ll + last) < precision)
				break;
		}

		normalize();
	}
}

//===========================================================================
//
// Combination of EM and jack-knife estimation (replaces the original M step)
//
// according to Z. R. Yang & S. Chen (1998).
//              Robust maximum likelihood training of heteroscedastic
//              probabilistic neural networks.
//              Neural Networks 11(4), pp. 739-747. 1998.
//
// For the sake of computational efficiency the variances are not estimated
// directly but by the relation V[x] = E[x^2] - E[x]^2.
// This allows the use of an one-pass estimation of the variances.
// (This may be bought at the expense of a decreased robustness of the
//  estimation.)
//
// (epsilon ignored)
//
void MixtureOfGaussians::jack_knife_em(const Array< double >& x,
									   double   precision,
									   unsigned maxIteration,
									   double   minVariance,
									   double   beta)
{
	//SIZE_CHECK( x.ndim( ) == 2 )

	unsigned i, j, k, iteration;
	double   px, last, ll = -1e30;

	if (size() == 1) {
		Array< double > newM(dim());
		Array< double > newV(dim());

		for (i = dim(); i--;) {
			newM(i) = 0.;
			newV(i) = 0.;
		}

		for (k = x.dim(0); k--;)
			for (i = dim(); i--;) {
				newM(i) += x(k, i);
				newV(i) += Shark::sqr(x(k, i));
			}

		a(0) = 1.;
		for (i = dim(); i--;) {
			m(0, i) = newM(i) / x.dim(0);
			v(0, i) = newV(i) / x.dim(0) - Shark::sqr(m(0, i));
		}

		iteration = 0;
	}
	else {
		double diffa;
		Array< double > pi(size());
		Array< double > newA(size());
		Array< double > newM(size(), dim());
		Array< double > newV(size(), dim());
		Array< double > jackA(size());
		Array< double > jackM(size(), dim());
		Array< double > jackV(size(), dim());

		for (iteration = 0; iteration < maxIteration; ++iteration) {
			last = ll;
			ll   = 0.;

			newA = 0.;
			newM = 0.;
			newV = 0.;

			//
			// loop over all data vectors
			//
			for (k = x.dim(0); k--;) {
				if (beta < 1)
					for (px = 0., i = size(); i--;)
						px += (pi(i) = pow(p(x[ k ], i) * a(i), beta));
				else
					for (px = 0., i = size(); i--;)
						px += (pi(i) = p(x[ k ], i) * a(i));

				ll += log(px = Shark::max(px, MIN_VAL));

				for (i = size(); i--;) {
					newA(i) += (pi(i) /= px);
					for (j = dim(); j--;) {
						newM(i, j) += pi(i) * x(k, j);
						newV(i, j) += pi(i) * Shark::sqr(x(k, j));
					}
				}
			}

			//
			// loop over all data vectors again to build the
			// jack-knife estimate
			//
			for (i = size(); i--;) {
				jackA(i) = newA(i);
				for (j = dim(); j--;) {
					jackM(i, j) = x.dim(0) * newM(i, j) / newA(i);
					jackV(i, j) = x.dim(0) * newV(i, j) / newA(i);
				}
			}

			for (k = x.dim(0); k--;) {
				if (beta < 1)
					for (px = 0., i = size(); i--;)
						px += (pi(i) = pow(p(x[ k ], i) * a(i), beta));
				else
					for (px = 0., i = size(); i--;)
						px += (pi(i) = p(x[ k ], i) * a(i));

				px = Shark::max(px, MIN_VAL);

				for (i = size(); i--;) {
					diffa       = (newA(i) - (pi(i) /= px));
					jackA(i) -= diffa / x.dim(0);
					diffa      *= x.dim(0) / (x.dim(0) - 1.);
					for (j = dim(); j--;) {
						jackM(i, j) -= (newM(i, j) - pi(i) * x(k, j)) / diffa;
						jackV(i, j) -= (newV(i, j) - pi(i) * Shark::sqr(x(k, j))) / diffa;
					}
				}
			}

			for (i = size(); i--;) {
				a(i) = jackA(i);
				for (j = dim(); j--;) {
					m(i, j) = jackM(i, j);
					v(i, j) = Shark::max(jackV(i, j) - Shark::sqr(m(i, j)),
									   minVariance);
				}
			}

			if (2 *(ll - last) / fabs(ll + last) < precision)
				break;
		}

		normalize();
	}
}

//===========================================================================

void MixtureOfGaussians::annealing_em(const Array< double >& x,
									  double   precision,
									  unsigned maxIteration,
									  double   minVariance,
									  double   epsilon,
									  double   minBeta,
									  double   incBeta)
{
	for (double beta = minBeta; beta <= 1; beta *= incBeta)
		em(x, precision, maxIteration, minVariance, epsilon, beta);
}

//===========================================================================

void MixtureOfGaussians::stochastic_em(const Array< double >& x,
									   double   precision,
									   unsigned maxIteration,
									   double   minVariance,
									   double   epsilon)
{
	//SIZE_CHECK( x.ndim( ) == 2 )

	unsigned i, j, k, iteration;
	double   last, ll = -1e30;

	if (size() == 1)
		em(x);
	else {
		Array< double > pi(size());
		Array< double > newA(size());
		Array< double > newM(size(), dim());
		Array< double > newV(size(), dim());

		for (iteration = 0; iteration < maxIteration; ++iteration) {
			last = ll;
			ll   = 0.;
			newA = 0.;
			newM = 0.;
			newV = 0.;

			//
			// loop over all data vectors
			//
			for (k = x.dim(0); k--;) {
				i = sampleModel(x[ k ]);
				newA(i)++;
				for (j = dim(); j--;) {
					newM(i, j) += x(k, j);
					newV(i, j) += Shark::sqr(x(k, j));
				}
			}

			ll = logLikelihood(x);

			if (epsilon > 1) {
				m *= 1 - epsilon;
				v *= 1 - epsilon;
				for (i = size(); i--;)
					for (k = dim(); k--;)
						if (newA(i) > MIN_VAL) {
							m(i, k) += (newM(i, k) / newA(i)) * epsilon;
							v(i, k) += (newV(i, k) / newA(i)
										- Shark::sqr(m(i, k))) * epsilon;
							if (v(i, k) < minVariance)
								v(i, k) = minVariance;
						}

				a *= 1 - epsilon;
				a += (newA / double(x.dim(0))) * epsilon;
			}
			else {
				for (i = size(); i--;)
					for (k = dim(); k--;)
						if (newA(i) > MIN_VAL) {
							m(i, k) = newM(i, k) / newA(i);
							v(i, k) = newV(i, k) / newA(i)
									  - Shark::sqr(m(i, k));
							if (v(i, k) < minVariance)
								v(i, k) = minVariance;
						}

				a = newA / double(x.dim(0));
			}

			if (2 *(ll - last) / fabs(ll + last) < precision)
				break;
		}

		normalize();
	}
}

//===========================================================================

double MixtureOfGaussians::deficientP(const Array< double >& x,
									  const Array< double >& xvar,
									  unsigned i, double maxV) const
{
	/*
	  SIZE_CHECK( x.ndim( ) == 1 && x.nelem( ) <= dim( ) )
	  SIZE_CHECK( x.samedim( xvar ) )
	*/

	unsigned  d = 0;
	double marg = 0.;
	double varg = 1.;

	for (unsigned j = x.dim(0); j--;) {
		if (xvar(j) < maxV) {
			marg -= Shark::sqr(x(j) - m(i, j)) /
					Shark::max(v(i, j) + xvar(j), MIN_VAL);
			if (varg < SQRT_MAX_VAL &&
					v(i, j) + xvar(j) < SQRT_MAX_VAL)
				varg *= v(i, j) + xvar(j);
			++d;
		}
	}

	return exp(marg / 2) / sqrt(Shark::max(varg, MIN_VAL) * pow(2 * M_PI, int(d)));
}

//===========================================================================

double MixtureOfGaussians::deficientP(const Array< double >& x,
									  const Array< double >& xvar,
									  double maxV) const
{
	double p = 0;

	for (unsigned i = 0; i < size(); ++i)
		p += a(i) * deficientP(x, xvar, i, maxV);

	return p;
}

//===========================================================================

double MixtureOfGaussians::deficientLogLikelihood(const Array< double >& x,
		const Array< double >& xvar,
		double maxV) const
{
	//SIZE_CHECK( x.ndim( ) == 2 )

	double l = 0;

	for (unsigned k = x.dim(0); k--;)
		l += log(Shark::max(deficientP(x[ k ], xvar[ k ], maxV), 1e-100));        // !!!

	return l;
}

//===========================================================================

void MixtureOfGaussians::em_deficient(const Array< double >& x,
									  const Array< double >& xvar,
									  double   precision,
									  unsigned maxIteration,
									  double   minVariance,
									  double   maxVariance)
{
	//SIZE_CHECK( x.ndim( ) == 2 )

	unsigned i, j, k, iteration;
	double   px, last, ll = -1e30;
	double   S, C, D;

	Array< double > pi(size());
	Array< double > newA(size());
	Array< double > newM(size(), dim());
	Array< double > newV(size(), dim());

	for (iteration = 0; iteration < maxIteration; ++iteration) {
		last = ll;
		ll   = 0.;
		newA = 0.;
		newM = 0.;
		newV = 0.;

		//
		// loop over all data vectors
		//
		for (k = x.dim(0); k--;) {
			for (px = 0., i = size(); i--;)
				px += (pi(i) = deficientP(x[ k ], xvar[ k ], i, maxVariance) * a(i));

			ll += log(px = Shark::max(px, MIN_VAL));
			pi /= px;

			for (i = 0; i < size(); ++i) {
				newA(i) += pi(i);
				for (j = 0; j < dim(); ++j) {
					if (xvar(k, j) >= maxVariance)
						newM(i, j) += pi(i) * m(i, j);
					else {
						S = var(i, j) + xvar(k, j);
						C = (var(i, j) * x(k, j) + xvar(k, j) * m(i, j)) / S;
						D = var(i, j) * xvar(k, j) / S;

						newM(i, j) += pi(i) * C;
					}
				}
			}
		}

		for (i = size(); i--;)
			for (k = dim(); k--;)
				if (newA(i) > MIN_VAL)
					m(i, k) = newM(i, k) / newA(i);

		//
		// loop over all data vectors
		//
		for (k = x.dim(0); k--;) {
			for (px = 0., i = size(); i--;)
				px += (pi(i) = deficientP(x[ k ], xvar[ k ], i, maxVariance) * a(i));

			ll += log(px = Shark::max(px, MIN_VAL));
			pi /= px;

			for (i = 0; i < size(); ++i) {
				for (j = 0; j < dim(); ++j) {
					if (xvar(k, j) >= maxVariance)
						newV(i, j) += pi(i) * var(i, j);
					else {
						S = var(i, j) + xvar(k, j);
						C = (var(i, j) * x(k, j) + xvar(k, j) * m(i, j)) / S;
						D = var(i, j) * xvar(k, j) / S;

						newV(i, j) += pi(i) * (D + Shark::sqr(C - m(i, j)));
					}
				}
			}
		}

		for (i = size(); i--;)
			for (k = dim(); k--;)
				if (newA(i) > MIN_VAL) {
					v(i, k) = newV(i, k) / newA(i);
					if (v(i, k) < minVariance)
						v(i, k) = minVariance;
				}

		a = newA / double(x.dim(0));

		if (2 *(ll - last) / fabs(ll + last) < precision)
			break;
	}

	normalize();
}

//===========================================================================

//
// new curvature
//
/*
static inline bool equal( double x, double y )
{
    const double eps = 1e-10;

    return 2 * fabs( ( x - y ) / ( x + y ) ) <= eps;
}

static inline bool equal( const Array< double >& x,
			  const Array< double >& y )
{
    for( unsigned i = 0; i < x.nelem( ); ++i )
        if( ! equal( x.elem( i ), y.elem( i ) ) )
	    return false;
    return true;
}
*/

double MixtureOfGaussians::A(unsigned r) const
{
	double sum = 0;

	for (unsigned i = 0; i < size(); ++i) {
		for (unsigned j = 0; j < i; ++j)
			sum += 2 * a(i) * a(j) * E(i, j) * U(i, j, r);
		sum += a(i) * a(i) * E(i, i) * U(i, i, r);
	}

	return sum;
}

double MixtureOfGaussians::B(unsigned r, unsigned t) const
{
	double sum = 0;

	for (unsigned i = 0; i < size(); ++i) {
		for (unsigned j = 0; j < i; ++j)
			sum += 2 * a(i) * a(j) * E(i, j) * V(i, j, r, t);
		sum += a(i) * a(i) * E(i, i) * V(i, i, r, t);
	}

	return sum;
}

double MixtureOfGaussians::E(unsigned i, unsigned j) const
{
	double s = 1, u = 0;

	for (unsigned k = 0; k < dim(); ++k) {
		u += Shark::sqr(m(i, k) - m(j, k)) / (v(i, k) + v(j, k));
		s *= v(i, k) + v(j, k);
	}

	return exp(-u / 2) / (pow(Sqrt2PI, int(dim())) * sqrt(s));
}

double MixtureOfGaussians::U(unsigned i, unsigned j, unsigned r) const
{
	return (3 * Shark::sqr(v(i, r) + v(j, r))
			- 6 *(v(i, r) + v(j, r)) * Shark::sqr(m(i, r) - m(j, r))
			+ pow(m(i, r) - m(j, r), 4)) /
		   pow(v(i, r) + v(j, r), 4);
}

double MixtureOfGaussians::V(unsigned i, unsigned j,
							 unsigned r, unsigned t) const
{
	return (v(i, r) + v(j, r) - Shark::sqr(m(i, r) - m(j, r)))
		   *(v(i, t) + v(j, t) - Shark::sqr(m(i, t) - m(j, t))) /
		   (Shark::sqr(v(i, r) + v(j, r)) * Shark::sqr(v(i, t) + v(j, t)));
}

inline double delta(unsigned i, unsigned j)
{
	return i == j ? 1. : 0.;
}

double MixtureOfGaussians::Delta1(unsigned s, unsigned j, unsigned q) const
{
	return (12*(v(j, q) + v(s, q)) - 4*Shark::sqr(m(j, q) - m(s, q)))
		   / pow(v(j, q) + v(s, q), 3);
}

double MixtureOfGaussians::Delta2(unsigned s, unsigned j, unsigned q) const
{
	return (36*(v(j, q) + v(s, q))*Shark::sqr(m(j, q) - m(s, q))
			- 12*Shark::sqr(v(j, q) + v(s, q))
			- 8*pow(m(j, q) - m(s, q), 4))
		   / pow(v(j, q) + v(s, q), 4);
}

double MixtureOfGaussians::Delta3(unsigned s, unsigned j,
								  unsigned q, unsigned k) const
{
	return 2 *(v(j, k) + v(s, k) - Shark::sqr(m(j, k) - m(s, k)))
		   / ((v(j, q) + v(s, q))*Shark::sqr(v(j, k) + v(s, k)));
}

double MixtureOfGaussians::Delta4(unsigned s, unsigned j,
								  unsigned q, unsigned k) const
{
	return Delta3(s, j, q, k)
		   *(2*Shark::sqr(m(j, q) - m(s, q)) / (v(j, q) + v(s, q)) - 1);
}

double MixtureOfGaussians::dAda(unsigned r, unsigned s) const
{
	double sum = 0;

	for (unsigned j = 0; j < size(); ++j)
		sum += a(j) * E(s, j) * U(s, j, r);

	return 2 * sum;
}

double MixtureOfGaussians::dAdb(unsigned r, unsigned s) const
{
	double sum = 0;

	for (unsigned j = 0; j < size(); ++j)
		sum += a(j) * E(s, j) * U(s, j, r);

	return 2 * a(s) *(sum - A(r));
}

double MixtureOfGaussians::dAdm(unsigned r, unsigned s, unsigned q) const
{
	double sum = 0;

	for (unsigned j = 0; j < size(); ++j)
		sum += a(j) * E(s, j)
			   * ((m(j, q) - m(s, q)) / (v(j, q) + v(s, q)))
			   * (U(s, j, r) + delta(r, q) * Delta1(s, j, q));

	return 2 * a(s) * sum;
}

double MixtureOfGaussians::dAds(unsigned r, unsigned s, unsigned q) const
{
	double sum = 0;

	for (unsigned j = 0; j < size(); ++j)
		sum += a(j) * E(s, j)
			   * (sqrt(v(s, q)) / (v(j, q) + v(s, q)))
			   * (U(s, j, r) * (Shark::sqr(m(j, q) - m(s, q)) / (v(j, q) + v(s, q)) - 1)
				  + delta(r, q) * Delta2(s, j, q));

	return 2 * a(s) * sum;
}

double MixtureOfGaussians::dBda(unsigned r, unsigned t, unsigned s) const
{
	double sum = 0;

	for (unsigned j = 0; j < size(); ++j)
		sum += a(j) * E(s, j) * V(s, j, r, t);

	return 2 * sum;
}

double MixtureOfGaussians::dBdb(unsigned r, unsigned t, unsigned s) const
{
	double sum = 0;

	for (unsigned j = 0; j < size(); ++j)
		sum += a(j) * E(s, j) * V(s, j, r, t);

	return 2 * a(s) *(sum - B(r, t));
}

double MixtureOfGaussians::dBdm(unsigned r, unsigned t,
								unsigned s, unsigned q) const
{
	double sum = 0;

	for (unsigned j = 0; j < size(); ++j)
		sum += a(j) * E(s, j)
			   * ((m(j, q) - m(s, q)) / (v(j, q) + v(s, q)))
			   * (V(s, j, r, t)
				  + delta(q, r) * Delta3(s, j, q, t)
				  + delta(q, t) * Delta3(s, j, q, r));

	return 2 * a(s) * sum;
}

double MixtureOfGaussians::dBds(unsigned r, unsigned t,
								unsigned s, unsigned q) const
{
	double sum = 0;

	for (unsigned j = 0; j < size(); ++j)
		sum += a(j) * E(s, j)
			   * (sqrt(v(s, q)) / (v(j, q) + v(s, q)))
			   * (V(s, j, r, t) * (Shark::sqr(m(j, q) - m(s, q)) / (v(j, q) + v(s, q)) - 1)
				  + delta(q, r) * Delta4(s, j, q, t)
				  + delta(q, t) * Delta4(s, j, q, r));

	return 2 * a(s) * sum;
}

double MixtureOfGaussians::curvature() const
{
	double sum = 0;

	for (unsigned r = 0; r < dim(); ++r) {
		for (unsigned t = 0; t < r; ++t)
			sum += 2 * B(r, t);
		sum += A(r);
	}

	return sum;
}

/*
void MixtureOfGaussians::gradientCurve( )
{
    for( unsigned s = 0; s < size( ); ++s ) {
        for( unsigned r = 0; r < dim( ); ++r ) {
	    for( unsigned t = 0; t < r; ++t ) {
	        db( s ) += 2 * dBdb( r, t, s ) * lambda;
		for( unsigned q = 0; q < dim( ); ++q ) {
		    dm( s, q ) += 2 * dBdm( r, t, s, q ) * lambda;
		    ds( s, q ) += 2 * dBds( r, t, s, q ) * lambda;
		}
	    }
	    db( s ) += dAdb( r, s ) * lambda;
	    for( unsigned q = 0; q < dim( ); ++q ) {
	        dm( s, q ) += dAdm( r, s, q ) * lambda;
		ds( s, q ) += dAds( r, s, q ) * lambda;
	    }
	}
    }
}
*/

void MixtureOfGaussians::dCurve(Array< double >& db,
								Array< double >& dm,
								Array< double >& ds) const
{
	db.resize(size(), false);
	dm.resize(size(), dim(), false);
	ds.resize(size(), dim(), false);

	db = 0;
	dm = 0;
	ds = 0;

	for (unsigned s = 0; s < size(); ++s) {
		for (unsigned r = 0; r < dim(); ++r) {
			for (unsigned t = 0; t < r; ++t) {
				db(s) += 2 * dBdb(r, t, s);
				for (unsigned q = 0; q < dim(); ++q) {
					dm(s, q) += 2 * dBdm(r, t, s, q);
					ds(s, q) += 2 * dBds(r, t, s, q);
				}
			}
			db(s) += dAdb(r, s);
			for (unsigned q = 0; q < dim(); ++q) {
				dm(s, q) += dAdm(r, s, q);
				ds(s, q) += dAds(r, s, q);
			}
		}
	}
}

void MixtureOfGaussians::dCurveams(Array< double >& da,
								   Array< double >& dm,
								   Array< double >& ds) const
{
	da.resize(size(), false);
	dm.resize(size(), dim(), false);
	ds.resize(size(), dim(), false);

	da = 0;
	dm = 0;
	ds = 0;

	for (unsigned s = 0; s < size(); ++s) {
		for (unsigned r = 0; r < dim(); ++r) {
			for (unsigned t = 0; t < r; ++t) {
				da(s) += 2 * dBda(r, t, s);
				for (unsigned q = 0; q < dim(); ++q) {
					dm(s, q) += 2 * dBdm(r, t, s, q);
					ds(s, q) += 2 * dBds(r, t, s, q);
				}
			}
			da(s) += dAda(r, s);
			for (unsigned q = 0; q < dim(); ++q) {
				dm(s, q) += dAdm(r, s, q);
				ds(s, q) += dAds(r, s, q);
			}
		}
	}
}

//===========================================================================

double MixtureOfGaussians::mse
(
	const Array< double >& x,
	const Array< double >& y
) const
{
	//SIZE_CHECK( x.ndim( ) <= 2 )

	Array< double > yy;
	double e = 0.;

	if (x.ndim() == 2) {
		for (unsigned i = 0; i < x.dim(0); i++) {
			recall(x[ i ], yy);
			e += ::sqrDistance(y[ i ], yy);
		}
		return e / x.dim(0);
	}
	else {
		recall(x, yy);
		return ::sqrDistance(y, yy);
	}
}

//===========================================================================

// Computes the overall expectation of the model
// \f$
//     \mu_{overall} = \sum_{i=0}^n \alpha_i \mu_i
// \f$
Array< double > MixtureOfGaussians::overallExpectation() const
{
	Array< double > mean;

	if (size() > 0) {
		mean = a(0) * m[ 0 ];
		for (unsigned i = size() - 1; i--;) {
			mean += a(i) * m[ i ];
		}
	}

	return mean;
}

//===========================================================================

// Computes the overall variance of the model
Array< double > MixtureOfGaussians::overallVariance() const
{
	Array< double > mean;
	Array< double > variance;

	if (size() > 0) {
		mean = overallExpectation();

		variance = a(0) * (v[ 0 ] + Shark::sqr(m[ 0 ] - mean));
		for (unsigned i = size() - 1; i--;) {
			variance += a(i) * (v[ i ] + Shark::sqr(m[ i ] - mean));
		}
	}

	return variance;
}

//===========================================================================

