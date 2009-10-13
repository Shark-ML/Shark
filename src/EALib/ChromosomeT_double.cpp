/*!
*  \file ChromosomeT_double.cpp
*
*  \author  M. Kreutz
*
*  \brief Functions for real-valued  chromosomes.
*
*  \date    1995
*
*  \par
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
*      EALib
*
*
*  <BR>
*
*
*  <BR><HR>
*  This file is part of the EALib. This library is free software;
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

#include <LinAlg/LinAlg.h>
#include <EALib/ChromosomeT.h>

//===========================================================================

ChromosomeT< double >::~ChromosomeT()
{}

//===========================================================================

inline unsigned long pow2(unsigned n)
{
	return 1UL << n;
}

void ChromosomeT< double >::decodeBinary(const Chromosome& chrom,
		const Interval&   range,
		unsigned          nbits,
		bool              useGray)
{
	SIZE_CHECK(chrom.size() % nbits == 0)

	const std::vector< bool >& src = dynamic_cast< const std::vector< bool >& >(chrom);

	double stepSize = (pow2(nbits) - 1) / range.width();
	unsigned i, j, k;
	unsigned long l, m;

	resize(src.size() / nbits);


	for (j = size(), k = src.size(); j--;) {
		if (useGray)
			for (l = m = 0, i = nbits; i--;)
				l = (l << 1) | (m ^= (src[ --k ] ? 1 : 0));
		else
			for (l = 0, i = nbits; i--;)
				l = (l << 1) | (src[ --k ] ? 1 : 0);

		(*this)[ j ] = l / stepSize + range.lowerBound();
	}
}

//===========================================================================

void ChromosomeT< double >::accumulate(const std::vector< double >& acc, double c)
{
	SIZE_CHECK(size() == acc.size())

	for (unsigned i = size(); i--;)
		(*this)[ i ] = (1 - c) * (*this)[ i ] + c * acc[ i ];
}

//===========================================================================

void ChromosomeT< double >::accumulate(const Chromosome& acc, double c)
{
	accumulate(dynamic_cast< const std::vector< double >& >(acc), c);
}

//===========================================================================

void ChromosomeT< double >::mutateNormal(double variance)
{
	for (unsigned i = size(); i--;)
		(*this)[ i ] += Rng::gauss(0, variance);
}

//===========================================================================

void ChromosomeT< double >::mutateNormal(const std::vector< double >& variances, bool cycle)
{
	RANGE_CHECK(variances.size() <= size())

	for (unsigned i = cycle ? size() : variances.size(); i--;)
		(*this)[ i ] += Rng::gauss(0, variances[ i % variances.size()]);
}

//===========================================================================

void ChromosomeT< double >::mutateNormal(const ChromosomeT< double >& variances,
		bool cycle)
{
	mutateNormal(static_cast< const std::vector< double >& >(variances), cycle);
}

//===========================================================================

void ChromosomeT< double >::mutateNormal(const Chromosome& variances,
		bool cycle)
{
	mutateNormal(dynamic_cast< const std::vector< double >& >(variances), cycle);
}

//===========================================================================

void ChromosomeT< double >::mutateCauchy(double scale)
{
	for (unsigned i = size(); i--;)
		(*this)[ i ] += Rng::cauchy() * scale;
}

//===========================================================================

void ChromosomeT< double >::mutateCauchy(const std::vector< double >& scale,
		bool cycle)
{
	RANGE_CHECK(scale.size() <= size())

	for (unsigned i = cycle ? size() : scale.size(); i--;)
		(*this)[ i ] += Rng::cauchy() * scale[ i % scale.size()];
}

//===========================================================================

void ChromosomeT< double >::mutateCauchy(const ChromosomeT< double >& scale,
		bool cycle)
{
	mutateNormal(static_cast< const std::vector< double >& >(scale), cycle);
}

//===========================================================================

void ChromosomeT< double >::mutateCauchy(const Chromosome& scale,
		bool cycle)
{
	mutateNormal(dynamic_cast< const std::vector< double >& >(scale), cycle);
}

//===========================================================================

void ChromosomeT< double >::mutateNormalRotAngles(const Chromosome& sigma,
		const Chromosome& alpha)
{
	mutateNormalRotAngles(dynamic_cast< const std::vector< double >& >(sigma),
						  dynamic_cast< const std::vector< double >& >(alpha));
}

//===========================================================================

void
ChromosomeT< double >::mutateNormalRotAngles(const std::vector< double >& sigma_sqr,
		const std::vector< double >& alpha)
{
	RANGE_CHECK(sigma_sqr.size() > 0 || size() == 0)

	register unsigned m, i, j, k;

	std::vector< double >& v = *this;
	std::vector< double > z(v.size());

	//
	// create random vector z
	//
	for (m = 0; m < z.size() && m < sigma_sqr.size(); m++)
		z[ m ] = Rng::gauss(0, sigma_sqr[ m ]);

	for (; m < z.size(); m++)
		z[ m ] = Rng::gauss(0, sigma_sqr[ sigma_sqr.size() - 1 ]);

	//
	// rotate vector z
	//
	for (k = 0, j = 1; j < m && k < alpha.size(); j++)
		for (i = 0; i < m - j && k < alpha.size(); i++, k++) {
			double sina = sin(alpha[ k ]);
			double cosa = cos(alpha[ k ]);
			double t    = cosa * z[ i ] - sina * z[ i + j ];
			z[ i + j ]  = sina * z[ i ] + cosa * z[ i + j ];
			z[ i     ]  = t;
		}

	//
	// mutate *this with rotated random vector z
	//
	for (m = v.size(); m--;)
		v[ m ] += z[ m ];
}

//===========================================================================

void ChromosomeT< double >::mutateLogNormal(double overallVariance,
		double indivVariance)
{
	double overall = Rng::gauss(0, overallVariance);

	for (unsigned i = size(); i--;)
		(*this)[ i ] *= exp(overall + Rng::gauss(0, indivVariance));
}

//===========================================================================

void
ChromosomeT< double >::recombineIntermediate(const Chromosome& dadChrom,
		const Chromosome& momChrom)
{
	SIZE_CHECK(dadChrom.size() == momChrom.size())

	const std::vector< double >& dad = dynamic_cast< const std::vector< double >& >(dadChrom);
	const std::vector< double >& mom = dynamic_cast< const std::vector< double >& >(momChrom);

	resize(dad.size());

	for (unsigned i = size(); i--;)
		(*this)[ i ] = (dad[ i ] + mom[ i ]) / 2.;
}

//===========================================================================
//
// generalized intermediate recombination
//
void
ChromosomeT< double >::recombineGenIntermediate(const Chromosome& dadChrom,
		const Chromosome& momChrom)
{
	SIZE_CHECK(dadChrom.size() == momChrom.size())

	const std::vector< double >& dad = dynamic_cast< const std::vector< double >& >(dadChrom);
	const std::vector< double >& mom = dynamic_cast< const std::vector< double >& >(momChrom);

	resize(dad.size());

	for (unsigned i = size(); i--;)
		(*this)[ i ] = Rng::uni(dad[ i ], mom[ i ]);
}

//===========================================================================
//
// generalized intermediate recombination
//
void
ChromosomeT< double >::recombineGeomIntermediate(const Chromosome& dadChrom,
		const Chromosome& momChrom)
{
	SIZE_CHECK(dadChrom.size() == momChrom.size())

	const std::vector< double >& dad = dynamic_cast< const std::vector< double >& >(dadChrom);
	const std::vector< double >& mom = dynamic_cast< const std::vector< double >& >(momChrom);

	resize(dad.size());

	for (unsigned i = size(); i--;)
		(*this)[ i ] = sqrt(dad[ i ] * mom[ i ]);
}

//===========================================================================

void ChromosomeT< double >::recombineIntermediate(Chromosome& mate)
{
	SIZE_CHECK(size() == mate.size())

	std::vector< double >& v = *this;
	std::vector< double >& w = dynamic_cast< std::vector< double >& >(mate);

	for (unsigned i = v.size(); i--;)
		v[ i ] = w[ i ] = (v[ i ] + w[ i ]) / 2.;
}

//===========================================================================
//
// generalized intermediate recombination
//
void ChromosomeT< double >::recombineGenIntermediate(Chromosome& mate)
{
	SIZE_CHECK(size() == mate.size())

	std::vector< double >& v = *this;
	std::vector< double >& w = dynamic_cast< std::vector< double >& >(mate);

	for (unsigned i = v.size(); i--;) {
		double a = v[ i ];
		double b = w[ i ];
		v[ i ] = Rng::uni(a, b);
		w[ i ] = Rng::uni(a, b);
	}
}

//===========================================================================
//
// generalized intermediate recombination
//
void ChromosomeT< double >::recombineGeomIntermediate(Chromosome& mate)
{
	SIZE_CHECK(size() == mate.size())

	std::vector< double >& v = *this;
	std::vector< double >& w = dynamic_cast< std::vector< double >& >(mate);

	for (unsigned i = v.size(); i--;)
		v[ i ] = w[ i ] = sqrt(v[ i ] * w[ i ]);
}

//===========================================================================

void ChromosomeT< double >::mutateRotate(ChromosomeT<double> & sigma)
{
	int    Dimension   = (*this).size();
	double epsi       = 0.0001;
	int    sigmaCheck = 0;
	double tau1       = 1. / sqrt(2. * Dimension);
	double tau2       = 1. / sqrt(2. * sqrt(Dimension * 1.0));
	double beta       = 0.0873;

	mutateRotate(sigma, tau1, tau2, beta, sigmaCheck, epsi);

}

//=============================================================

void ChromosomeT< double >::mutateRotate(ChromosomeT<double> & sigma_sqr,
		double tau1, double tau2,
		double beta, int sigmaCheck,
		double epsi)

{

	int    Dimension   = (*this).size();
	double alpha = 0, *s, *ss, rdno;

	int i, j, k, index = 0, sign_of_sigma;

	s       = new double[ Dimension ];
	ss	  = new double[ Dimension ];

	// here the adaptation parameters are mutated first!
	rdno = Rng::gauss(0, 1);
	for (i = 0; i < Dimension; i++) {
		sigma_sqr[ i ] *= exp(tau1 * rdno + tau2 * Rng::gauss(0, 1));
		// check that sigma is not too small!
		if ((sigmaCheck) && (sigma_sqr[ i ] < epsi*fabs((*this)[ i ])))
			sigma_sqr[ i ] = epsi * fabs((*this)[ i ]);
	}
	for (i = 0; i < Dimension - 1; i++) {
		for (j = i + 1; j < Dimension; j++) {
			index = int(Dimension * (i + 1) - i - 1 + j - 0.5 * i * (i + 1));
			sigma_sqr[ index ] += beta * Rng::gauss(0, 1);
			// check that angles are within limits!
			if (fabs(sigma_sqr[ index ]) > 3.141592654) {
				if (sigma_sqr[ index ] >= 0) sign_of_sigma = 1; else sign_of_sigma = -1;
				sigma_sqr[ index ] = sigma_sqr[ index ] - 2 * 3.141592654
									 * sign_of_sigma;
			}
		}
	}

	// draw rd. numbers according to sigma values!
	for (k = 0; k < Dimension; k++) {
		s[ k ] = Rng::gauss(0, sigma_sqr[ k ]);
		ss[k]  = 0;
	}

	// do the funny rotation bits!
	for (i = 0; i < Dimension - 1; i++) {
		for (j = i + 1; j < Dimension; j++) {
			for (k = 0; k < Dimension; k++) {
				alpha = sigma_sqr[ int(Dimension*(i+1) -i-1+j -0.5*i*(i+1))];
				if (i == k)
					ss[ k ] = s[ k ] * cos(alpha) - s[ j ] * sin(alpha);
				else
					if (j == k)
						ss[ k ] = s[ i ] * sin(alpha) + s[ k ] * cos(alpha);
					else
						ss[k] = s[k];
			}
			for (k = 0; k < Dimension; k++) s[k] = ss[k];
		}
	}

	// change the objective parameters
	for (i = 0; i < Dimension; i++)
		(*this)[ i ] += s[ i ];

	delete[ ] s;
	delete[ ] ss;
}

//=============================================================

void ChromosomeT< double >::initializeRotate(double SigmaMin,
		double SigmaMax)
{
	int i, j;
	int Dimension = int(-0.5 + sqrt(0.25 + 2 * (*this).size()));

	for (i = 0; i < Dimension; i++)
		(*this)[ i ] = Rng::uni(SigmaMin, SigmaMax);;

	for (i = 0; i < Dimension - 1; i++)
		for (j = i + 1; j < Dimension; j++)
			(*this)[ int(Dimension*(i+1) -i-1+j -0.5*i*(i+1))] = 0;

}

//=============================================================

void ChromosomeT< double >::initializeRotate(const ChromosomeT< double >&
		sigma)
{

	initializeRotate(static_cast< const std::vector< double >& >(sigma));

}
//=============================================================

void ChromosomeT< double >::initializeRotate(const std::vector< double >&
		sigma)
{

	int i;

	for (i = 0; i < int(sigma.size()); i++)(* this)[ i ] = sigma[ i ];

}

//=============================================================

void ChromosomeT< double >::showRotate()
{

	int i, j;

	int n   = int((*this).size());
	int dim = int(- 0.5 + sqrt(2.0 * n + 0.25));

	std::cout << "sigma       = ";
	for (i = 0; i < dim; i++)
		std::cout << (*this)[ i ] << "  ";
	std::cout << std::endl;
	std::cout << "alpha       = ";
	for (i = 0; i < dim - 1; i++) {
		for (j = i + 1; j < dim; j++) {
			std::cout << (*this)[ int(dim*(i+1) -i-1+j -0.5*i*(i+1))] << "  ";
		}
		std::cout << std::endl << "             ";
	}
	std::cout << std::endl;

}

//=============================================================


void ChromosomeT< double >::mutateMSR(double xi_prob)
{

        int    Dimension   = (*this).size();
        double xi;

        for (int i = 0; i < Dimension; i++) {
                if (Rng::coinToss(xi_prob)) xi = 1.5;
                else xi = 1 / 1.5;
                (*this)[ i ] *= xi;
        }

}

//=============================================================

// for unconstrained problems
void ChromosomeT< double >::SBX(ChromosomeT< double >& mate,
								double nc, double p)
{
	unsigned i, n = (*this).size();
	double beta, x, u = 0.;

	if (n != mate.size()) {
	  throw SHARKEXCEPTION ("SBX is only defined for chromosomes of equal length");
	}

	for (i = 0; i < n; i++) {
		if (Rng::coinToss(p)) {
			do {
				u = Rng::uni(0, 1);
			}
			while (u == 1.);
			if (u <= .5)
				beta = pow(2 * u, 1. / (nc + 1.));
			else
				beta = pow(1. / (2 - 2 * u), 1. / (nc + 1.));
			x = (*this)[i];
			(*this)[i] = .5 * ((1 + beta) * x + (1 - beta) * mate[i]);
			mate[i]    = .5 * ((1 - beta) * x + (1 + beta) * mate[i]);
		}
	}
}

// for constrained problems
void ChromosomeT< double >::SBX(ChromosomeT< double >& mate ,
								double lower,
								double upper,
								double nc, double p,
								double epsilon)
{
	unsigned i, n = (*this).size();
	double beta, betaQ, alpha, expp, y1 = 0, y2 = 0, u = 0.;

	if (n != mate.size()) {
	  throw SHARKEXCEPTION("SBX is only defined for chromosomes of equal length");
	}

	for (i = 0; i < n; i++) {
		if (Rng::coinToss(p)) {
			if (mate[i] < (*this)[i]) {
				y1 = mate[i];
				y2 = (*this)[i];
			}
			else {
				y1 = (*this)[i];
				y2 = mate[i];
			}
			if (fabs(y2 - y1) > epsilon) { //  -> from Deb's implementation, not contained in any paper: prevents division by zero
				// Find beta value
				if ((y1 -  lower) > (upper - y2)) {
					beta = 1 + (2 * (upper - y2) / (y2 - y1));
				}
				else {
					beta = 1 + (2 * (y1 -  lower) / (y2 - y1));
				}

				expp = (nc + 1.);
				beta = 1. / beta;

				// Find alpha
				alpha = 2. - pow(beta , expp) ;

				if (alpha < 0.0) {
				  throw SHARKEXCEPTION( "Error in ChromosomeT_double::SBX alpha<0");
				}

				expp = 1. / expp;

				u = Rng::uni(0, 1);
				//  -> from Deb's implementation, not contained in any paper
				// do { u = Rng::uni(0, 1); } while(u == 1.);

				if (u <= 1. / alpha) {
					alpha *= u;
					betaQ = pow(alpha, expp);
				}
				else {
					alpha *= u;
					alpha = 1. / (2. - alpha);
					if (alpha < 0.0) {
					  throw SHARKEXCEPTION( "Error in ChromosomeT_double::SBX alpha<0");
					}
					betaQ = pow(alpha, expp);
				}
			}
			else { // if genes are equal -> from Deb's implementation, not contained in any paper
				betaQ = 1.;
			}

			(*this)[i]   = .5 * ((y1 + y2) - betaQ * (y2 - y1));
			mate[i]        = .5 * ((y1 + y2) + betaQ * (y2 - y1));

			//  -> from Deb's implementation, not contained in any paper
			if ((*this)[i] < lower)(*this)[i] = lower;
			if ((*this)[i] > upper)(*this)[i] = upper;
			if (mate[i]      < lower) mate[i]      = lower;
			if (mate[i]      > upper) mate[i]      = upper;
		}
	}
}
// for constrained problems
void ChromosomeT< double >::SBX(ChromosomeT< double >& mate ,
								std::vector<double > & lower,
								std::vector<double > & upper,
								double nc, double p,
								double epsilon)
{
	unsigned i, n = (*this).size();
	double beta, betaQ, alpha, expp, y1 = 0, y2 = 0, u = 0.;

	if (n != mate.size()) {
	  throw SHARKEXCEPTION("SBX is only defined for chromosomes of equal length");
	}

	for (i = 0; i < n; i++) {
		if (Rng::coinToss(p)) {
			if (mate[i] < (*this)[i]) {
				y1 = mate[i];
				y2 = (*this)[i];
			}
			else {
				y1 = (*this)[i];
				y2 = mate[i];
			}
			if (fabs(y2 - y1) > epsilon) { //  -> from Deb's implementation, not contained in any paper: prevents division by zero

				// Find beta value
				if ((y1 - lower[i]) > (upper[i] - y2))
					beta = 1 + (2 * (upper[i] - y2) / (y2 - y1));
				else
					beta = 1 + (2 * (y1 - lower[i]) / (y2 - y1));


				expp = (nc + 1.);
				beta = 1. / beta;

				// Find alpha
				alpha = 2. - pow(beta , expp) ;

				if (alpha < 0.0) {
				  throw SHARKEXCEPTION( "Error in ChromosomeT_double::SBX alpha<0");
				}

				expp = 1. / expp;

				u = Rng::uni(0, 1);

				//  -> from Deb's implementation, not contained in any paper
				// do { u = Rng::uni(0, 1); } while(u == 1.);

				if (u <= 1. / alpha) {
					alpha *= u;
					betaQ = pow(alpha, expp);
				}
				else {
					alpha *= u;
					alpha = 1. / (2. - alpha);
					if (alpha < 0.0) {
					  throw SHARKEXCEPTION( "Error in ChromosomeT_double::SBX alpha<0");
					}
					betaQ = pow(alpha, expp);
				}
			}
			else { // if genes are equal -> from Deb's implementation, not contained in any paper
				betaQ = 1.;
			}

			(*this)[i]   = .5 * ((y1 + y2) - betaQ * (y2 - y1));
			mate[i]        = .5 * ((y1 + y2) + betaQ * (y2 - y1));

			//  -> from Deb's implementation, not contained in any paper
			if ((*this)[i] < lower[i])(*this)[i] = lower[i];
			if ((*this)[i] > upper[i])(*this)[i] = upper[i];
			if (mate[i]      < lower[i]) mate[i]      = lower[i];
			if (mate[i]      > upper[i]) mate[i]      = upper[i];
		}
	}
}

// for unconstrained problems
void ChromosomeT< double >::simpleMutatePolynomial(double lower, double upper,
		double nm, double p)
{
	unsigned i, n = (*this).size();
	double delta,  r = 0.;

	for (i = 0; i < n; i++) {
		if (Rng::coinToss(p)) {
			r = Rng::uni(0, 1);
			if (r < .5)
				delta = pow(2. * r, 1. / (nm + 1.)) - 1;
			else
				delta = 1 - pow(2. - 2. * r, 1. / (nm + 1.));
			(*this)[i] += delta * (upper - lower);
		}
	}
}
// for unconstrained problems
void ChromosomeT< double >::simpleMutatePolynomial(std::vector<double > &lower,
		std::vector<double > &upper,
		double nm, double p)
{
	unsigned i, n = (*this).size();
	double delta,  r = 0.;

	for (i = 0; i < n; i++) {
		if (Rng::coinToss(p)) {
			r = Rng::uni(0, 1);
			if (r < .5)
				delta = pow(2 * r, 1. / (nm + 1.)) - 1;
			else
				delta = 1 - pow(2 - 2 * r, 1. / (nm + 1.));
			(*this)[i] += delta * (upper[i] - lower[i]);
		}
	}
}
// for constrained problems
void ChromosomeT< double >::mutatePolynomial(double lower,
		double upper,
		double nm, double p)
{
	unsigned i, n = (*this).size();
	double delta, deltaQ, expp,  u = 0.;

	for (i = 0; i < n; i++) {
		if (Rng::coinToss(p)) {
			u  = Rng::uni(0, 1);

			if ((*this)[i] <=  lower || (*this)[i] >= upper) { //  -> from Deb's implementation, not contained in any paper
				(*this)[i] = u * (upper -  lower) +  lower;
#ifdef DEBUG
				std::cerr << "Warning: parameter out of bounds, random resetting ..." << std::endl;
#endif
			}
			else {
				// Calculate delta
				if (((*this)[i] -  lower) < (upper - (*this)[i]))
					delta = ((*this)[i] -  lower) / (upper -  lower);
				else
					delta = (upper - (*this)[i]) / (upper -  lower);

				delta = 1. - delta;
				expp  = (nm + 1.);
				delta = pow(delta , expp);
				expp  = 1. / expp;

				if (u <= .5) {
					deltaQ =  2. * u + (1 - 2. * u) * delta;
					deltaQ = pow(deltaQ, expp) - 1. ;
				}
				else {
					deltaQ = 2. - 2. * u + 2. * (u  - .5) * delta;
					deltaQ = 1. - pow(deltaQ , expp);
				}

				(*this)[i] += deltaQ * (upper -  lower);

				//  -> from Deb's implementation, not contained in any paper
				if ((*this)[i] < lower)(*this)[i] = lower;
				if ((*this)[i] > upper)(*this)[i] = upper;
			}
		}
	}
}
// for constrained problems
void ChromosomeT< double >::mutatePolynomial(std::vector<double > &lower,
		std::vector<double > &upper,
		double nm, double p)
{
	unsigned i, n = (*this).size();
	double delta, deltaQ, expp,  u = 0.;


	for (i = 0; i < n; i++) {

		if (Rng::coinToss(p)) {
			u  = Rng::uni(0, 1);
			if ((*this)[i] <= lower[i] || (*this)[i] >= upper[i]) { //  -> from Deb's implementation, not contained in any paper
				(*this)[i] = u * (upper[i] - lower[i]) + lower[i];
#ifdef DEBUG
				std::cerr << "Warning: parameter out of bounds, random resetting ..." << std::endl;
#endif
			}
			else {
				// Calculate delta
				if (((*this)[i] - lower[i]) < (upper[i] - (*this)[i]))
					delta = ((*this)[i] - lower[i]) / (upper[i] - lower[i]);
				else
					delta = (upper[i] - (*this)[i]) / (upper[i] - lower[i]);

				delta = 1. - delta;
				expp  = (nm + 1.);
				delta = pow(delta , expp);
				expp  = 1. / expp;

				if (u <= .5) {
					deltaQ =  2. * u + (1 - 2. * u) * delta;
					deltaQ = pow(deltaQ, expp) - 1. ;
				}
				else {
					deltaQ = 2. - 2. * u + 2. * (u  - .5) * delta;
					deltaQ = 1. - pow(deltaQ , expp);
				}

				(*this)[i] += deltaQ * (upper[i] - lower[i]);

				//  -> from Deb's implementation, not contained in any paper
				if ((*this)[i] < lower[i])
					(*this)[i] = lower[i];

				if ((*this)[i] > upper[i])
					(*this)[i] = upper[i];

			}
		}
	}
}


