//===========================================================================
/*!
 *  \file RNG.h
 *
 *  \brief This file contains a class, that defines a generator
 *         for uniformally distributed pseudo random numbers of
 *         the interval (0,1).
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Copyright (c) 1995,1999:
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
 *      Rng
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Rng. This library is free software;
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

#ifndef __RNG_H
#define __RNG_H

#include <cstdlib>
#include <sys/types.h>
#include <ctime>

//===========================================================================
/*!
 *  \brief This class defines a generator
 *         for uniformally distributed pseudo random numbers of
 *         the interval (0,1).

 *  A computer can only create "pseudo-random numbers", i.e. a sequence
 *  of numbers generated in some systematic way such that its statistical
 *  properties are as close as possible to those of true random numbers,
 *  e.g. negligible correlation between consecutive numbers. The most common
 *  methods are based on the "multiplicative congruential algorithm". <br>
 *  This basic algorithm is defined as:
 *
 *  \f$
 *      n_i \equiv ( a \cdot n_{i-1} ) \pmod m
 *  \f$
 *
 *  The resulting integers \f$n_i\f$ are then divided by \f$m\f$ to give
 *  uniformly distributed pseudo-random numbers lying in the interval
 *  (0,1). <br>
 *  The start number \f$n_0\f$ is also called the "seed". <br>
 *  As you can see the random numbers created are cyclic with a maximum
 *  frequency of \f$m\f$. When choosing unintelligent numbers for
 *  \f$a\f$ and \f$m\f$, the frequency will be further shortened. <br>
 *  The aim is to achieve a most possible large frequency. <br>
 *  Therefore, this class uses an alternative algorithm, the
 *  Wichman-Hill algorithm, that uses three generators: <br>
 *
 *  \f$
 *      \mbox{\ }\\ \noindent
 *      n_{1,i} \equiv ( a_1 \cdot n_{1, i-1} ) \pmod {m_1}\\
 *      n_{2,i} \equiv ( a_2 \cdot n_{2, i-1} ) \pmod {m_2}\\
 *      n_{3,i} \equiv ( a_3 \cdot n_{3, i-1} ) \pmod {m_3}
 *  \f$
 *
 *  From this the pseudo random numbers \f$U_i\f$ are generated
 *  as: <br>
 *
 *  \f$
 *      U_i \equiv (\frac{n_{1,i}}{m_1} + \frac{n_{2,i}}{m_2} +
 *             \frac{n_{3,i}}{m_3}) \pmod {1.0}
 *  \f$
 *
 *  The random numbers \f$U_i\f$ are uniformally distributed in the
 *  interval (0,1). <br>
 *  By using intelligent numbers for \f$a_i\f$ and \f$m_i\f$ for
 *  \f$i = 1, 2, 3\f$ a frequency of \f$6953607871644\f$ is achieved. <br>
 *  For more information about this algorithm please refer to the
 *  BYTE magazine, March, 1987, pp. 127-128. <br>
 *  There will be one static instance of this class named
 *  #globalRng. This instance can be used as base for the template
 *  class RandomVar from which all random generators used in
 *  library "#Rng" are derived. <br>
 *  This is done, because for every distribution simulation, where
 *  the distribution is not uniform, the uniformally distributed pseudo
 *  random numbers delivered by this class are transformed to the
 *  types of the other distributions. <br>
 *  So, if you have the uniformally distributed random numbers
 *  \f$z_i\f$ you can transform them into random numbers \f$x_i\f$
 *  that are distributed by the function \f$F(X)\f$ by applying
 *  the inverse function of \f$F\f$ to \f$z_i\f$:
 *
 *  \f$
 *      x_i = F^{-1}(z_i)
 *  \f$
 *
 *  This is also called the \em inverse \em transformation.
 *
 *  \author  M. Kreutz
 *  \date    1998-08-17
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 */
class RNG
{
public:

	virtual ~RNG()
	{};

	//========================================================================
	/*!
	 *  \brief Creates a new pseudo random number instance and generates seed
	 *         values for the three internal number generators.
	 *
	 *  \em s is used as base for the generation of the seed numbers
	 *  \f$n_{0,1},\mbox{\ }n_{0,2},\mbox{\ }n_{0,3}\f$ (see description
	 *  of the class) for the three number generators.
	 *
	 *  \param s the base for the generation of the three seed values,
	 *           by default the value "1" is taken.
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1998-08-17
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	RNG(long s = 1)
	{
		seed(s);
	}


	//========================================================================
	/*!
	 *  \brief Uses "s" to generate the seed values for the three
	 *         internal random number generators.
	 *
	 *  \em s is used as base for the generation of the seed numbers
	 *  \f$n_{0,1},\mbox{\ }n_{0,2},\mbox{\ }n_{0,3}\f$ (see description
	 *  of the class) for the three number generators.
	 *
	 *  \param s the base for the generation of the three seed values,
	 *           by default the current system clock time is taken 
	 *           (i.e. the number of seconds since 00:00 hours, Jan 1, 
	 *           1970 UTC).
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1998-08-17
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void       seed(long s = time(NULL))
	{
		//if( s == 0 ) s = time( NULL );
		initialSx = sx = (s         & 0xff) + 1;
		initialSy = sy = ((s >>  8) & 0xff) + 10000;
		initialSz = sz = ((s >> 16) & 0xffff) + 3000;
	}


	//========================================================================
	/*!
	 *  \brief Resets the current results of the three internal random number
	 *         generators to the seed values.
	 *
	 *  The current results
	 *  \f$n_{i,1},\mbox{\ }n_{i,2},\mbox{\ }n_{i,3}\f$ (see description
	 *  of the class) for the three number generators are set to the
	 *  stored seed values \f$n_{0,1},\mbox{\ }n_{0,2},\mbox{\ }n_{0,3}\f$.
	 *
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1998-08-17
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void       reset()
	{
		sx = initialSx;
		sy = initialSy;
		sz = initialSz;
	}


	//========================================================================
	/*!
	 *  \brief Returns a new discrete uniformally distributed pseudo 
	 *         random number from the interval (0,1).
	 *
	 *  Generates and returns the current random number \f$U_i\f$
	 *  (see description of class) in the interval (0,1). 
	 *  The type of the random number will be "long", so the original
	 *  continuous type of the random number will be transformed
	 *  to a discrete type.
	 *
	 *  \return the current random number \f$U_i\f$ as long value
	 *
	 *  \author  M. Kreutz
	 *  \date    1998-08-17
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	long       genLong()
	{
		const long MaxRand = 0x7fffffffL;

		return (long)(genDouble() *((double)MaxRand + 1));
	}


	//========================================================================
	/*!
	 *  \brief Returns a new continuous uniformally distributed pseudo 
	 *         random number from the interval (0,1).
	 *
	 *  Generates and returns the current random number \f$U_i\f$
	 *  (see description of class) in the interval (0,1). 
	 *  The type of the continuous random number will be "double".
	 *
	 *  \return the current random number \f$U_i\f$ as double value
	 *
	 *  \author  M. Kreutz
	 *  \date    1998-08-17
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	double     genDouble()
	{
		double rn;

		// the three congruential generators
		sx = (unsigned)(sx * 171UL % 30269UL);
		sy = (unsigned)(sy * 172UL % 30307UL);
		sz = (unsigned)(sz * 170UL % 30323UL);

		rn = sx / 30269. + sy / 30307. + sz / 30323.;

		return rn - (unsigned)rn;
	}


	//========================================================================
	/*!
	 *  \brief Returns the current results of the three internal random 
	 *         number generators.
	 *
	 *  Stores the current values of 
	 *  \f$n_{i,1},\mbox{\ }n_{i,2},\mbox{\ }n_{i,3}\f$ (see description
	 *  of the class) in \em x, \em y and \em z.
	 *
	 *  \param x the current result \f$n_{i,1}\f$ of generator no. 1
	 *  \param y the current result \f$n_{i,2}\f$ of generator no. 2
	 *  \param z the current result \f$n_{i,3}\f$ of generator no. 3
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1998-08-17
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void       getStatus(unsigned &x, unsigned &y, unsigned &z) const
	{
		x = sx;
		y = sy;
		z = sz;
	}


	//========================================================================
	/*!
	 *  \brief Sets the current results of the three internal random 
	 *         number generators.
	 *
	 *  Sets the current values of 
	 *  \f$n_{i,1},\mbox{\ }n_{i,2},\mbox{\ }n_{i,3}\f$ (see description
	 *  of the class), that will be used as base for the generation
	 *  of the next random number.
	 *
	 *  \param x the new current result \f$n_{i,1}\f$ of generator no. 1
	 *  \param y the new current result \f$n_{i,2}\f$ of generator no. 2
	 *  \param z the new current result \f$n_{i,3}\f$ of generator no. 3
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1998-08-17
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void       setStatus(const unsigned x, const unsigned y, const unsigned z)
	{
		// initial state is not restored
		sx = x;
		sy = y;
		sz = z;
	}

	//========================================================================
	/*!
	 *  \brief Returns a new continuous uniformally distributed pseudo random 
	 *         number of the interval (0,1).
	 *
	 *  Generates and returns the current random number \f$U_i\f$
	 *  (see description of the class) as continuous value of
	 *  type "double".
	 *
	 *  \return the current random number \f$U_i\f$ as double value
	 *
	 *  \author  M. Kreutz
	 *  \date    1998-08-17
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	virtual double operator()()
	{
		return genDouble();
	}


	//! Global instantiation of class RNG, can be used by all random number
	//! generators in library "#Rng".
	static RNG globalRng;


private:

	// Seed value for internal generator no. 1.
	unsigned initialSx;

	// Seed value for internal generator no. 2.
	unsigned initialSy;

	// Seed value for internal generator no. 3.
	unsigned initialSz;

	// Current value of ni,1 (see description of class).
	unsigned sx,

	// Current value of ni,2 (see description of class).
	sy,

	// Current value of ni,3 (see description of class).
	sz;
};

#endif  /* !__RNG_H */






