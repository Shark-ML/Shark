/**
 * \file TriangularFS.h
 *
 * \brief FuzzySet with triangular membership function
 * 
 * \authors Marc Nunkesser, Copyright (c) 2008, Marc Nunkesser
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */

/* $log$ */


#ifndef TRIANGULARFS_H
#define TRIANGULARFS_H

#include <Fuzzy/FuzzySet.h>
#include <Fuzzy/FuzzyException.h>


/**
 * \brief FuzzySet with triangular membership function.
 * 
 * This class implements a FuzzySet with a triangular membership function.
 * A triangular mambership function has a triangular shape with a
 * maximum value of 1 at a certain point b. 
 *
 * \f[
 * 		\mu(x) = \left\{\begin{array}{ll} 0 & x < a \\ 
 *      \frac{1}{b-a}(x-a) & a \le x < b \\
 * 		\frac{1}{c-b}(c-x) & b \le x < c \\
 *      0 & x \geq c\end{array}\right.
 * \f]
 *  
 * <img src="../images/TriangularFS.png">
 * 
 */
class TriangularFS: virtual public FuzzySet {
public:
	

	/**
	 * \brief Constructor.
	 *
	 * @param a the minimal value for which the membership function is nonzero
	 * @param b the value for which the membership function has value 1
	 * @param c the maximal value for which the membership function is nonzero
	 */
	TriangularFS(double a,double b,double c);
	
     /**
	 * \brief Destructor.	
 	 */	
	virtual                ~TriangularFS();

     /**
      * \brief Returns the lower boundary of the support
      * 
      * @return the min. value for which the membership function is nonzero (or exceeds a
      * given threshold)
      */
	virtual double  getMin() const {
		return(a);
	};

     /**
      * \brief Returns the upper boundary of the support
      * 
      * @return the max. value for which the membership function is nonzero (or exceeds a
      * given threshold)
      */
	virtual double  getMax() const {
		return(c);
	};

	/**
	 * \brief Sets the parameter of the fuzzy set.
	 *
	 * @param a the minimal value for which the membership function is nonzero
	 * @param b the value for which the membership function has value 1
	 * @param c the maximal value for which the membership function is nonzero
	 */
	virtual void    setParams(double a,double b,double c);

private:
	// overloaded operator () - the mu-function
	// The triangular mf (normalized i.e the max is always 1) is represented
	// by three points: the first nonzero value a, value b so that mu(b)=1
	// and the last nonzero value c.

	double                 mu(double x) const;
	double                 a,b,c;// parameters of a triangular MF

};


#endif
