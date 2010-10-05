
/**
 * \file TriangularLT.h
 *
 * \brief LinguisticTerm with triangular membership function
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
#ifndef TRIANGULARLT_H
#define TRIANGULARLT_H

// #include <ZNminmax.h>

#include <Fuzzy/LinguisticTerm.h>
#include <Fuzzy/LinguisticVariable.h>
#include <Fuzzy/TriangularFS.h>

/**
 * \brief LinguisticTerm with triangular membership function.
 * 
 * This class implements a LinguisticTerm with a triangular membership function.
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
class TriangularLT: public LinguisticTerm, public TriangularFS {
public:
	
	/**
	 * \brief Constructor.
	 *
	 * @param name the name 
	 * @param parent the associated linguistic variable
	 * @param a the minimal value for which the membership function is nonzero
	 * @param b the value for which the membership function has value 1
	 * @param c the maximal value for which the membership function is nonzero
	 */
	TriangularLT(const std::string             name,
	             const RCPtr<LinguisticVariable>& parent,
	             double                           a,
	             double                           b,
	             double                           c);


	// overloaded operator () - the mu function
	// inline double         operator()(double x) const;



   /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double         getMin() const {
		return(std::max(TriangularFS::getMin(), parent->getLowerBound()));
	};

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */	
	inline double         getMax() const {
		return(std::min(TriangularFS::getMax(), parent->getUpperBound()));
	};

};




///////////////////////////////////////////////
/////// inline functions
///////////////////////////////////////////////





//double TriangularLT::operator()(double x) const {return(TriangularFS::operator()(x));}


#endif
