
/**
 * \file TrapezoidLT.h
 *
 * \brief LinguisticTerm with trapezoid membership function
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

#ifndef TRAPEZOIDLT_H
#define TRAPEZOIDLT_H


#include <Fuzzy/LinguisticTerm.h>
#include <Fuzzy/LinguisticVariable.h>
#include <Fuzzy/TrapezoidFS.h>

/**
 * \brief LinguisticTerm with trapezoid membership function.
 * 
 * This class implements a LinguisticTerm with a trapezoid membership function.
 * A trapezoid membership function is definded by four values a,b,c,d. 
 * Points smaller than a and bigger than d have the value 0. The function
 * increases constantly to the value 1 between a and b, stays 1 between b 
 * and c, and decreases constantly between c and d.
 * 
 * \f[
 * 		\mu(x) = \left\{\begin{array}{ll} 0 & x < a \\ 
 *      \frac{1}{b-a}(x-a) & a \le x < b \\
 *      1 & b \leq x < c \\
 * 		\frac{1}{d-c}(d-x) & c \le x < d \\
 *      0 & x \geq d\end{array}\right.
 * \f]
 * 
 * <img src="../images/TrapezoidFS.png">
 * 
 */
class TrapezoidLT: public LinguisticTerm, public TrapezoidFS {
public:
	
    /**
	* \brief Constructor.
	*
	* @param name the name
	* @param parent the associated linguistic variable
	* @param a the minimal value for which the membership function is nonzero
	* @param b the value to which the membership function increases to value 1 
	* @param c the value to which the membership function stays 1
	* @param d the maximal value for which the membership function is nonzero
 	*/
	TrapezoidLT(const std::string             name,
	            const RCPtr<LinguisticVariable>& parent,
	            double                           a,
	            double                           b,
	            double                           c,
	            double                           d);

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double         getMin() const {
		return(std::max(TrapezoidFS::getMin(), parent->getLowerBound()));
	};

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double         getMax() const {
		return(std::min(TrapezoidFS::getMax(), parent->getUpperBound()));
	};

};


#endif
