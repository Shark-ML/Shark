
/**
 * \file GeneralizedBellFS.h
 *
 * \brief FuzzySet with a generalized bell-shaped membership function
 * 
 * \authors Thomas Voß
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


#ifndef __GENERALIZEDBELLFS_H__
#define __GENERALIZEDBELLFS_H__

#include <Fuzzy/FuzzySet.h>

// Calculate according to 1/(1+( (x-center) / (width) )^2*slope)
/**
 * \brief FuzzySet with a generalized bell-shaped membership function.
 * 
 * This class implements a FuzzySet with membership function:
 * 
 * \f[
 *      \mu(x) = \frac{1}{1+ (\frac{x-center}{width})^{2*slope}}
 * \f]
 * <img src="../images/GeneralizedBellFS.png"> 
 */
class GeneralizedBellFS : virtual public FuzzySet {
public:

/**
 * \brief Constructor.
 *
 * @param slope the slope
 * @param center the center of the bell-shaped function
 * @param width the width of the bell-shaped function
 * @param scale the scale of the bell-shaped function
 */
	GeneralizedBellFS( double slope = 1, double center = 0, double width = 1, double scale = 1 );
	
/**
 * \brief Returns the lower boundary of the support.
 * 
 * @return the min. value for which the membership function is nonzero (or exceeds a
 * given threshold)
 */
	double getMin() const;

/**
 * \brief Returns the upper boundary of the support.
 * 
 * @return the max. value for which the membership function is nonzero (or exceeds a
 * given threshold)
 */
	double getMax() const;
	
protected:

private:
	double mu( double x ) const;
	double m_slope;
	double m_center;
	double m_width;
};

#endif // __GENERALIZEDBELLFS_H__
