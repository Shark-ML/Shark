/**
 * \file CustomizedFS.h
 *
 * \brief A FuzzySet with an user defined mambership function
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

#ifndef CustomizedFS_H
#define CustomizedFS_H

#include <Fuzzy/FuzzySet.h>

/**
 * \brief A FuzzySet with an user defined mambership function.
 *
 * This class implements a FuzzySet with an user definded membership 
 * function.
 */
class CustomizedFS : virtual public FuzzySet {
public:

    /**
     * \brief Constructor.
     *
     * @param userFunction membership function defined by the user
     * @param min the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)	
     * @param max the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline             CustomizedFS(double (*userFunction)(double),
	                                double min,
	                                double max): userDefinedMF(userFunction),minimum(min),maximum(max) {};
	//inline             ~CustomizedFS();

    /**
     * \brief Sets the membership function of the fuzzy set.
     * 
     * @param userFunction membership function defined by the user
     */
	inline void        setMF( double (*userFunction)(double) ){
		userDefinedMF = userFunction;
	};

    /**
     * \brief Returns the lower boundary of the support.
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double      getMin() const{
	    return(minimum);
	};
 
   /**
    * \brief Returns the upper boundary of the support.
    * 
    * @return the max. value for which the membership function is nonzero (or exceeds a
    * given threshold)
    */
    inline double    	getMax() const{
    	return(maximum);
    };

private:
	inline double mu(double x) const{
		return (*userDefinedMF)(x);
	};
	
	double (*userDefinedMF)(double);
	double  minimum,maximum;



};


#endif
