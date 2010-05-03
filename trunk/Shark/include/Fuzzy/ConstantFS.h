
/**
 * \file ConstantFS.h
 *
 * \brief FuzzySet with constant membership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */

#ifndef CONSTANTFS_H
#define CONSTANTFS_H

#include <Fuzzy/FuzzySet.h>
#ifdef __SOLARIS__
#include <climits>
#endif
#ifdef __LINUX__
#include <float.h>
#endif


/**
 * \brief FuzzySet with constant membership function.
 * 
 * This class implements a FuzzySet with constant membership function.
 * 
 * <img src="../images/ConstantFS.png"> 
 * 
 */
class ConstantFS: virtual public FuzzySet {
public:

    /**
	* \brief Constructor.
	*
	* @param x the constant value of the membership function
	*/
	inline             ConstantFS(double x):c(x) {}; //c is the constant so that the memberfunction equals c.

    /**
	* \brief Defuzzification of the fuzzy set.  
	*/
	inline double defuzzify() const{
		return(0);
	};

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double      getMin() const {
		return(-std::numeric_limits<double>::max());
	} ;

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double      getMax() const {
		return(std::numeric_limits<double>::max());
	};

    /**
	* \brief Sets the parameter of the fuzzy set.
	*
	* @param x the constant value of the membership function
	*/
	inline void        setParams(double x){
		c=x;
	};
	
private:
	inline double      mu(double x) const{
		return(c);
	};
	double             c;
};


#endif
