/**
 * \file SingletonFS.h
 *
 * \brief FuzzySet with a single point of positive membership
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */

#ifndef SINGLETONFS_H
#define SINGLETONFS_H

#include <Fuzzy/FuzzySet.h>
#include <cmath>

/**
 * \brief FuzzySet with a single point of positive membership.
 * 
 * The membership function of a SingletonFS takes value 1 only at a given point
 * and value 0 everywhere else.
 */
class SingletonFS: virtual public FuzzySet {
public:

    /**
	* \brief Constructor.
	* 
	* @param x position of membership point
	* @param yValue value of μ(x)
    * @param eps range of membership around x
	*/

	inline             SingletonFS(double x=0.0,
	                               double yValue = 1.0,
	                               double eps = 1E-5
	                              ):c(x), yValue(yValue),epsilon(eps) {};
	
    /**
     * \brief Defuzzification
     *
     */
	inline double      defuzzify() const{
		return c;
	};

   /**
    * \brief Returns the lower boundary of the support
	* 
	* @return the min. value for which the membership function is nonzero (or 
    * exceeds a given threshold)
	*/
	inline double      getMin() const {
		return c;
	};

    /**
 	* \brief Returns the upper boundary of the support
 	* 
	* @return the max. value for which the membership function is nonzero (or exceeds a
	* given threshold)
	*/
	inline double      getMax() const {
		return c;
	};

    /**
	* \brief Sets the parameters of the fuzzy set.
	* 
    * @param x new position of membership point
    * @param yValue new value of μ(x)
    * @param eps new range of membership around x
	*/
	inline void        setParams(double x=0.0,
	                             double yValue = 1.0,
	                             double eps = 1E-5);

private:
	inline double      mu( double x) const {
		return(fabs(x-c)<epsilon?yValue:0.0);
	};
	//the position at which the singleton is defined (mu(c)=value);
	double             c;
	//the precision to which the comparison is carried out

	double             yValue;
	double             epsilon;
};


#endif
