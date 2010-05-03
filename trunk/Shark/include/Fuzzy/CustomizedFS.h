/**
 * \file CustomizedFS.h
 *
 * \brief A FuzzySet with an user defined mambership function
 * 
 * \authors Marc Nunkesser
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
