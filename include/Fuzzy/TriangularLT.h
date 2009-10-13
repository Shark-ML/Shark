
/**
 * \file TriangularLT.h
 *
 * \brief LinguisticTerm with triangular membership function
 * 
 * \authors Marc Nunkesser
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
