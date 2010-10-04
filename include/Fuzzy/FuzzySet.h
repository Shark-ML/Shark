/**
 * \file FuzzySet.h
 *
 * \brief Abstract super class for specific fuzzy sets 
 *
 * \author Marc Nunkesser
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


#ifndef FUZZYSET_H
#define FUZZYSET_H

#include <string>
#include "RCObject.h"
#include "FuzzySet.h"
#ifdef __SOLARIS__
#include <climits>
#endif
#ifdef __LINUX__
#include <float.h>
#endif

#include <limits>

#define ERR_RELATIVE 0.0001
#define RECURSION_MIN 5
#define RECURSION_MAX 14

class NDimFS;

/**
 * \brief Abstract super class for specific fuzzy sets.
 * 
 * 
 */
class FuzzySet: virtual public RCObject {
public:
	
    /**
     * \brief Constructor.
     * 
     * @param scale scaling factor for the membership function
     */
    FuzzySet(double scale = 1.0); // Constructor
    
    /// Destructor
	virtual ~FuzzySet();
    
	// virtual                 FuzzySet(const FuzzySet&) // copy constructor

    /**
     * \brief The membership function.
     * 
     * If <i>fs</i> is a FuzzySet, <i>fs( x )</i> returns the value of the 
     * membership function (the \f$\mu\f$-function) at point <i>x</i>.
     */
	inline double operator()( double x ) const {
		return scaleFactor*mu(x);
	}; //=0

    /**
     * \brief Rescales the membership function.
     * 
     * <b>Note</b>: This doesn't set a new scaling factor but scales the
     * current one by the given factor.
     * 
     * @param factor factor for rescaling
     */
	inline void scale( double factor ) {
		scaleFactor *= factor;
	};
    
    /**
     * \brief Defuzzification by centroid method.
     * 
     * This method defuzzifies the fuzzy set by determining the value of the
     * abscissa of the centre of gravity of the area below the membership
     * function:
     * 
     * \f[
     *      x_0 = \frac{ \int x\mu(x)\,dx }{ \int \mu(x)\,dx }
     * \f]
     * 
     * @param lowerBound lower bound for defuzzification interval
     * @param upperBound upper bound for defuzzification interval
     * @param errRel relative approximation error that is tollerated during numerical integration
     * @param recursionMax max. depth of recursion during numerical integration (i.e. max. \f$2^n\f$ steps)
     */
	virtual double defuzzify( double lowerBound = -std::numeric_limits<double>::max(),
	                          double upperBound =  std::numeric_limits<double>::max(),
	                          double errRel = ERR_RELATIVE, 
	                          int recursionMax = RECURSION_MAX ) const;
                                      
    /**
     * \brief Defuzzification by Smallest-of-Maximum (SOM) method.
     * 
     * This method defuzzifies the fuzzy set by searching for the point (in the
     * given interval) where the membership functions reaches its maximum. If 
     * this maximum isn't unique, the first one will be taken.
     * 
     * @param steps number of equally distributed points the membership function
     * will be evaluated at 
     * @param low lower bound for defuzzification interval
     * @param high upper bound for defuzzification interval
     */
	virtual double defuzzifyMax( unsigned int steps = 100,
	                             double low  =-std::numeric_limits<double>::max(),
	                             double high = std::numeric_limits<double>::max() ) const;
    
    /**
     * \brief Writes gnuplot suited data into a file.
     * 
     * This function writes the membership function into a gnuplot suited data 
     * file. Only the range where the membership function is nonzero or above a 
     * given threshold is considered. If the support of the membership function 
     * is not limited, we assume the limits of the data type <i>double</i>.
     * 
     * @param fileName name of the outputfile (existing files will be
     * overwritten)
     * @param steps number of sampling points
     */
	virtual void makeGNUPlotData( const std::string fileName = "mf.dat",
	                              unsigned int steps = 100 ) const;

    /**
     * \brief Writes gnuplot suited data into a file.
     * 
     * This function writes the membership function into a gnuplot suited data 
     * file. Only the specified range is considered.
     * 
     * @param fileName name of the outputfile (existing files will be
     * overwritten)
     * @param steps number of sampling points
     * @param lowerBound lower bound of plotting interval
     * @param upperBound upper bound of plotting interval
     */
	virtual void makeGNUPlotData( const std::string fileName,
	                              unsigned int steps,
	                              double lowerBound,
	                              double upperBound ) const;

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	virtual double getMin() const = 0;

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	virtual double getMax() const = 0;

	// set the parameters of the fuzzy set. To be overloaded in
	// derived classes.
	// virtual void            setParams();
	// commented out to avoid some warnings
    
private:
	// methods:
	// the mu-function, returns the value of the membership-function at x
	virtual double mu(double x) const = 0;
	// attributes:
	// the scale factor cf. operator()
	double scaleFactor;
    
    double xmu( double x ) const {
        return x * mu( x );
    }
    
    double integrate( double (FuzzySet::*f)(double) const, double lowerBound, double upperBound, double errRel = ERR_RELATIVE, int recursionMax = RECURSION_MAX ) const;
    
    /**
     * \brief Integrates the given memberfunction numerically with the Adaptive Trapezoidal Method
     * 
     * This method uses the trapezium rule to approximate the integral over an 
     * interval. If an estimate of the error exceeds an user defined toleranz, the 
     * algorithm calls for subdividing the interval in two and applying trapezoidal 
     * rule to each subinterval in a recursive manner until the estimated error mets
     * the given tolerance or the max. depth of recursion is reached.<br>
     * <br>
     * Trapezium rule works by approximating the region under the graph of the 
     * function f by a trapezium and calculating its area.
     * 
     * \f[
     *      \int_{a}^{b} f(x)\, dx \approx T(a,b) = (b-a)\frac{f(a) + f(b)}{2}
     * \f]
     * 
     * The error estimation used here is given by
     * \f$\varepsilon = | T(a,m) + T(m,b) - T(a,b) |\f$, s.t.
     * \f$m = \frac{1}{2}(a+b)\f$
     * 
     * This estimation is repeated for the subintervals until the error times
     * the integral mets the given tolerance:
     * 
     * \f[
     *      \varepsilon | T(a,m) + T(m,b) | < \tau
     * \f]
     * 
     * Then the subintervals final approximation is \f$T(a,m) + T(m,a)\f$
     * accordingly.
     */
    double adaptive_trapezoid( double (FuzzySet::*f)(double) const, double a, double b, double errRel, double recLevel, int recursionMax, double fa, double fm, double fb ) const;
    
    /**
     * \brief Integrates the given memberfunction numerically with the Adaptive Simpson's Method
     *
     * Analogical to the adaptive trapezoidal method this mehthod recursively 
     * applies the Simpon's rule to the subintervals until the designated 
     * tolerance is met or the max. level of recursion is reached.<br>
     * <br>
     * The Simpson's rule approximates the integral by the quadratic
     * polynomial that takes the same values as the integrand f in the endpoints 
     * a and b and in the midpoint m:
     * 
     * \f[
     *      \int_{a}^{b} f(x)\, dx \approx S(a,b) = \frac{b-a}{6}\left[f(a)+4f\left(\frac{a+b}{2}\right)+f(b)\right]
     * \f]
     * 
     * For error estimation the term above is devided by the factor 15 which 
     * produces a more accurate approximation in the case of Simpson's method.
     * 
     * \f[
     *  \varepsilon = \frac{1}{15} | T(a,m) + T(m,b) - T(a,b) |
     * \f]
     * 
     * For details see: Numerical Methods Using Matlab, 4th Edition, 2004, John H.
     * Mathews and Kurtis K. Fink;
     * http://math.fullerton.edu/mathews/n2003/adaptivequad/AdaptiveQuadProof.pdf
     */
    double adaptive_simpsons(  double (FuzzySet::*f)(double) const, double a, double b, double errRel, double recLevel, int recursionMax, double fa, double fm, double fb ) const;
    
};

#endif
