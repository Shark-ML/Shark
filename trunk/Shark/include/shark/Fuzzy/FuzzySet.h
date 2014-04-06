/**
*
* \brief Abstract super class for specific fuzzy sets 
*
* \author Marc Nunkesser
*/

/* $log$ */
#ifndef FUZZYSET_H
#define FUZZYSET_H

#include <shark/Core/Exception.h>

#include <boost/optional.hpp>

#include <limits>
#include <string>

#include <cmath>

namespace shark {

/**
 * \brief Abstract super class for specific fuzzy sets.
 *
 *
 */
class FuzzySet {
public:

    static double RELATIVE_ERROR() {
        return( 1E-4 );
    }

    static std::size_t RECURSION_MIN() {
        return( 5 );
    }

    static std::size_t RECURSION_MAX() {
        return( 14 );
    }

    /**
    * \brief Constructor
    *
    * @param scale Scaling factor for the membership function
    */
    FuzzySet(double scale = 1.0) : m_scaleFactor( scale ) {}

    /**
    * \brief Virtual d'tor.
    */
    virtual ~FuzzySet() {}

    /**
    * \brief The membership function
    *
    * If <i>fs</i> is a FuzzySet, <i>fs( x )</i> returns the value of the
    * membership function (the \f$\mu\f$-function) at point <i>x</i>.
    */
    inline double operator()( double x ) const {
        return( m_scaleFactor*mu( x ) );
    };

    /**
    * \brief Rescales the membership function
    *
    * <b>Note</b>: This doesn't set a new scaling factor but scales the
    * current one by the given factor.
    *
    * @param factor factor for rescaling
    */
    inline void scale( double factor ) {
        m_scaleFactor *= factor;
    };

    /**
    * \brief Defuzzification by centroid method
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
                              double errRel = FuzzySet::RELATIVE_ERROR(),
                              int recursionMax = FuzzySet::RECURSION_MAX() ) const {
        lowerBound = std::max( min(), lowerBound );
        upperBound = std::min( max(), upperBound );

        if( upperBound - lowerBound < 1E-10 )
            return( lowerBound );

        double numerator = integrate( &FuzzySet::xmu, lowerBound, upperBound, 0.00001 );
        double denominator = integrate( &FuzzySet::mu,  lowerBound, upperBound, 0.00001 );

        return( denominator < 1E-20 ? 0 : numerator / denominator );
    }

    /**
    * \brief Defuzzification by Smallest-of-Maximum (SOM) method
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
                                 double high = std::numeric_limits<double>::max() ) const {
        double lowerBound = std::max( min(), low );
        double upperBound = std::min( max(), high );

        if ( upperBound - lowerBound < 1e-20 )
            return lowerBound;

        double increment = ( upperBound - lowerBound ) / steps;

        //make sure that borders will be taken into consideration
        lowerBound = std::max( lowerBound - increment,-std::numeric_limits<double>::max() );
        upperBound = std::min( upperBound + increment, std::numeric_limits<double>::max() );

        double out = 0;    // point where mf is max
        double maxVal = 0; // current max mf value

        double temp;
        for ( double d = lowerBound; d <= upperBound; d += increment ) {
            temp = (*this)(d);
            out = ( temp > maxVal ? d : out );
            maxVal = std::max( maxVal, temp );
        };

        return out;
    }

    /**
    * \brief Returns the lower boundary of the support
    *
    * @return the min. value for which the membership function is nonzero (or exceeds a
    * given threshold)
    */
    virtual double min() const = 0;

    /**
    * \brief Returns the upper boundary of the support
    *
    * @return the max. value for which the membership function is nonzero (or exceeds a
    * given threshold)
    */
    virtual double max() const = 0;

protected:
    // methods:
    // the mu-function, returns the value of the membership-function at x
    virtual double mu(double x) const = 0;
    // attributes:
    // the scale factor cf. operator()
    double m_scaleFactor;

    double xmu( double x ) const {
        return x * mu( x );
    }

    double integrate( double (FuzzySet::*f)(double) const,
                      double lowerBound,
                      double upperBound,
                      double errRel = FuzzySet::RELATIVE_ERROR(),
                      std::size_t recursionMax = FuzzySet::RECURSION_MAX() ) const {
        return adaptive_simpsons(  f,
                                   lowerBound,
                                   upperBound,
                                   errRel,
                                   1,
                                   recursionMax,
                                   (this->*f)( lowerBound ),
                                   (this->*f)( ( lowerBound + upperBound ) / 2.0 ),
                                   (this->*f)( upperBound ) );
    }

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
    double adaptive_trapezoid( double (FuzzySet::*f)(double) const,
                               double a,
                               double b,
                               double errRel,
                               double recLevel,
                               int recursionMax,
                               double fa,
                               double fm,
                               double fb ) const {
        if( recLevel >= recursionMax ) {
            //std::cout << recLevel << " levels of recursion reached. Giving up on this interval." << std::endl;
            return ( b - a ) * ( fa + 2*fm + fb ) / 4.0;
        }

        double h    = b - a;
        double flm  = (this->*f)( a + h/4.0 );
        double frm  = (this->*f)( b - h/4.0 );

        double trapl  = h * ( ( fa + 2*flm + fm ) / 8.0 );
        double trapr  = h * ( ( fm + 2*frm + fb ) / 8.0 );

        double result = trapl + trapr;
        double trap   = h * ( fa + 2*fm + fb ) / 4.0;
        double err    = ( result - trap );


        if( ( recLevel <= FuzzySet::RECURSION_MIN() ) || ( std::fabs(err) > errRel * std::fabs( result ) ) ) {
            double m = (a + b) / 2.0;
            return adaptive_trapezoid( f, a, m, errRel, recLevel+1, recursionMax, fa, flm, fm )
                    + adaptive_trapezoid( f, m, b, errRel, recLevel+1, recursionMax, fm, frm, fb );
        }

        return result;
    }

    /**
      * \brief Integrates the given memberfunction numerically with the Adaptive Simpson's Method
      *
      * Analogical to the adaptive trapezoidal method this method recursively
      * applies the Simpson's rule to the subintervals until the designated
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
      * For error estimation the term above is divided by the factor 15 which
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
    double adaptive_simpsons( double (FuzzySet::*f)(double) const,
                              double a,
                              double b,
                              double errRel,
                              double recLevel,
                              int recursionMax,
                              double fa,
                              double fm,
                              double fb ) const {
        if( recLevel >= recursionMax ) {
            //std::cout << recLevel << " levels of recursion reached. Giving up on this interval." << std::endl;
            return ( b - a ) * ( fa + 4*fm + fb ) / 6.0;
        }

        // Divide the interval in half and apply Simpson's rule on each half.
        // A new approximation of the integral is given by the sum of this results.
        // As an error estimate for these we use 1/15 times the difference between it
        // and the rougher approximation based on the simple Simpson's rule on the whole
        // interval.
        double h    = b - a;
        double flm  = (this->*f)( a + h/4.0 );
        double frm  = (this->*f)( b - h/4.0 );

        double simpl  = h * ( fa + 4*flm + fm ) / 12.0;
        double simpr  = h * ( fm + 4*frm + fb ) / 12.0;

        double result = simpl + simpr;
        double simp   = h * ( fa + 4*fm + fb ) / 6.0;
        double err    = ( result - simp ) / 15.0;


        // If the error estimate exceeds the fraction of the new approximation
        // determined by the relative error tolerated by the user (errRel), refine approximation
        if( ( recLevel <= FuzzySet::RECURSION_MIN() ) || ( std::fabs(err) > errRel * std::fabs( result ) ) ) {
            double m = (a + b) / 2.0;
            return( adaptive_simpsons( f, a, m, errRel, recLevel+1, recursionMax, fa, flm, fm )
                    + adaptive_simpsons( f, m, b, errRel, recLevel+1, recursionMax, fm, frm, fb ) );
        }

        return result;
    }

};

}

#endif
