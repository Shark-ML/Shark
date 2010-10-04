
/**
 * \file FuzzySet.cpp
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
#include <Fuzzy/FuzzySet.h>
#include <Fuzzy/FuzzyException.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <Fuzzy/NDimFS.h>


#include <algorithm>
#ifdef __SOLARIS__
#include <climits>
#endif
#ifdef __LINUX__
#include <float.h>
#endif

FuzzySet::FuzzySet( double _scale ) : scaleFactor(_scale) { }

FuzzySet::~FuzzySet() { }


// FuzzySet::operator NDimFS&()
// {  NDimFS::FuzzyArrayType out(1);
//    out.push_back(this);
//    NDimFS ndfs(out);
//    return(ndfs);
// };

void FuzzySet::makeGNUPlotData(const std::string fileName,
                               unsigned int steps,
                               double lowerBound,
                               double upperBound) const {
	if (steps<3) {
		steps=3;
	};
	double increment = upperBound/steps-lowerBound/steps;
	// =(upperbound-lowerbound)/steps, avoiding overflow
	assert(fileName!="");
	std::ofstream dataFile(fileName.c_str(),std::ios::out);
	if (!dataFile) {
		throw(FuzzyException(10,"Cannot write to disk"));
	};
	if (lowerBound == upperBound)
		dataFile<<lowerBound<<" "<<operator()(lowerBound)<<std::endl;
	else
		for (double d=std::max(lowerBound-increment,-std::numeric_limits<double>::max());d<=std::min(upperBound+increment,std::numeric_limits<double>::max());d+=increment) {
			dataFile<<d<<" "<<operator()(d)<<std::endl;
		};
	dataFile.close();
}


void FuzzySet::makeGNUPlotData(const std::string fileName,
                               unsigned int steps) const {
	makeGNUPlotData(fileName, steps, getMin(),getMax());
};


double FuzzySet::defuzzify(double l,
                           double u,
                           double errRel,
                           int recursionMax ) const
// We use the centroid method, according to Bothe p146 we get
// y=integral(y*mu(y)*dy)/integral(mu(y)*dy))
// We approximate y=Sum(y_i*mu(y_i/sum(mu(y_i)
{
	double lowerBound = std::max( getMin(), l );
	double upperBound = std::min( getMax(), u );
    
	if ( upperBound == lowerBound )
        return lowerBound;
        
	double numerator   = integrate( &FuzzySet::xmu, lowerBound, upperBound, 0.00001 );
	double denominator = integrate( &FuzzySet::mu,  lowerBound, upperBound, 0.00001 );
    
	return( ( denominator != 0 ) ? ( numerator / denominator ) : 0 );
}

double FuzzySet::defuzzifyMax(unsigned int steps,
                              double l,
                              double u) const
{
	assert( steps != 0 );
    
	double lowerBound = std::max( getMin(), l );
	double upperBound = std::min( getMax(), u );
    
	if ( upperBound == lowerBound )
        return lowerBound;
        
	double increment = ( upperBound - lowerBound ) / steps;
    
	//make sure that borders will be taken into consideration
	lowerBound = std::max( lowerBound - increment,-std::numeric_limits<double>::max() );
	upperBound = std::min( upperBound + increment, std::numeric_limits<double>::max() );
    
	double out = 0;    // point where mf is max
	double maxVal = 0; // current max mf value
    
	double temp;
    assert( increment > 0 );
    
	for ( double d = lowerBound; d <= upperBound; d += increment ) {
		temp = operator()(d);
		out = ( temp > maxVal ? d : out );
		maxVal = std::max( maxVal, temp );
	};
    
	return out;
}

double  FuzzySet::getMin() const {
	return( 0 );
}
double  FuzzySet::getMax() const {
	return( 0 );
}
// void    FuzzySet::setParams() {} // dummy method


double FuzzySet::integrate( double (FuzzySet::*f)(double) const,
                            double lowerBound,
                            double upperBound,
                            double errRel,
                            int recursionMax ) const
{
  //return adaptive_trapezoid( f, 
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
 * \brief Implementation of the adaptive trapezoidal method
 * 
 * @param f the function to be inegrated
 * @param a lower bound of the region of integration
 * @param b upper bound of the region of integration
 * @param errRel relative approximation error that is tollerated
 * @param recLevel the current level of recursion
 * @param recursionMax max. depth of recursion
 * @param fa the value of the function for a, i.e. f(a) 
 * @param fm the value of the function for (a+b)/2, i.e. f((a+b)/2)
 * @param fb the value of the function for b, i.e. f(b)  
 * 
 * @return the integral
 */
double FuzzySet::adaptive_trapezoid( double (FuzzySet::*f)(double) const, 
                                     double a, 
                                     double b,
                                     double errRel, 
                                     double recLevel,
                                     int    recursionMax,
                                     double fa,
                                     double fm,
                                     double fb ) const
{

    
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
            

    if( ( recLevel <= RECURSION_MIN ) || ( fabs(err) > errRel * fabs( result ) ) ) {
        double m = (a + b) / 2.0;
        return adaptive_trapezoid( f, a, m, errRel, recLevel+1, recursionMax, fa, flm, fm )
             + adaptive_trapezoid( f, m, b, errRel, recLevel+1, recursionMax, fm, frm, fb );
    }
    
    return result;
}


/**
 * \brief Implementation of the adaptive Simpson's method
 * 
 * @param f the function to be inegrated
 * @param a lower bound of the region of integration
 * @param b upper bound of the region of integration
 * @param errRel relative approximation error that is tollerated
 * @param recLevel the current level of recursion
 * @param recursionMax max. depth of recursion
 * @param fa the value of the function for a, i.e. f(a) 
 * @param fm the value of the function for (a+b)/2, i.e. f((a+b)/2)
 * @param fb the value of the function for b, i.e. f(b)
 *   
 * @return the integral
 */
double FuzzySet::adaptive_simpsons( double (FuzzySet::*f)(double) const,
                                    double a,
                                    double b,
                                    double errRel,
                                    double recLevel,
                                    int    recursionMax,
                                    double fa,
                                    double fm,
                                    double fb ) const
{
    
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
    if( ( recLevel <= RECURSION_MIN ) || ( fabs(err) > errRel * fabs( result ) ) ) {
        double m = (a + b) / 2.0;
        return adaptive_simpsons( f, a, m, errRel, recLevel+1, recursionMax, fa, flm, fm )
             + adaptive_simpsons( f, m, b, errRel, recLevel+1, recursionMax, fm, frm, fb );
    }
    
    return result;
}
