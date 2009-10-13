
/**
 * \file GeneralizedBellFS.cpp
 *
 * \brief FuzzySet with a generalized bell-shaped membership function
 * 
 * \authors Thomas Voß
 */

/* $log$ */
#include <Fuzzy/GeneralizedBellFS.h>

#include <math.h>

GeneralizedBellFS::GeneralizedBellFS( double slope, double center, double width, double scale ) : FuzzySet( scale ),
m_slope( slope ),
m_center( center ),
m_width( width ) {
}

double GeneralizedBellFS::mu( double x ) const {
	// printf( "(II) GBell( %f ) = %f \n", x, 1/(1+pow( (x-m_center) / (m_width), 2*m_slope ) ) );
	return( 1/(1+pow( fabs( (x-m_center) / (m_width) ), 2*m_slope ) ) );
}

double GeneralizedBellFS::getMin() const {
	return( m_center - m_width * pow(1E6 - 1, 1/(2*m_slope)) );
}

double GeneralizedBellFS::getMax() const {
	return( m_center + m_width * pow(1E6 - 1, 1/(2*m_slope)) );
}

