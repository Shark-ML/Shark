
/**
 * \file GeneralizedBellFS.cpp
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

