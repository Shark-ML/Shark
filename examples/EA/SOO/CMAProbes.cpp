/**
 *
 * \brief Example for examining characteristics of the CMA with the help of
 * the Probe framework.
 * 
 * \author tvoss
 *  
 * <BR><HR>
 * This file is part of Shark. This library is free software;
 * you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software
 * Foundation; either version 3, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 *  
 */
//###begin<includes>
#include <shark/Core/Probe.h>
#include <shark/Algorithms/DirectSearch/CMA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>
//###end<includes>

using namespace shark;
using namespace std;

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/units/systems/si.hpp>

#include <limits>

namespace shark {
		
    /**
     * \brief Helper class for storing the most recent value reported from a probe.
     */
    struct Store {

	/**
	 * \brief Updates the stored values according to the type contained in the variant.
	 */
	void operator()( const Probe::time_type & time, const Probe::variant_type & value ) {
	    try {
		m_currentPopulationMean = boost::get< RealVector >( value );
	    } catch( ... ) {
		// Do nothing if access to probe value fails.
	    }

	    try {
		m_sigma = boost::get< double >( value );
	    } catch( ... ) {
		// Do nothing if access to probe value fails.
	    }

	    try {
		m_covarianceMatrix = boost::get< RealMatrix >( value );
		eigensymm( m_covarianceMatrix, m_eigenVectors, m_eigenValues );
		m_condition = *max_element( m_eigenValues.begin(), m_eigenValues.end() ) / *min_element( m_eigenValues.begin(), m_eigenValues.end() );

		

	    } catch( ... ) {
		// Do nothing if access to probe value fails.
	    }
	}
	double m_condition; ///< Condition of current covariance matrix.
	RealMatrix m_covarianceMatrix; ///< Current covariance matrix.
	RealMatrix m_eigenVectors; ///< Eigenvalues of the current covariance matrix.
	RealVector m_eigenValues; ///< Eigenvalues of the current covariance matrix.
	double m_sigma; ///< Current step size.
	RealVector m_currentPopulationMean; ///< Current population mean
		
    };
}

int main( int argc, char ** argv ) {

	// Results go here.
	ofstream results( "results.txt" );
	// Plotting commands (gnuplot) go here.
	ofstream plot( "plot.txt" );
	plot << "set key outside bottom center" << endl;
	plot << "set zeroaxis" << endl;
	plot << "set border 0" << endl;
	plot << "set xrange [-4:4]" << endl;
	plot << "set yrange [-4:4]" << endl;
	// Adjust the floating-point format to scientific and increase output precision.
	results.setf( ios_base::scientific );
	results.precision( 10 );
	plot.setf( ios_base::scientific );
	plot.precision( 10 );

	// Instantiate both the problem and the optimizer.
	//###begin<init>
	Himmelblau hb;
	hb.setNumberOfVariables( 2 );
	CMA cma;
	cma.init( hb );
	//###end<init>
	// Instantiate the value store and get access to probes.
	//###begin<probes>
	Store valueStore;
	ProbeManager::ProbePtr populationMeanProbe = cma[ "PopulationMean" ];
	if( populationMeanProbe )
		populationMeanProbe->signalUpdated().connect( boost::bind( &Store::operator(), boost::ref( valueStore ), _1, _2 ) );
	ProbeManager::ProbePtr sigmaProbe = cma[ "Sigma" ];
	if( sigmaProbe )
		sigmaProbe->signalUpdated().connect( boost::bind( &Store::operator(), boost::ref( valueStore ), _1, _2 ) );
	ProbeManager::ProbePtr covarianceMatrixProbe = cma[ "CovarianceMatrix" ];
	if( covarianceMatrixProbe )
		covarianceMatrixProbe->signalUpdated().connect( boost::bind( &Store::operator(), boost::ref( valueStore ), _1, _2 ) );
	//###end<probes>
	// Iterate the optimizer until a solution of sufficient quality is found.
	//###begin<train>
	do{

	cma.step( hb );

	double sigmaX = valueStore.m_eigenValues( 0 );
	double sigmaY = valueStore.m_eigenValues( 1 );
	// Print error ellipses for covariance matrices.
	plot << "set object " 
	     << hb.evaluationCounter() 
	     << " ellipse center " 
	     << valueStore.m_currentPopulationMean( 0 ) << ","
	     << valueStore.m_currentPopulationMean( 1 ) << " size "
	     << sigmaX << ","
	     << sigmaY << " angle " 
	     << 360 + 360 * ::atan( valueStore.m_eigenVectors( 0, 1 ) / valueStore.m_eigenVectors( 0, 0 ) ) * 1./(2*M_PI) << " front fillstyle empty border 2" << endl;

	// Report information on the optimizer state and the current solution to the console.
	results << hb.evaluationCounter() << " "		// Column 1
		<< valueStore.m_condition << " "			// Column 2
		<< valueStore.m_sigma << " "				// Column 3
		<< cma.solution().value << " ";		        // Column 4
	copy(
		cma.solution().point.begin(),				
		cma.solution().point.end(),                         	// Column 5 & 6
		ostream_iterator< double >( results, " " ) 
	);
	copy( 
		valueStore.m_currentPopulationMean.begin(),// Column 7 & 8
		valueStore.m_currentPopulationMean.end(), 
		ostream_iterator< double >( results, " " ) 
	);
	results << endl;
	}while( cma.solution().value> 1E-20 );
	//###end<train>

	plot << "plot 'results.txt' every ::2 using 7:8 with lp title 'Population mean'" << endl;

	return( EXIT_SUCCESS );	
}
