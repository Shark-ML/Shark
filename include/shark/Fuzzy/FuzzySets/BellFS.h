
/**
*
* \brief FuzzySet with a bell-shaped (Gaussian) membership function
* 
* \authors Marc Nunkesser
*/

/* $log$ */
#ifndef SHARK_FUZZY_BELLFS_H
#define SHARK_FUZZY_BELLFS_H

#include <shark/SharkDefs.h>
#include <shark/Fuzzy/LinguisticTerm.h>

#include <boost/math/special_functions.hpp>

#include <climits>
#include <cassert>

namespace shark {

	/**
	* \brief FuzzySet with a bell-shaped (Gaussian) membership function.
	* 
	* This class implements a FuzzySet with membership function:
	* 
	* \f[
	*      \mu(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-offset)^2}{2\sigma^2}}
	* \f]
	* 
	* <img src="../images/BellFS.png">
	*  
	*/
	class BellFS: virtual public FuzzySet {
	public:

		/**
		* \brief Constructor.
		* 
		* @param sigma controlls the width of the Gaussian
		* @param offset position of the center of the peak
		* @param scale scales the whole function
		*/
                BellFS( double sigma = 1., double offset = 0., double scale = 1 ) : m_sigma( sigma ),
			m_offset( offset ),
			m_scale( scale ),
			m_threshold( 1E-6 ) {
				updateBounds();
		}


		// The bell mf is represented
		// by three parameters sigma, offset and scale:
		// bell(sigma,offset,scale) = 1/(sigma*sqrt(2pi))*exp(-(sigma*(x-offset))²/2sigma^2)
		//

		/**
		* \brief Defuzzifies the set by returning the Bell's offset.
		* 
		*/
		virtual double defuzzify() const {
			return( m_offset );
		};

		/**
		* \brief Returns the lower boundary of the support.
		* 
		* @return the min. value for which the membership function is nonzero (or exceeds a
		* given threshold)
		*/
		virtual double min() const {
			return( m_min );
		};

		/**
		* \brief Returns the upper boundary of the support.
		* 
		* @return the max. value for which the membership function is nonzero (or exceeds a
		* given threshold)
		*/
		virtual double max() const {
			return( m_max );
		};

		/**
		* \brief Returns the theshold under which values of the function are set to be zero.
		* 
		* @return the theshold
		*/
		double threshold() const {
			return( m_threshold );
		};

		/**
		* \brief Sets the theshold under which values of the function are set to be zero.
		* 
		* @param thresh the theshold
		*/
		void setThreshold( double thresh ) {
			m_threshold = thresh;

			updateBounds();
		}

		/**
		* \brief Sets the parameters of the fuzzy set.
		* 
		* @param sigma controlls the width of the Gaussian
		* @param offset position of the center of the peak
		* @param scale scales the whole function
		*/
		// void setParams(double sigma, double offset, double scale = 1);

	private:
		// overloaded operator () - the mu-function
		double mu( double x ) const {
			return( m_scale/m_sigma * factor * ::exp(-::boost::math::pow<2>( x-m_offset )/( 2*boost::math::pow<2>( m_sigma ) ) ) );
		}

		void updateBounds() {
			double radicand = ::log(m_threshold*m_sigma/m_scale*factor2);
			if( radicand > 0 ) {
				m_min = -std::numeric_limits<double>::max();
				m_max = std::numeric_limits<double>::max();
			} else {
				const double temp = ::sqrt(-2*radicand) * m_sigma;
				m_min = m_offset-temp;
				m_max = m_offset+temp;
			}
		}

		double m_sigma;
		double m_offset;
		double m_scale;// parameters of a sigmoidal MF
		double m_threshold;
		double m_min; // to avoid repeated calculation, these are stored.
		double m_max;

		static const double factor; //  = 0.5*M_2_SQRTPI*M_SQRT1_2; //1/(sqrt(2pi))
		static const double factor2; // = 2*M_SQRT2/M_2_SQRTPI; //sqrt(2pi)
	};

	const double BellFS::factor = 0.5*M_2_SQRTPI*M_SQRT1_2;
	const double BellFS::factor2 = 2*M_SQRT2/M_2_SQRTPI;

}
#endif
