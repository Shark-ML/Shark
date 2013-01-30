#ifndef WILCOXONRANKSUMTEST_H
#define WILCOXONRANKSUMTEST_H

#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/erf.hpp>

namespace shark {
	/// \brief  Wilcoxon rank-sum test / Mann–Whitney U test.
        ///
        /// Non-parametric statistical hypothesis test for assessing
        /// whether two independent samples of observations have
        /// equally large elements.
	struct WilcoxonRankSumTest {
		/// Stores result of Wilcoxon rank-sum test.
		struct Result {
			double m_pH0;
			double m_pH1ARightOfB;
			double m_pH1BRightOfA;
		};
		/// Stores information about an observation.
		struct Element {
			enum Sample {
				SAMPLE_A,
				SAMPLE_B
			};
			double m_value;
			double m_rank;
			Sample m_sample;

			bool operator<( const Element & rhs ) const {
				return( m_value < rhs.m_value );
			}
		};

		std::size_t faculty( std::size_t n ) {
			if( n == 1 )
				return( 1 );

			return( n * faculty( n-1 ) );			
		}

		double frequency( double u, int sampleSizeA, int sampleSizeB ) {
			if( u < 0. || sampleSizeA < 0 || sampleSizeB < 0 )
				return( 0. );
			
			if( u == 0 && sampleSizeA >= 0 && sampleSizeB >= 0 )
				return( 1 );

			return( frequency( u - sampleSizeB, sampleSizeA - 1, sampleSizeB ) + frequency( u, sampleSizeA, sampleSizeB - 1 ) );
		}

		template<typename SampleType>
		Result operator()( const SampleType & x, const SampleType & y ) {

			Result result;

			std::vector<Element> combinedSample( x.size() + y.size() );
			
			for( unsigned int i = 0; i < x.size(); i++ ) {
				combinedSample[i].m_value = x.at( i );
				combinedSample[i].m_sample = Element::SAMPLE_A;
			}
			for( unsigned int i = 0; i < y.size(); i++ ) {
				combinedSample[i + x.size()].m_value = y.at( i );
				combinedSample[i + x.size()].m_sample = Element::SAMPLE_B;
			}

			std::sort( combinedSample.begin(), combinedSample.end() );

			std::pair< std::vector<Element>::iterator, std::vector<Element>::iterator > p;
			std::vector<Element>::iterator it = combinedSample.begin();
			std::size_t rank = 1;
			while( it != combinedSample.end() ) {
				p = std::equal_range( it, combinedSample.end(), *it );				
				it->m_rank = rank;
				std::size_t c = std::distance( p.first, p.second );				
				if( c == 1 ) {
					++it;
					rank += c;
					continue;
				}
				it = p.first;
				while( it != p.second ) {
					it->m_rank = rank + c/2.;
					++it;					
				}

				rank += c;																	
			}
			
			// std::copy( combinedSample.begin(), combinedSample.end(), std::ostream_iterator<Element>( std::cout, "" ) );

			double wA = 0.;
			double wB = 0.;

			for( it = combinedSample.begin(); it != combinedSample.end(); ++it ) {
				if( it->m_sample == Element::SAMPLE_A )
					wA += it->m_rank;
				else
					wB += it->m_rank;
			}

			// Check on consistency
			/*if( wA + wB != ( 0.5 * (x.size() + y.size()) * (x.size() + y.size() + 1) ) )
				throw( Exception( "Wilcoxon sums are inconsistent", __LINE__, __FILE__ ) );*/

			double uA = wA - x.size() * ( x.size() + 1 ) / 2.;
			double uB = wB - y.size() * ( y.size() + 1 ) / 2.;

			double pA = 1.;
			double pB = 1.;

			std::cout << "ua: " << uA << ", ub: " << uB << std::endl;

			double cases = boost::math::binomial_coefficient<double>( x.size() + y.size(), x.size() );
			if( cases < 0 || cases > 1000000 ) {
				std::cout << "Normal approximation" << std::endl;
				// normal approximation
				double mu = (0.5 * x.size()) * y.size();
				double sigma = ::sqrt( ( ( x.size() + 1.0) * x.size() ) * y.size() / 12.0 );
				double Za = (uA - mu) / sigma;
				double Zb = (uB - mu) / sigma;
				pA = 0.5 * boost::math::erfc(-Za / M_SQRT2);
				pB = 0.5 * boost::math::erfc(-Zb / M_SQRT2);
			} else {
				pA = ( faculty( x.size() ) * faculty( y.size() ) ) / ( faculty( x.size() + y.size() ) ) * frequency( uA, x.size(), y.size() );
				pB = ( faculty( x.size() ) * faculty( y.size() ) ) / ( faculty( x.size() + y.size() ) ) * frequency( uB, x.size(), y.size() );
			}

			result.m_pH1 = 1.-pA;
			result.m_pH2 = 1.-pB;

			return( result );
		}

	};

	template<typename Stream>
	Stream & operator<<( Stream & s, const WilcoxonRankSumTest::Element & element ) {
		s << "Element=" << element.m_value << " " << element.m_rank << " " << element.m_sample << std::endl;
	}

}

#endif // WILCOXONRANKSUMTEST_H
