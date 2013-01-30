#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_GSP_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_GSP_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {
/// \brief Real-valued benchmark function with two objectives.
struct GSP : public AbstractObjectiveFunction< VectorSpace<double>, std::vector<double> > {

	GSP() : m_numberOfVariables( 0 ) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= IS_CONSTRAINED_FEATURE;
		m_name = "GSP";
	}

	unsigned int noObjectives() const {
		return 2 ;
	}

	void init() {
	}

	void configure( const boost::property_tree::ptree & node ) {
		m_gamma = node.get<double>( "gamma" );
	}

	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		std::vector<double> value( 2 );
		double alpha = 1. / ( 2. * m_gamma );

		double sum1 = 0., sum2 = 0.;

		for( unsigned int i = 0; i < x.size(); i++ ) {
			sum1 += sqr( x( i ) );
			sum2 += sqr( 1 - x( i ) );
		}

		double alphaN = 1. / ( std::pow( x.size(), alpha ) );

		value[0] = alphaN * ::pow( sum1, alpha );
		value[1] = alphaN * ::pow( sum2, alpha );

		return( value );
	}

	void proposeStartingPoint( SearchPointType & x ) const {
		x.resize( m_numberOfVariables );
		for( unsigned int i = 0; i < m_numberOfVariables; i++ )
			x( i ) = Rng::uni( 0., 10000. );
	}

	bool isFeasible( const SearchPointType & v ) const {
		for( unsigned int i = 0; i < m_numberOfVariables; i++ ) {
			if( v( i ) < 0 || v( i ) > 10000 )
				return false;
		}
		return true;
	}

	void closestFeasible( SearchPointType & v ) const {
		for( unsigned int i = 0; i < numberOfVariables(); i++ ) {
			v( i ) = std::min( v( i ), 10000. );
			v( i ) = std::max( v( i ), 0. );
		}
	}

private:
	unsigned int m_numberOfVariables;
	double m_gamma;
};

//template<> struct ObjectiveFunctionTraits<GSP> {
//
//	static GSP::SearchPointType lowerBounds( unsigned int n ) {
//		return( GSP::SearchPointType( n, 0. ) );
//	}
//
//	static GSP::SearchPointType upperBounds( unsigned int n ) {
//		return( GSP::SearchPointType( n, 10000. ) );
//	}
//
//};
}
#endif
