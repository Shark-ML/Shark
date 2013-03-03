#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_GSP_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_GSP_H

#include <shark/ObjectiveFunctions/AbstractMultiObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {
/// \brief Real-valued benchmark function with two objectives.
struct GSP : public AbstractMultiObjectiveFunction< VectorSpace<double> >
{
	typedef AbstractMultiObjectiveFunction< VectorSpace<double> > super;
	GSP(std::size_t numVariables=5) : super(2), m_handler(SearchPointType(numVariables,0),SearchPointType(numVariables,10000))  {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= IS_CONSTRAINED_FEATURE;
		m_features |= HAS_CONSTRAINT_HANDLER;
		m_features |= CAN_PROVIDE_CLOSEST_FEASIBLE;
		m_name = "GSP";
	}
	
	std::size_t numberOfVariables()const{
		return m_handler.dimensions();
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}
	
	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_handler.setBounds(
			SearchPointType(numberOfVariables,0),
			SearchPointType(numberOfVariables,10000)
		);
	}
	
	BoxConstraintHandler<SearchPointType> const& getConstraintHandler()const{
		return m_handler;
	}

	void configure( PropertyTree const& node ) {
		m_gamma = node.get<double>( "gamma" );
	}

	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		ResultType value( 2 );
		double alpha = 1. / ( 2. * m_gamma );

		double sum1 = 0., sum2 = 0.;

		for( unsigned int i = 0; i < x.size(); i++ ) {
			sum1 += sqr( x( i ) );
			sum2 += sqr( 1 - x( i ) );
		}

		double alphaN = 1. / ( std::pow( x.size(), alpha ) );

		value[0] = alphaN * std::pow( sum1, alpha );
		value[1] = alphaN * std::pow( sum2, alpha );

		return( value );
	}
private:
	BoxConstraintHandler<SearchPointType> m_handler;
	double m_gamma;
};

}
#endif
