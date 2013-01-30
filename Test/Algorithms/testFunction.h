#ifndef TEST_ALGORITHMS_TESTFUNCTION_H
#define TEST_ALGORITHMS_TESTFUNCTION_H

#include <shark/Core/Chart.h>
#include <shark/Core/Renderers/HighchartRenderer.h>
#include <shark/Statistics/Statistics.h>

#include <boost/parameter.hpp>
#include <boost/progress.hpp>

#include <fstream>

namespace shark {
BOOST_PARAMETER_NAME(optimizer)    // Note: no semicolon
BOOST_PARAMETER_NAME(function)
BOOST_PARAMETER_NAME(trials)
BOOST_PARAMETER_NAME(iterations)
BOOST_PARAMETER_NAME(epsilon)
BOOST_PARAMETER_NAME(storageFrequency)
BOOST_PARAMETER_NAME(fstop)

BOOST_PARAMETER_FUNCTION(
    (void),
    test_function,
    tag,
    (required ( in_out(optimizer), * ) ( in_out(function), * ) )
    (optional
     (trials, *, 10)
     (iterations, *, 1000)
     (epsilon, *, 1E-15)
     (storageFrequency, *, 100)
     (fstop, *, 1E-10)
     )) {

  Chart c;
  c.title().m_text = ( boost::format( "%1% on %2%" ) % optimizer.name() % function.name() ).str();

  boost::progress_display pd( trials * iterations );
		
  shark::Statistics stats;

  for( size_t trial =0;trial != static_cast<size_t>(trials);++trial ){

    Chart::Series trialSeries;
    trialSeries.m_name = ( boost::format( "Trial %1%" ) % trial ).str();
    trialSeries.m_type = shark::Chart::Series::LINE_TYPE;
    optimizer.init(function);
			
    double error=0;

    for( size_t iteration = 0; iteration < static_cast<size_t>(iterations); ++iteration ) {
      optimizer.step( function );
      error=optimizer.solution().value;

      if( iteration % storageFrequency == 0 )
        trialSeries.m_data.push_back( shark::Chart::Series::ElementType( double( iteration ), error ) );

      ++pd;

      if( fstop > error )
        break;
    }

    c.series().push_back( trialSeries );
    stats( error );
  }

  BOOST_CHECK_SMALL( stats( shark::Statistics::Median() ), epsilon );

  std::ofstream out( ( boost::format( "%1%_%2%.html" ) % optimizer.name() % function.name() ).str().c_str() );
  HighchartRenderer< std::ofstream > renderer( out );

  renderer.render( c );
}
}
template<class Optimizer,class Function>
void testFunction(Optimizer& optimizer,Function& function,unsigned int trials,unsigned int iterations, double epsilon = 1.e-15){

  boost::progress_display pd( trials * iterations );

  for(unsigned int trial =0;trial!=trials;++trial){
    optimizer.init(function);
    double error=0;
    for(size_t iteration=0;iteration<iterations;++iteration)
    {
      optimizer.step(function);
      error=optimizer.solution().value;

      ++pd;
    }
    BOOST_CHECK_SMALL(error,epsilon);
  }
}

#endif
