#include <shark/Statistics/Statistics.h>

#include <boost/program_options.hpp>

#include <iostream>
#include <iterator>
#include <fstream>

int main( int argc, char ** argv ) {

  boost::program_options::options_description options;
  options.add_options()
    ( "headerLine", "Print out header line with column lines." );

  boost::program_options::variables_map vm;
  try {
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options), vm);
    boost::program_options::notify(vm);
  } catch( ... ) {
    std::cerr << options << std::endl;
    return( EXIT_FAILURE );
  }

  shark::Statistics stats;
  stats = std::for_each( std::istream_iterator< double >( std::cin ), std::istream_iterator<double>(), stats );

  if( vm.count( "headerLine" ) > 0 ) {
    std::cout << "Mean Variance Median LowerQuartile UpperQuartile Min Max Count" << std::endl;
  }
  std::cout << 
    stats( shark::Statistics::Mean() ) << " " << 
    stats( shark::Statistics::Variance() ) << " " <<
    stats( shark::Statistics::Median() ) << " " <<
    stats( shark::Statistics::LowerQuartile() ) << " " <<
    stats( shark::Statistics::UpperQuartile() ) << " " <<
    stats( shark::Statistics::Min() ) << " " <<
    stats( shark::Statistics::Max() ) << " " <<
    stats( shark::Statistics::NumSamples() ) << " ";

  return( EXIT_SUCCESS );
}
