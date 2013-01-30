#include <shark/Algorithms/DirectSearch/HypervolumeCalculator.h>
#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Algorithms/DirectSearch/ParetoDominanceComparator.h>
#include <shark/Algorithms/DirectSearch/FastNonDominatedSort.h>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

#include <string>
#include <vector>

namespace shark {
    typedef std::vector< shark::RealVector > FrontType;

    template<typename Stream>
    FrontType read_front( Stream & in, std::size_t noObjectives, const std::string & separator = " ", std::size_t headerLines = 0 ) {

	if( !in ) {
	    throw( shark::Exception( "Bad stream in shark::read_front", __FILE__, __LINE__ ) );
	}

	std::string line;

	// Skip header lines
	std::size_t counter = 0;
	for( counter = 0; counter < headerLines; counter++ )
	    std::getline( in, line );

	FrontType result;

	while( std::getline( in, line ) ) {
	    if( line.empty() )
		continue;

	    std::vector< std::string > tokens;
	    boost::algorithm::split( tokens, line, boost::is_any_of( separator ), boost::token_compress_on );

	    if( tokens.size() < noObjectives )
		continue;

	    shark::RealVector v( noObjectives, 0. );

	    for( std::size_t i = 0; i < noObjectives; i++ ) {
		v[ i ] = boost::lexical_cast<double>( tokens[ i ] );		
		// TODO: floating point checks.
	    }

	    result.push_back( v );
	}

	return( result );
    }
}

int main( int argc, char ** argv ) {

    boost::program_options::options_description options;
    options.add_options()
	( "noObjectives", boost::program_options::value< std::size_t >(), "# of objectives" )
	( "separator", boost::program_options::value< std::string >()->default_value( " " ), "Character that separates fields" )
	( "headerLines", boost::program_options::value< std::size_t >()->default_value( 0 ), "Amount of header lines to skip" );
		
    boost::program_options::variables_map vm;
    try {		
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options), vm);
	boost::program_options::notify(vm);
    } catch( ... ) {
	std::cerr << options << std::endl;
	return( EXIT_FAILURE );
    }

    if( vm.count( "noObjectives" ) == 0 ) {
	std::cerr << options << std::endl;
	return( EXIT_FAILURE );
    }

    std::size_t noObjectives= vm[ "noObjectives" ].as< std::size_t >();

    if( noObjectives < 2 ) {
	std::cerr << "# of objectives needs to be equal to or greater than 2." << std::endl;
	std::cerr << options << std::endl;
	return( EXIT_FAILURE );
    }


    shark::RealVector utopianFitness( noObjectives, std::numeric_limits< double >::max() );
    shark::RealVector nadirFitness( noObjectives, -std::numeric_limits< double >::max() );

    shark::FrontType front;

    try {
	front = shark::read_front( std::cin, noObjectives, vm[ "separator" ].as< std::string >(), vm[ "headerLines" ].as< std::size_t >() );
    } catch( ... ) {
	std::cerr << "Problem reading front from std::cin, aborting now." << std::endl;
	return( EXIT_FAILURE );
    }	

    shark::IdentityFitnessExtractor e;
    shark::BoundingBoxCalculator< shark::IdentityFitnessExtractor > bbc( e, utopianFitness, nadirFitness );

    shark::FrontType::iterator it = front.begin();
    while( it != front.end() ) {
	bbc( *it );
	++it;
    }

    std::cout << utopianFitness << std::endl;
    std::cout << nadirFitness << std::endl;

    return( EXIT_SUCCESS );

}
