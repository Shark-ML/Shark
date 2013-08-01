#include <shark/Algorithms/DirectSearch/HypervolumeApproximator.h>
#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Algorithms/DirectSearch/ParetoDominanceComparator.h>
#include <shark/Algorithms/DirectSearch/FastNonDominatedSort.h>
#include <shark/Rng/GlobalRng.h>

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
	( "referencePoint", boost::program_options::value< std::string >(), "Reference point in the format [m]( r1, ..., rm)" )
	( "separator", boost::program_options::value< std::string >()->default_value( " " ), "Character that separates fields" )
	( "headerLines", boost::program_options::value< std::size_t >()->default_value( 0 ), "Amount of header lines to skip" )
	( "Epsilon", boost::program_options::value< double >()->default_value( 1E-2 ), "Error bound for FPRAS" )
	( "Delta", boost::program_options::value< double >()->default_value( 1E-2 ), "Error probability for FPRAS" );
		

    boost::program_options::variables_map vm;
    try {		
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options), vm);
	boost::program_options::notify(vm);
    } catch( ... ) {
	std::cerr << options << std::endl;
	return( EXIT_FAILURE );
    }

    if( vm.count( "referencePoint" ) == 0 ) {
	std::cerr << options << std::endl;
	return( EXIT_FAILURE );
    }

    std::string referencePoint = vm[ "referencePoint" ].as< std::string >();

    if( referencePoint.size() == 0 ) {
	std::cerr << "Missing reference point, aborting now." << std::endl;
	std::cerr << options << std::endl;
	return( EXIT_FAILURE );
    }


    shark::RealVector refPoint;
    std::stringstream ss( referencePoint );
    ss >> refPoint;

    if( refPoint.size() < 2 ) {
	std::cerr << "Reference point needs to be of size 2 or greater, aborting now." << std::endl;
	return( EXIT_FAILURE );
    }

    shark::FrontType front;

    try {
	front = shark::read_front( std::cin, refPoint.size(), vm[ "separator" ].as< std::string >(), vm[ "headerLines" ].as< std::size_t >() );
    } catch( ... ) {
	std::cerr << "Problem reading front from std::cin, aborting now." << std::endl;
	return( EXIT_FAILURE );
    }	

    shark::IdentityFitnessExtractor e;
    shark::HypervolumeApproximator<shark::Rng> hc;
    shark::ParetoDominanceComparator< shark::tag::PenalizedFitness > pdc;

    shark::FrontType::iterator it = front.begin();
    while( it != front.end() ) {

	if( !(pdc( *it, refPoint, e ) >= shark::ParetoDominanceComparator< shark::tag::PenalizedFitness >::A_WEAKLY_DOMINATES_B) ) {
	    it = front.erase( it );
	    continue;
	}

	bool dominated = false;
	for( shark::FrontType::iterator itt = front.begin(); itt != front.end(); ++itt ) {
	    if( pdc( *it, *itt, e ) <= shark::ParetoDominanceComparator< shark::tag::PenalizedFitness >::B_WEAKLY_DOMINATES_A ) {
		dominated = true;
		break;
	    }
	}
		
	if( dominated )
	    it = front.erase( it );
	else
	    ++it;
    }

    // std::copy( front.begin(), front.end(), std::ostream_iterator< shark::RealVector >( std::cout, "\n" ) );

    std::cout << hc( front.begin(), front.end(), e, refPoint, vm[ "Epsilon" ].as<double>(), vm[ "Delta" ].as<double>() ) << std::endl;

    return( EXIT_SUCCESS );

}
