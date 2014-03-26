#define BOOST_TEST_MODULE DirectSearch_Selection
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/TypedIndividual.h>
//~ #include <shark/Algorithms/DirectSearch/Operators/Selection/LinearRanking.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/UniformRanking.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/TournamentSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/RouletteWheelSelection.h>

//#include <boost/range.hpp>

namespace shark {
	namespace detail {
		struct Extractor {
			template<typename T>
			const double & operator()( const T & t ) const {
				return( reinterpret_cast< const double & >( t ) );
			}
		};
	}
}

struct FitnessComparator {
	template<class Individual>
	bool operator()( const Individual & a, const Individual & b ) {
		return( a.fitness( shark::tag::UnpenalizedFitness() )( 0 ) < b.fitness( shark::tag::UnpenalizedFitness() )( 0 ) );
	}

};

BOOST_AUTO_TEST_CASE( Tournament_Selection ) {

	double pop[] = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };	

	shark::TournamentSelection ts;
	shark::detail::Extractor e;

	BOOST_CHECK_THROW( ts( pop, pop + 10, e, 0 ), shark::Exception );
	BOOST_CHECK_THROW( ts( pop, pop + 5, e, 8 ), shark::Exception );

	double counter[ 10 ];
	::memset( counter, 0, 10 * sizeof( double ) );

	for( unsigned int i = 0; i < 10000; i++ ) {
		counter[ std::distance( pop, ts( pop, pop + 11, e, 10 ) ) ]++;
	}

	std::copy( counter, counter + 10, std::ostream_iterator<double>( std::cout, "," ) );
	BOOST_CHECK( counter[ 9 ] > counter[ 0 ] );
}

BOOST_AUTO_TEST_CASE( RouletteWheel_Selection ) {

	double pop[] = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };	

	shark::RouletteWheelSelection ts;
	shark::detail::Extractor e;
	for( std::size_t trial = 0; trial < 100; trial++ )
		std::distance( pop, ts( pop, pop + 11, e ) );

	double pop1[] = { 1, 3 };
	unsigned int counter[2];
	counter[0] = 0;
	counter[1] = 0;
	for( unsigned int i = 0; i < 1000; i++ )
		counter[ std::distance( pop1, ts( pop1, pop1 + 2, e ) ) ]++;
	BOOST_CHECK( counter[ 0 ] > counter[1] );
}

//~ BOOST_AUTO_TEST_CASE( LinearRanking ) {
    
    //~ typedef shark::TypedIndividual< std::string > Individual;
    //~ typedef std::vector< Individual > Population;
    
    //~ Population parents( 10 );
    //~ for( Population::iterator it = parents.begin(); it != parents.end(); ++it ) {
        //~ it->fitness( shark::tag::UnpenalizedFitness() )( 0 ) = std::distance( parents.begin(), it );
        //~ **it = ( boost::format( "Parent_%1%" ) % std::distance( parents.begin(), it ) ).str();
    //~ }
    //~ Population offspring( 10 );
    //~ for( Population::iterator it = offspring.begin(); it != offspring.end(); ++it ) {
        //~ it->fitness( shark::tag::UnpenalizedFitness() )( 0 ) = std::distance( offspring.begin(), it );
        //~ **it = ( boost::format( "Offspring_%1%" ) % std::distance( offspring.begin(), it ) ).str();
    //~ }
    
    //~ std::sort( parents.begin(), parents.end(), FitnessComparator() );
    //~ std::sort( offspring.begin(), offspring.end(), FitnessComparator() );
    
    //~ Population newParents( 10 );
    //~ shark::LinearRankingSelection< shark::tag::UnpenalizedFitness > lrs;
    //~ lrs( 
        //~ parents.begin(), 
        //~ parents.end(), 
        //~ offspring.begin(), 
        //~ offspring.end(), 
        //~ newParents.begin(), 
        //~ newParents.end(), 
        //~ 3. );
    
    //~ std::cout << "########## LINEAR RANKING ##########" << std::endl;
    //~ for( Population::iterator it = newParents.begin(); it != newParents.end(); ++it )
        //~ std::cout << "Individual: " << **it << std::endl;
    
    //~ lrs( 
        //~ parents, 
        //~ offspring, 
        //~ newParents, 
        //~ 3. );
    
    //~ std::cout << "########## LINEAR RANKING (Ranges) ##########" << std::endl;
    //~ for( Population::iterator it = newParents.begin(); it != newParents.end(); ++it )
        //~ std::cout << "Individual: " << **it << std::endl;
//~ }

BOOST_AUTO_TEST_CASE( UniformRanking ) {
    
    typedef shark::TypedIndividual< std::string > Individual;
    typedef std::vector< Individual > Population;
    
    Population parents( 10 );
    for( Population::iterator it = parents.begin(); it != parents.end(); ++it ) {
        it->fitness( shark::tag::UnpenalizedFitness() )( 0 ) = std::distance( parents.begin(), it );
        **it = ( boost::format( "Parent_%1%" ) % std::distance( parents.begin(), it ) ).str();
    }
    Population offspring( 10 );
    for( Population::iterator it = offspring.begin(); it != offspring.end(); ++it ) {
        it->fitness( shark::tag::UnpenalizedFitness() )( 0 ) = std::distance( offspring.begin(), it );
        **it = ( boost::format( "Offspring_%1%" ) % std::distance( offspring.begin(), it ) ).str();
    }
    
    std::sort( parents.begin(), parents.end(), FitnessComparator() );
    std::sort( offspring.begin(), offspring.end(), FitnessComparator() );
    
    Population newParents( 10 );
    
    shark::UniformRankingSelection< shark::tag::UnpenalizedFitness > urs;
    urs( 
        parents.begin(), 
        parents.end(), 
        offspring.begin(), 
        offspring.end(), 
        newParents.begin(), 
        newParents.end() );

    std::cout << "########## UNIFORM RANKING ##########" << std::endl;
    for( Population::iterator it = newParents.begin(); it != newParents.end(); ++it )
        std::cout << "Individual: " << **it << std::endl;
    
    urs( 
        parents, 
        offspring, 
        newParents );
    
    std::cout << "########## UNIFORM RANKING (Ranges) ##########" << std::endl;
    for( Population::iterator it = newParents.begin(); it != newParents.end(); ++it )
        std::cout << "Individual: " << **it << std::endl;
    
}

/*
BOOST_AUTO_TEST_CASE( EPTournament ) {
    
    typedef shark::TypedIndividual< std::string > Individual;
    typedef std::vector< Individual > Population;
    
    Population parents( 10 );
    for( Population::iterator it = parents.begin(); it != parents.end(); ++it ) {
        it->fitness( shark::tag::UnpenalizedFitness() )( 0 ) = std::distance( parents.begin(), it );
        **it = ( boost::format( "Parent_%1%" ) % std::distance( parents.begin(), it ) ).str();
    }
    Population offspring( 10 );
    for( Population::iterator it = offspring.begin(); it != offspring.end(); ++it ) {
        it->fitness( shark::tag::UnpenalizedFitness() )( 0 ) = std::distance( offspring.begin(), it );
        **it = ( boost::format( "Offspring_%1%" ) % std::distance( offspring.begin(), it ) ).str();
    }
    
    std::sort( parents.begin(), parents.end(), FitnessComparator() );
    std::sort( offspring.begin(), offspring.end(), FitnessComparator() );
    
    Population newParents( 10 );
    
    shark::EPTournamentSelection< shark::tag::UnpenalizedFitness > ept;
    ept( 
        parents.begin(), 
        parents.end(), 
        offspring.begin(), 
        offspring.end(), 
        newParents.begin(), 
        newParents.end(), 
        3 );
    
    std::cout << "########## EP Tournament Selection ##########" << std::endl;
    for( Population::iterator it = newParents.begin(); it != newParents.end(); ++it )
        std::cout << "Individual: " << **it << std::endl;

    ept( 
        parents, 
        offspring, 
        newParents, 
        3 );
    
    std::cout << "########## EP Tournament Selection (Ranges) ##########" << std::endl;
    for( Population::iterator it = newParents.begin(); it != newParents.end(); ++it )
        std::cout << "Individual: " << **it << std::endl;
    
	}*/
