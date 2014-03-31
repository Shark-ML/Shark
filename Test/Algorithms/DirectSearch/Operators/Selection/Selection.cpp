#define BOOST_TEST_MODULE DirectSearch_Selection
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/LinearRanking.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/UniformRanking.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/TournamentSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/RouletteWheelSelection.h>

struct FitnessComparator {
	template<class Individual>
	bool operator()( const Individual & a, const Individual & b ) {
		return a.unpenalizedFitness() < b.unpenalizedFitness();
	}

};

struct NumberComparator {
	template<class Individual>
	bool operator()( const Individual & a, const Individual & b ) {
		return a < b;
	}

};

BOOST_AUTO_TEST_CASE( Tournament_Selection ) {

	double pop[] = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };

	shark::TournamentSelection<NumberComparator> ts;
	ts.tournamentSize = 10;

	std::vector<unsigned int> counter(11,0);

	for( unsigned int i = 0; i < 10000; i++ ) {
		counter[ std::distance( pop, ts( pop, pop + 11 ) ) ]++;
	}

	std::copy( counter.begin(), counter.begin() + 11, std::ostream_iterator<double>( std::cout, "," ) );
	BOOST_CHECK( counter[ 10 ] > counter[ 0 ] );
}

BOOST_AUTO_TEST_CASE( RouletteWheel_Selection ) {

	double pop[] = { 0,1,2,3,4,5,6,7,8,9,10 };	

	shark::RouletteWheelSelection ts;
	
	shark::RealVector prob(11,0);
	for(std::size_t i = 0; i != 11; ++i){
		prob(i) = shark::Rng::uni(0.5,1);
	}
	prob/=sum(prob);
	
	shark::RealVector hist(11,0);
	for( unsigned int i = 0; i < 1000000; i++ )
		hist[ *ts( pop, pop + 11,prob) ]+=1.0;
	
	hist /=1000000;
	for(std::size_t i = 0; i != 11; ++i){
		BOOST_CHECK_CLOSE(hist[i],prob[i], 1.0);
	}
}

//~ BOOST_AUTO_TEST_CASE( LinearRanking ) {
    
    //~ typedef shark::TypedIndividual< std::string > Individual;
    //~ typedef std::vector< Individual > Population;
    
    //~ Population parents( 10 );
    //~ for( Population::iterator it = parents.begin(); it != parents.end(); ++it ) {
        //~ it->unpenalizedFitness() = std::distance( parents.begin(), it );
        //~ **it = ( boost::format( "Parent_%1%" ) % std::distance( parents.begin(), it ) ).str();
    //~ }
    //~ Population offspring( 10 );
    //~ for( Population::iterator it = offspring.begin(); it != offspring.end(); ++it ) {
        //~ it->unpenalizedFitness() = std::distance( offspring.begin(), it );
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

//~ BOOST_AUTO_TEST_CASE( UniformRanking ) {
    
    //~ typedef shark::Individual< std::string,double > Individual;
    //~ typedef std::vector< Individual > Population;
    
    //~ Population parents( 10 );
    //~ for( Population::iterator it = parents.begin(); it != parents.end(); ++it ) {
        //~ it->unpenalizedFitness() = std::distance( parents.begin(), it );
        //~ **it = ( boost::format( "Parent_%1%" ) % std::distance( parents.begin(), it ) ).str();
    //~ }
    //~ Population offspring( 10 );
    //~ for( Population::iterator it = offspring.begin(); it != offspring.end(); ++it ) {
        //~ it->unpenalizedFitness() = std::distance( offspring.begin(), it );
        //~ **it = ( boost::format( "Offspring_%1%" ) % std::distance( offspring.begin(), it ) ).str();
    //~ }
    
    //~ std::sort( parents.begin(), parents.end(), FitnessComparator() );
    //~ std::sort( offspring.begin(), offspring.end(), FitnessComparator() );
    
    //~ Population newParents( 10 );
    
    //~ shark::UniformRankingSelection< shark::tag::UnpenalizedFitness > urs;
    //~ urs( 
        //~ parents.begin(), 
        //~ parents.end(), 
        //~ offspring.begin(), 
        //~ offspring.end(), 
        //~ newParents.begin(), 
        //~ newParents.end() );

    //~ std::cout << "########## UNIFORM RANKING ##########" << std::endl;
    //~ for( Population::iterator it = newParents.begin(); it != newParents.end(); ++it )
        //~ std::cout << "Individual: " << **it << std::endl;
    
    //~ urs( 
        //~ parents, 
        //~ offspring, 
        //~ newParents );
    
    //~ std::cout << "########## UNIFORM RANKING (Ranges) ##########" << std::endl;
    //~ for( Population::iterator it = newParents.begin(); it != newParents.end(); ++it )
        //~ std::cout << "Individual: " << **it << std::endl;
    
//~ }

/*
BOOST_AUTO_TEST_CASE( EPTournament ) {
    
    typedef shark::TypedIndividual< std::string > Individual;
    typedef std::vector< Individual > Population;
    
    Population parents( 10 );
    for( Population::iterator it = parents.begin(); it != parents.end(); ++it ) {
        it->unpenalizedFitness() = std::distance( parents.begin(), it );
        **it = ( boost::format( "Parent_%1%" ) % std::distance( parents.begin(), it ) ).str();
    }
    Population offspring( 10 );
    for( Population::iterator it = offspring.begin(); it != offspring.end(); ++it ) {
        it->unpenalizedFitness() = std::distance( offspring.begin(), it );
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
