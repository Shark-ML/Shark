#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/UniformCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ackley.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

using namespace shark;

namespace example {

	typedef Individual< RealVector, double, double> IndividualType;
	typedef std::vector< IndividualType > Population;

	struct FitnessComparator {
		bool operator()( const IndividualType & a, const IndividualType & b ) {
			return( a.unpenalizedFitness() < b.unpenalizedFitness() );
		}

	};
}


int main( int argc, char ** argv ) {

	const unsigned Mu           = 15;
	const unsigned Lambda       = 100;
	const unsigned Dimension    = 30;
	const double InitialSigma   = 3.;
	
	// Instantiate the objective function
	benchmarks::Ackley ackley( Dimension );
	
	// Initialize the mutation distribution
	MultiVariateNormalDistribution mutationDistribution;
	mutationDistribution.resize(Dimension);
	
	example::IndividualType prototypeIndividual;
	prototypeIndividual.chromosome() = InitialSigma;
	
	example::Population parents( Mu, prototypeIndividual );
	example::Population offspring( Lambda );
	
	// Initialize parents (not a god idea to start in a single point, shouldn't do this in practice)
	for(auto& ind: parents ) {
		ind.searchPoint() = ackley.proposeStartingPoint( );
	}
	
	// Evolutionary operators
	UniformCrossover uniform;
	
	// standard deviations for mutation of sigma
	double     tau0 = 1. / sqrt(2. * Dimension);
	double     tau1 = 1. / sqrt(2. * sqrt( static_cast<double>( Dimension ) ) );

	
	while( ackley.evaluationCounter() < 10000 ) {
	
		for( std::size_t i = 0; i < offspring.size(); i++ ) {
		
			// Select two parent individuals at random
			example::Population::const_iterator mom = parents.begin() + random::discrete(random::globalRng,  std::size_t(0), parents.size() - 1 );
			example::Population::const_iterator dad = parents.begin() + random::discrete(random::globalRng,  std::size_t(0), parents.size() - 1 );
		
			// Recombine step size
			offspring[i].chromosome() = random::uni(random::globalRng,  mom->chromosome(), dad->chromosome() );			
			// Mutate step size
			offspring[i].chromosome() *= random::logNormal(random::globalRng,  0, tau0 + tau1 );
			
			// Recombine search points
			offspring[i].searchPoint() = uniform(random::globalRng, mom->searchPoint(), dad->searchPoint() );
			// Mutate search point
			offspring[i].searchPoint() = offspring[i].chromosome() * mutationDistribution(random::globalRng).first;
	
			// Assign fitness
			offspring[i].unpenalizedFitness() = ackley.eval( offspring[i].searchPoint() );
		}
	
		// Selection 
		example::FitnessComparator comp;
		std::sort( offspring.begin(), offspring.end(), comp );
		std::copy( offspring.begin(), offspring.begin() + Mu, parents.begin() );

		
		std::cout 	<< ackley.evaluationCounter() << " " 
				<< parents.front().unpenalizedFitness() << " "
				<< parents.front().chromosome()  
				<< std::endl;
	}	
}
