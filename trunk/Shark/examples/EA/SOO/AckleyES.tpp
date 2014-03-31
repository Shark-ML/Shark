#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/UniformCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ackley.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

#include <boost/foreach.hpp>

namespace shark {

namespace example {
	struct Chromosome {
		shark::MultiVariateNormalDistribution m_mutationDistribution;
	};

	typedef Individual< RealVector, double, double> IndividualType;
	typedef std::vector< IndividualType > Population;

	struct FitnessComparator {
		bool operator()( const IndividualType & a, const IndividualType & b ) {
			return( a.unpenalizedFitness() < b.unpenalizedFitness() );
		}

	};
}

}

int main( int argc, char ** argv ) {

	const unsigned Mu           = 15;
	const unsigned Lambda       = 100;
	const unsigned Dimension    = 30;
	const double InitialSigma   = 3.;
	
	// Instantiate the objective function
	shark::Ackley ackley( Dimension );
	
	// Initialize the mutation distribution
	shark::MultiVariateNormalDistribution mutationDistribution;
	mutationDistribution.setCovarianceMatrix( shark::blas::identity_matrix< double >( Dimension ) );
	
	shark::example::IndividualType prototypeIndividual;
	prototypeIndividual.chromosome() = InitialSigma;
	
	shark::example::Population parents( Mu, prototypeIndividual );
	shark::example::Population offspring( Lambda );
	
	// Initialize parents (not a god idea to start in a single point, shouldn't do this in practice)
	BOOST_FOREACH( shark::example::IndividualType & ind, parents ) {
		ackley.proposeStartingPoint( ind.searchPoint() );
	}
	
	// Evolutionary operators
	shark::UniformCrossover uniform;
	
	// standard deviations for mutation of sigma
	double     tau0 = 1. / sqrt(2. * Dimension);
	double     tau1 = 1. / sqrt(2. * sqrt( static_cast<double>( Dimension ) ) );

	
	while( ackley.evaluationCounter() < 10000 ) {
	
		for( std::size_t i = 0; i < offspring.size(); i++ ) {
		
			// Select two parent individuals at random
			shark::example::Population::const_iterator mom = parents.begin() + shark::Rng::discrete( 0, parents.size() - 1 );
			shark::example::Population::const_iterator dad = parents.begin() + shark::Rng::discrete( 0, parents.size() - 1 );
		
			// Recombine step size
			offspring[i].chromosome() = shark::Rng::uni( mom->chromosome(), dad->chromosome() );			
			// Mutate step size
			offspring[i].chromosome() *= shark::Rng::logNormal( 0, tau0 + tau1 );
			
			// Recombine search points
			offspring[i].searchPoint() = uniform( mom->searchPoint(), dad->searchPoint() );
			// Mutate search point
			offspring[i].searchPoint() = offspring[i].chromosome() * mutationDistribution().first;
	
			// Assign fitness
			offspring[i].unpenalizedFitness() = ackley.eval( offspring[i].searchPoint() );
		}
	
		// Selection 
		shark::example::FitnessComparator comp;
		std::sort( offspring.begin(), offspring.end(), comp );
		std::copy( offspring.begin(), offspring.begin() + Mu, parents.begin() );

		
		std::cout 	<< ackley.evaluationCounter() << " " 
				<< parents.front().unpenalizedFitness() << " "
				<< parents.front().chromosome()  
				<< std::endl;
	}	
}
