#include <shark/Algorithms/DirectSearch/FitnessComparator.h>
#include <shark/Algorithms/DirectSearch/TypedIndividual.h>
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
	
	typedef TypedIndividual< RealVector, double > Individual;
	typedef std::vector< Individual > Population;
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
	
	shark::example::Individual prototypeIndividual;
	prototypeIndividual.get<0>() = InitialSigma;
	
	shark::example::Population parents( Mu, prototypeIndividual );
	shark::example::Population offspring( Lambda );
	
	// Initialize parents (not a god idea to start in a single point, shouldn't do this in practice)
	BOOST_FOREACH( shark::example::Individual & ind, parents ) {
		ackley.proposeStartingPoint( *ind );
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
			offspring[i].get< 0 >() = shark::Rng::uni( (*mom).get< 0 >(), (*dad).get< 0 >() );			
			// Mutate step size
			offspring[i].get< 0 >() *= shark::Rng::logNormal( 0, tau0 + tau1 );
			
			// Recombine search points
			*offspring[i] = uniform( **mom, **dad );
			// Mutate search point
			*offspring[i] = offspring[i].get< 0 >() * mutationDistribution().first;
	
			// Assign fitness
			offspring[i].fitness( shark::tag::UnpenalizedFitness() )(0) = ackley.eval( *offspring[i] );
		}
	
		// Selection 
		shark::UnpenalizedFitnessComparator comp;
		std::sort( offspring.begin(), offspring.end(), comp );
		std::copy( offspring.begin(), offspring.begin() + Mu, parents.begin() );

		
		std::cout 	<< ackley.evaluationCounter() << " " 
					<< parents.front().fitness( shark::tag::UnpenalizedFitness() )( 0 ) << " "
					<< parents.front().get< 0 >()  
					<< std::endl;
	}	
}
