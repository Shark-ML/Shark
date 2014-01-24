/*!
 * 
 * \file        TSP.tpp
 *
 * \brief       A 10-city traveling salesman problem.

 * The implementation follows:
 * <p>
 * D. E. Goldberg and R. Lingle, Alleles, loci, and traveling
 * salesman problem. In <em> Proc. of the International Conference on
 * Genetic Algorithms and Their Applications</em>, pages 154-159,
 * Pittsburg, PA, 1985 </p> 

 * The traveling salsman problem is a combinatorial optimization task. A
 * salesman is supposed to visit $n$ cities. Each travelling connection
 * is associated with a cost (i.e. the time fot the trip). The problem is
 * to find the cheapest round-route that visits each city exactly once
 * and returns to the starting point.

 * 
 *
 * \author      T. Voss
 * \date        -
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>

#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/RouletteWheelSelection.h>
#include <shark/Algorithms/DirectSearch/TypedIndividual.h>

#include <shark/LinAlg/Base.h>

#include <boost/format.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphml.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/metric_tsp_approx.hpp>
#include <boost/property_map/dynamic_property_map.hpp>
#include <boost/range/algorithm_ext/iota.hpp>

typedef shark::IntVector Tour;

/**
 * \brief Defines the problem, i.e., its cost matrix.
 */
const double cities[10][10] = {
    {       0,      28,     57,     72,     81,     85,     80,     113,    89,     80 },
    {       28,     0,      28,     45,     54,     57,     63,     85,     63,     63 },
    {       57,     28,     0,      20,     30,     28,     57,     57,     40,     57 },
    {       72,     45,     20,     0,      10,     20,     72,     45,     20,     45 },
    {       81,     54,     30,     10,     0,      22,     81,     41,     10,     41 },
    {       85,     57,     28,     20,     22,     0,      63,     28,     28,     63 },
    {       80,     63,     57,     72,     81,     63,     0,      80,     89,     113 },
    {       113,    85,     57,     45,     41,     28,     80,     0,      40,     80 },
    {       89,     63,     40,     20,     10,     28,     89,     40,     0,      40 },
    {       80,     63,     57,     45,     41,     63,     113,    80,     40,     0 }
};

namespace shark {

	typedef boost::adjacency_matrix< boost::undirectedS,
		boost::property< boost::vertex_color_t, std::string >,
		boost::property< boost::edge_weight_t, double,
			boost::property< boost::edge_color_t, std::string >
		>,
		boost::no_property 
	> Graph;
	typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
	typedef boost::graph_traits<Graph>::edge_descriptor Edge;
	typedef boost::property_map<Graph, boost::edge_weight_t>::type WeightMap;
	typedef boost::property_map<Graph, boost::edge_color_t>::type ColorMap;

	typedef TypedIndividual< Tour > Individual;
	typedef std::vector< Individual > Population;

	template<typename FitnessTag>
	bool compare_fitness( const Individual & a, const Individual & b ) {
		return( a.fitness( FitnessTag() )( 0 ) < b.fitness( FitnessTag() )( 0 ) );
	}

	/**
	 * \brief Calculates the cost of a tour w.r.t. to a cost matrix.
	 */
	struct TspTourLength : public shark::AbstractObjectiveFunction< shark::VectorSpace< std::size_t >, double > {
		typedef shark::AbstractObjectiveFunction< shark::VectorSpace< std::size_t >, double > base_type;

		/**
		 * \brief Default c'tor, initializes the cost matrix.
		 */
		TspTourLength( const shark::RealMatrix & costMatrix = shark::RealMatrix() ) : m_costMatrix( costMatrix )
		{ }

		std::string name() const
		{ return "TspTourLength"; }

		std::size_t numberOfVariables() const
		{ return m_costMatrix.size1(); }

		/**
		 * \brief Calculates the costs of the supplied tour.
		 */
		base_type::ResultType eval( const base_type::SearchPointType & input ) const {		    	    
			SIZE_CHECK( input.size() == m_costMatrix.size1() );
		    m_evaluationCounter++;
			base_type::ResultType result( 0 );
			for( std::size_t i = 0; i < input.size() - 1; i++ ) {
				result += m_costMatrix( input( i ), input( i+1 ) );
			}
		    return( result );
		}

		shark::RealMatrix m_costMatrix;
	};

	/**
	 * @brief Implements partially mapped crossover
	 * 
	 * Makes sure that only correct tours on the graph
	 * are generated.
	 */
	struct PartiallyMappedCrossover {

		/**
		 * @brief Performs the crossover.
		 *
		 * @param [in] parent1 First parent individual.
		 * @param [in] parent2 Second parent individual.
		 * @returns A pair of offspring individuals.
		 */
		template<typename IndividualType>
		std::pair<IndividualType, IndividualType> operator()(
				const IndividualType & parent1,
				const IndividualType & parent2 ) const
		{
			std::pair< IndividualType, IndividualType > offspring( parent1, parent2 );

			const Tour & t1 = *parent1;
			const Tour & t2 = *parent2;

			std::size_t cuttingPoint1 = shark::Rng::discrete( 0, t1.size() - 1 );
			std::size_t cuttingPoint2 = shark::Rng::discrete( 0, t2.size() - 1 );

			while( cuttingPoint1 == cuttingPoint2 )
			cuttingPoint2 = shark::Rng::discrete( 0, t2.size() - 1 );

			if( cuttingPoint1 > cuttingPoint2 )
			std::swap( cuttingPoint1, cuttingPoint2 );

			Tour r1( t1.size(), -1 ), r2( t2.size(), -1 );

			for( std::size_t i = cuttingPoint1; i <= cuttingPoint2; i++ ) {
				(*offspring.first)( i ) = t2( i );
				(*offspring.second)( i ) = t1( i );

				r1[ t2( i ) ] = t1( i );
				r2[ t1( i ) ] = t2( i );
			}

			for( std::size_t i = 0; i < t1.size(); i++) {
				if ((i >= cuttingPoint1) && (i <= cuttingPoint2)) continue;

				std::size_t n1 = t1[i] ;
				std::size_t m1 = r1[n1] ;

				std::size_t n2 = t2[i] ;
				std::size_t m2 = r2[n2] ;

				while (m1 != std::numeric_limits<std::size_t>::max()) {
					n1 = m1 ;
					m1 = r1[m1] ;
				}
				while (m2 != std::numeric_limits<std::size_t>::max()) {
					n2 = m2 ;
					m2 = r2[m2] ;
				}
				(*offspring.first)[i] = n1 ;
				(*offspring.second)[i] = n2 ;
			}

			return( offspring );
		}
	};
}


int main( int argc, char ** argv ) {

	// Define the problem instance
	shark::Graph g( 10 );
	// Iterate the graph and assign a label to every vertex
	boost::graph_traits<shark::Graph>::vertex_iterator v, v_end;
	for( boost::tie(v,v_end) = boost::vertices(g); v != v_end; ++v )
	boost::put(boost::vertex_color_t(), g, *v, ( boost::format( "City_%1%" ) % *v ).str() );
	// Get hold of the weight map and the color map for calculation and visualization purposes.
	shark::WeightMap weightMap = boost::get( boost::edge_weight, g );
	shark::ColorMap colorMap = boost::get( boost::edge_color, g );
    
	// Iterate the graph and insert distances.
	shark::Edge e;
	bool inserted = false;
    
	shark::RealMatrix costMatrix( 10, 10 );
	for( std::size_t i = 0; i < costMatrix.size1(); i++ ) {
		for( std::size_t j = 0; j < costMatrix.size1(); j++ ) {

			if( i == j ) continue;

			costMatrix(i,j) = cities[i][j];

			// Add the edge
			boost::tie( e, inserted ) = boost::add_edge( i, j, g );
			if( inserted ) {
				// Remember distance between cities i and j.
				weightMap[ e ] = cities[i][j];
				// Mark the edge as blue.
				colorMap[ e ] = "blue";
			}
		}
	}

	// Mating selection operator
	shark::RouletteWheelSelection rws;
	// Fitness extractor for extracting fitness values from individuals.
	shark::soo::FitnessExtractor fe;
	// Variation (crossover) operator
	shark::PartiallyMappedCrossover pmc;    
	// Fitness function instance.
	shark::TspTourLength ttl( costMatrix );

	// Size of the parent population
	const std::size_t mu = 100;
	// Size of the offspring population
	const std::size_t lambda = 100;

	// Parent population
	shark::Population parents( mu );
	// Offspring population
	shark::Population offspring( lambda );

	// Default tour: 0,...,9
	Tour t( 10 ); boost::iota( t, 0 );

	// Initialize parents
	for( std::size_t i = 0; i < parents.size(); i++ ) {
		*( parents[i] ) = t;
		// Generate a permutation.
		std::random_shuffle( ( *parents[ i ] ).begin(), ( *parents[ i ] ).end() );
		// Evaluate the individual.
		parents[i].fitness( shark::tag::PenalizedFitness() )( 0 ) = parents[i].fitness( shark::tag::UnpenalizedFitness() )( 0 ) = ttl( *( parents[i] ) );
	}

	// Loop until maximum number of fitness function evaluations is reached.
	while( ttl.evaluationCounter() < 10000 ) {

		// Crossover candidates.
		for( std::size_t i = 0; i < offspring.size() - 1; i+=2 ) {
			// Carry out fitness proportional fitness selection and
			// perform partially mapped crossover on parent individuals.
			std::pair< 
			shark::Individual, 
			shark::Individual 
			> result = pmc( *rws( parents.begin(), parents.end(), fe ), *rws( parents.begin(), parents.end(), fe ) );
			offspring[ i ] = result.first;
			offspring[ i + 1 ] = result.second;

			// Evaluate offspring individuals.
			offspring[ i ].fitness( shark::tag::PenalizedFitness() )( 0 ) = 
			offspring[ i ].fitness( shark::tag::UnpenalizedFitness() )( 0 ) = ttl( *offspring[ i ] );		

			offspring[ i+1 ].fitness( shark::tag::PenalizedFitness() )( 0 ) = 
			offspring[ i+1 ].fitness( shark::tag::UnpenalizedFitness() )( 0 ) = ttl( *offspring[ i+1 ] );
		}

		// Swap parent and offspring population.
		std::swap( parents, offspring );
	}

	// Sort in ascending order to find best individual.
	std::sort( parents.begin(), parents.end(), shark::compare_fitness< shark::tag::UnpenalizedFitness > );
    
	// Extract the best tour currently known.
	Tour final = *parents.front();

	// Mark the final tour green in the graph.
	bool extracted = false;
	for( std::size_t i = 0; i < final.size() - 1; i++ ) {
	boost::tie( e, extracted ) = boost::edge( shark::Vertex( final( i ) ), shark::Vertex( final( i + 1 ) ), g );
	
	if( extracted )
		colorMap[ e ] = "green";
	}

	// Calculate approximate solution
	double len = 0.0;
	std::vector< shark::Vertex > approxTour;
	boost::metric_tsp_approx( g, boost::make_tsp_tour_len_visitor( g, std::back_inserter( approxTour ) , len, weightMap ) );
	// Mark the approximation red in the graph.
	for( std::size_t i = 0; i < approxTour.size() - 1; i++ ) {
	boost::tie( e, extracted ) = boost::edge( approxTour[ i ], approxTour[ i+1 ], g );
	
	if( extracted )
		colorMap[ e ] = "red";
	}

	// Output the graph and the final path
	std::ofstream outGraphviz( "graph.dot" );
	boost::dynamic_properties dp;
	dp.property( "node_id", boost::get( boost::vertex_color, g ) );
	dp.property( "weight", boost::get( boost::edge_weight, g ) );
	dp.property( "label", boost::get( boost::edge_weight, g ) );
	dp.property( "color", boost::get( boost::edge_color, g ) );
	boost::write_graphviz_dp( outGraphviz, g, dp );

	// Output best solution and corresponding fitness.
	std::cout << *parents.front() << " -> " << parents.front().fitness( shark::tag::UnpenalizedFitness() ) << std::endl;
	// Output approximate solution and corresponding fitness.
	std::cout << "Approx: " << len << " vs. GA: " << parents.front().fitness( shark::tag::UnpenalizedFitness() ) << std::endl;

	return( EXIT_SUCCESS );
}
