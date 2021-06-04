/*!
 * 
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
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#include <shark/Algorithms/DirectSearch/Operators/Recombination/PartiallyMappedCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/RouletteWheelSelection.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>

#include <shark/LinAlg/Base.h>

#include <boost/format.hpp>
#if BOOST_VERSION == 106000
#include <boost/type_traits/ice.hpp>//Required because of boost 1.60 bug
#endif
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphml.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/metric_tsp_approx.hpp>
#include <boost/property_map/dynamic_property_map.hpp>
#include <numeric>//for iota

using namespace shark;
typedef IntVector Tour;

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

typedef Individual< Tour,double > IndividualType;
typedef std::vector< IndividualType > Population;


/// \brief Calculates the cost of a tour w.r.t. to a cost matrix.
struct TspTourLength : public SingleObjectiveFunction{

	/// \brief Default c'tor, initializes the cost matrix.
	TspTourLength( RealMatrix const& costMatrix) : m_costMatrix( costMatrix )
	{ }

	std::string name() const
	{ return "TspTourLength"; }

	std::size_t numberOfVariables() const
	{ return m_costMatrix.size1(); }

	/**
	 * \brief Calculates the costs of the supplied tour.
	 */
	ResultType eval( const SearchPointType & input ) const {		    	    
		SIZE_CHECK( input.size() == m_costMatrix.size1() );
		m_evaluationCounter++;
		ResultType result(0);
		for( std::size_t i = 0; i < input.size() - 1; i++ ) {
			result += m_costMatrix( input( i ), input( i+1 ) );
		}
		return result;
	}

	RealMatrix m_costMatrix;
};

int main( int argc, char ** argv ) {

	// Define the problem instance
	Graph g( 10 );
	// Iterate the graph and assign a label to every vertex
	boost::graph_traits<Graph>::vertex_iterator v, v_end;
	for( boost::tie(v,v_end) = boost::vertices(g); v != v_end; ++v )
	boost::put(boost::vertex_color_t(), g, *v, ( boost::format( "City_%1%" ) % *v ).str() );
	// Get hold of the weight map and the color map for calculation and visualization purposes.
	WeightMap weightMap = boost::get( boost::edge_weight, g );
	ColorMap colorMap = boost::get( boost::edge_color, g );
    
	// Iterate the graph and insert distances.
	Edge e;
	bool inserted = false;
    
	RealMatrix costMatrix( 10, 10 );
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
	RouletteWheelSelection rws;
	// Variation (crossover) operator
	PartiallyMappedCrossover pmx;    
	// Fitness function instance.
	TspTourLength ttl( costMatrix );
	ttl.init();

	// Size of the parent population
	const std::size_t mu = 100;
	// Size of the offspring population
	const std::size_t lambda = 100;

	// Parent population
	Population parents( mu );
	// Offspring population
	Population offspring( lambda );

	// Default tour: 0,...,9
	Tour t( 10 );
	std::iota( t.begin(),t.end(), 0 );

	// Initialize parents
	for( std::size_t i = 0; i < parents.size(); i++ ) {
		parents[i].searchPoint() = t;
		// Generate a permutation.
		std::random_shuffle( parents[ i ].searchPoint().begin(), parents[ i ].searchPoint().end() );
		// Evaluate the individual.
		parents[i].penalizedFitness() = parents[i].unpenalizedFitness() = ttl( parents[i].searchPoint() );
	}

	// Loop until maximum number of fitness function evaluations is reached.
	while( ttl.evaluationCounter() < 10000 ) {
		RealVector selectionProbabilities(parents.size());
		for(std::size_t i = 0; i != parents.size(); ++i){
			selectionProbabilities(i) = parents[i].unpenalizedFitness();
		}
		selectionProbabilities/= sum(selectionProbabilities);
		// Crossover candidates.
		for( std::size_t i = 0; i < offspring.size() - 1; i+=2 ) {
			// Carry out fitness proportional fitness selection and
			// perform partially mapped crossover on parent individuals.
			offspring[ i ] = *rws( random::globalRng, parents.begin(), parents.end(), selectionProbabilities );
			offspring[ i+1 ] = *rws( random::globalRng, parents.begin(), parents.end(), selectionProbabilities );
			pmx(random::globalRng, offspring[ i ], offspring[ i+1 ]);

			// Evaluate offspring individuals.
			offspring[ i ].penalizedFitness() = 
			offspring[ i ].unpenalizedFitness() = ttl( offspring[ i ].searchPoint() );		

			offspring[ i+1 ].penalizedFitness() = 
			offspring[ i+1 ].unpenalizedFitness() = ttl( offspring[ i+1 ].searchPoint()  );
		}

		// Swap parent and offspring population.
		std::swap( parents, offspring );
	}

	// Sort in ascending order to find best individual.
	std::sort( parents.begin(), parents.end(), IndividualType::FitnessOrdering());
    
	// Extract the best tour currently known.
	Tour final = parents.front().searchPoint();

	// Mark the final tour green in the graph.
	bool extracted = false;
	for( std::size_t i = 0; i < final.size() - 1; i++ ) {
		boost::tie( e, extracted ) = boost::edge( Vertex( final( i ) ), Vertex( final( i + 1 ) ), g );
	
	if( extracted )
		colorMap[ e ] = "green";
	}

	// Calculate approximate solution
	double len = 0.0;
	std::vector< Vertex > approxTour;
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
	std::cout << parents.front().searchPoint() << " -> " << parents.front().unpenalizedFitness() << std::endl;
	// Output approximate solution and corresponding fitness.
	std::cout << "Approx: " << len << " vs. GA: " << parents.front().unpenalizedFitness() << std::endl;

	return EXIT_SUCCESS;
}
