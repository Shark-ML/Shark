#ifndef SHARK_EA_GRID_H
#define SHARK_EA_GRID_H

#include <boost/foreach.hpp>

#include <map>
#include <vector>

namespace shark {
    /** \cond */
	class AdaptiveGrid {
	public:
		struct Hypercube {
			Hypercube() : m_size( 0. ),
				m_noSolutions( 0 ),
				m_isOccupied( false ) {
			}

			double m_size;
			unsigned int m_noSolutions;
			bool m_isOccupied;
		};

		typedef std::map< std::size_t, Hypercube >::iterator iterator;
		typedef std::map< std::size_t, Hypercube >::const_iterator const_iterator;

		std::map< std::size_t, Hypercube > m_denseGrid;
		iterator m_mostOccupiedHypercube;

		/**
		* Number of bi-divisions of the objective space
		*/
		unsigned int m_noBisections;

		/**
		*
		* Grid lower bounds
		*/
		RealVector m_lowerBounds;

		/**
		* Grid upper bounds
		*/
		RealVector m_upperBounds;

		/************************************************************************/
		/* Extens of the grid                                                   */
		/************************************************************************/
		RealVector m_extents;

		/**
		* Constructor.
		* Creates an instance of AdaptativeGrid.
		* @param noBisections Number of bi-divisions of the objective space.
		* @param nObjetives Number of objectives of the problem.
		*/
		AdaptiveGrid( unsigned int noBisections = 0, unsigned int noObjectives = 0 ) {
			reset( noBisections, noObjectives );
		} //AdaptativeGrid

		void reset( unsigned int noBisections, unsigned int noObjectives ) {
			m_noBisections = noBisections;

			m_lowerBounds = RealVector( noObjectives, std::numeric_limits<double>::max() );
			m_upperBounds = RealVector( noObjectives, -std::numeric_limits<double>::max() );

			m_extents = RealVector( noObjectives, 0 );

			// m_denseGrid = std::vector< Hypercube >( static_cast<std::size_t>( ::pow( 2., noBisections * noObjectives ) ) );
			m_denseGrid.clear();
			m_mostOccupiedHypercube = m_denseGrid.end();
		}

		/**
		*  Updates the grid limits considering the solutions contained in a
		*  <code>SolutionSet</code>.
		*  @param solutionSet The <code>SolutionSet</code> considered.
		*/
		template<typename Set>
		void updateLimits( const Set & solutionSet ) {

			m_lowerBounds = RealVector( m_lowerBounds.size(), std::numeric_limits<double>::max() );
			m_upperBounds = RealVector( m_lowerBounds.size(), -std::numeric_limits<double>::max() );

			typename Set::const_iterator it;
			for( it = solutionSet.begin(); it != solutionSet.end(); ++it ) {

				for( unsigned int i = 0; i < it->fitness( shark::tag::PenalizedFitness() ).size(); i++ ) {
					m_lowerBounds( i ) = std::min( m_lowerBounds( i ), it->fitness( shark::tag::PenalizedFitness() )[i] );
					m_upperBounds( i ) = std::max( m_upperBounds( i ), it->fitness( shark::tag::PenalizedFitness() )[i] );
				}
			}
			m_extents = m_upperBounds - m_lowerBounds;
		} //updateLimits

		/**
		* Updates the grid adding solutions contained in a specific
		* <code>SolutionSet</code>.
		* <b>REQUIRE</b> The grid limits must have been previously calculated.
		* @param solutionSet The <code>SolutionSet</code> considered.
		*/
		template<typename Set>
		void addSolutionSet( const Set & solutionSet) {
			m_mostOccupiedHypercube = m_denseGrid.begin();
			int itt;
			typename Set::const_iterator it;
			for( it = solutionSet.begin(); it != solutionSet.end(); ++it ) {
				itt = location( *it );

				if( itt == -1 )
					throw( shark::Exception( "AdaptiveGrid::addSolutionSet: The grid limits need to be calculated before." ) );

				/*itt->second.m_noSolutions++;
				if( m_mostOccupiedHypercube->second.m_noSolutions > itt->second.m_noSolutions )
				std::swap( m_mostOccupiedHypercube, itt );*/
				addSolution( itt );
			} // for

			//The grid has been updated, so also update ocuppied's hypercubes
			// calculateOccupied();
		}


		/**
		* Updates the grid limits and the grid content adding the solutions contained
		* in a specific <code>SolutionSet</code>.
		* @param solutionSet The <code>SolutionSet</code>.
		*/
		template<typename Set>
		void updateGrid( const Set & solutionSet ){

			//Update lower and upper limits
			updateLimits( solutionSet );

			m_denseGrid.clear();
			m_mostOccupiedHypercube = m_denseGrid.end();

			//Add the population
			addSolutionSet(solutionSet);
		} //updateGrid


		/**
		* Updates the grid limits and the grid content adding a new
		* <code>Solution</code>.
		* If the solution falls out of the grid bounds, the limits and content of the
		* grid must be re-calculated.
		* @param solution <code>Solution</code> considered to update the grid.
		* @param solutionSet <code>SolutionSet</code> used to update the grid.
		*/
		template<typename Solution, typename Set>
		void updateGrid( const Solution & solution, const Set & solutionSet ) {

			int it = location( solution );
			if ( it == -1 ) {
				//Update lower and upper limits
				updateLimits( solutionSet );

				//Actualize the lower and upper limits whit the individual
				for( std::size_t i = 0; i < solution.fitness( shark::tag::PenalizedFitness() ).size(); i++ ){
					m_lowerBounds( i ) = std::min( m_lowerBounds( i ), solution.fitness( shark::tag::PenalizedFitness() )[i] );
					m_upperBounds( i ) = std::max( m_upperBounds( i ), solution.fitness( shark::tag::PenalizedFitness() )[i] );
				} // for

				m_extents = m_upperBounds - m_lowerBounds;

				m_denseGrid.clear();
				m_mostOccupiedHypercube = m_denseGrid.end();

				//add the population
				addSolutionSet(solutionSet);
			} // if
		} //updateGrid


		/**
		* Calculates the hypercube of a solution.
		* @param solution The <code>Solution</code>.
		*/
		template<typename Solution>
		int location( const Solution & solution ) const {
			//Create a int [] to store the range of each objective
			std::vector< std::size_t > positions( solution.fitness( shark::tag::PenalizedFitness() ).size(), 0 );

			//Calculate the position for each objective
			for( std::size_t obj = 0; obj < solution.fitness( shark::tag::PenalizedFitness() ).size(); obj++ ) {

				if( solution.fitness( shark::tag::PenalizedFitness() )[ obj ] > m_upperBounds( obj ) )
					return( -1 );
				if( solution.fitness( shark::tag::PenalizedFitness() )[ obj ] < m_lowerBounds( obj ) )
					return( -1 );

				if( solution.fitness( shark::tag::PenalizedFitness() )[ obj ] == m_lowerBounds[obj] ) {
					positions[ obj ] = 0;
					continue;
				}

				if( solution.fitness( shark::tag::PenalizedFitness() )[ obj ] == m_upperBounds[ obj ] ) {
					positions[obj] = static_cast<std::size_t>( (::pow( 2.0, static_cast<double>( m_noBisections ) ) )-1 );
					continue;
				}


				double tmpSize = m_extents( obj );
				double value = solution.fitness( shark::tag::PenalizedFitness() )[ obj ];
				double account = m_lowerBounds( obj );
				std::size_t ranges = static_cast<std::size_t>( (::pow( 2.0, static_cast<double>( m_noBisections ) ) ) );

				for( unsigned int b = 0; b < m_noBisections; b++ ) {
					tmpSize /= 2.0;
					ranges /= 2;
					if( value > (account + tmpSize) ) {
						positions[ obj ] += ranges;
						account += tmpSize;
					}
				}
			}

			//Calculate the location into the hypercubes
			std::size_t location = 0;
			for( unsigned int obj = 0; obj < solution.fitness( shark::tag::PenalizedFitness() ).size(); obj++ ) {
				location += positions[obj] * static_cast<std::size_t>( (::pow( 2.0, obj * static_cast<double>( m_noBisections ) ) ) );
			}
			return( location );
		} //location

		/**
		* Returns the value of the most populated hypercube.
		* @return The hypercube with the maximum number of solutions.
		*/
		iterator mostPopulated() {
			return( m_mostOccupiedHypercube );
		} // getMostPopulated

		/**
		* Returns the number of solutions into a specific hypercube.
		* @param idx Number of the hypercube.
		* @return The number of solutions into a specific hypercube.
		*/
		unsigned int locationDensity( std::size_t idx ) {
			iterator it = m_denseGrid.find( idx );
			if( it == m_denseGrid.end() )
				return( 0 );
			return( it->second.m_noSolutions );
		} //getLocationDensity

		/**
		* Decreases the number of solutions into a specific hypercube.
		* @param location Number of hypercube.
		*/
		int removeSolution( std::size_t location ) {
			iterator it = m_denseGrid.find( location );
			if( it == m_denseGrid.end() ) //TODO: Throw exception here?
				return( -1 );
			//Decrease the solutions in the location specified.
			it->second.m_noSolutions--;

			if( m_mostOccupiedHypercube == it )
				for( iterator itt = m_denseGrid.begin(); itt != m_denseGrid.end(); ++itt )
					if( itt->second.m_noSolutions > m_mostOccupiedHypercube->second.m_noSolutions )
						m_mostOccupiedHypercube = itt;

			//If hypercubes[location] now becomes to zero, then update ocupped hypercubes
			if( it->second.m_noSolutions == 0 ) {
				m_denseGrid.erase( it );
				calculateOccupied();

				return( 0 );
			}
			return( it->second.m_noSolutions );
		} //removeSolution

		/**
		* Increases the number of solutions into a specific hypercube.
		* @param location Number of hypercube.
		*/
		void addSolution( std::size_t location ) {
			Hypercube & h = m_denseGrid[location];
			h.m_noSolutions++;

			if( m_mostOccupiedHypercube != m_denseGrid.end() ) {
				if( h.m_noSolutions > m_mostOccupiedHypercube->second.m_noSolutions ) {
					m_mostOccupiedHypercube = m_denseGrid.find( location );
				}
			} else
				m_mostOccupiedHypercube = m_denseGrid.find( location );

			//if hypercubes[location] becomes to one, then recalculate
			//the occupied hypercubes
			if( h.m_noSolutions == 1 )
				calculateOccupied();
		} //addSolution

		/**
		* Returns the number of bi-divisions performed in each objective.
		* @return the number of bi-divisions.
		*/
		unsigned int noBisections() {
			return( m_noBisections );
		} //getBisections

		/**
		* Returns a random hypercube using a rouleteWheel method.
		*  @return the number of the selected hypercube.
		*/
		// 	public int rouletteWheel(){
		// 		//Calculate the inverse sum
		// 		double inverseSum = 0.0;
		// 		for (int i = 0; i < hypercubes_.length; i++) {
		// 			if (hypercubes_[i] > 0) {
		// 				inverseSum += 1.0 / (double)hypercubes_[i];
		// 			}
		// 		}
		// 
		// 		//Calculate a random value between 0 and sumaInversa
		// 		double random = PseudoRandom.randDouble(0.0,inverseSum);
		// 		int hypercube = 0;
		// 		double accumulatedSum = 0.0;
		// 		while (hypercube < hypercubes_.length){
		// 			if (hypercubes_[hypercube] > 0) {
		// 				accumulatedSum += 1.0 / (double)hypercubes_[hypercube];
		// 			} // if
		// 
		// 			if (accumulatedSum > random) {
		// 				return hypercube;
		// 			} // if
		// 
		// 			hypercube++;
		// 		} // while
		// 
		// 		return hypercube;
		// 	} //rouletteWheel

		/**
		* Calculates the number of hypercubes having one or more solutions.
		* return the number of hypercubes with more than zero solutions.
		*/
		std::size_t calculateOccupied() {
			return( m_denseGrid.size() );	
		} //calculateOcuppied

		/**
		* Returns the number of hypercubes with more than zero solutions.
		* @return the number of hypercubes with more than zero solutions.
		*/
		std::size_t occupiedHypercubes(){
			return( m_denseGrid.size() );
		} // occupiedHypercubes


		/**
		* Returns a random hypercube that has more than zero solutions.
		* @return The hypercube.
		*/
// 		iterator randomOccupiedHypercube(){
// 			return( m_denseGrid.begin() + Rng::discrete( 0, m_denseGrid.size() ) );
// 		} //randomOccupiedHypercube
	}; //AdaptativeGrid
}

/** \endcond */
#endif 
