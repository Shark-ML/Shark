#ifndef SHARK_EA_PAES_H
#define SHARK_EA_PAES_H

#include <shark/Algorithms/AbstractOptimizer.h>

#include <shark/Algorithms/DirectSearch/ExternalGridArchive.h>
#include <shark/Algorithms/DirectSearch/TypedIndividual.h>
#include <shark/Algorithms/DirectSearch/ParetoDominanceComparator.h>

#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>

#include <shark/LinAlg/LinAlg.h>

#include <boost/foreach.hpp>

namespace shark {
	namespace detail {

	    /**
	     * \brief Implements the Pareto-archived evolutionary strategy.
	     */
		template<typename SearchPointType, typename Mutator = PolynomialMutator>
		class PAES {
			typedef TypedIndividual<SearchPointType> Individual;
			typedef ExternalGridArchive<Individual, shark::ParetoDominanceComparator< shark::tag::PenalizedFitness > > Archive;
		public:
			PAES() {
				init();
			}

			void configure( const PropertyTree & node ) {
				init( node.get<std::size_t>( "ArchiveCapacity", 100 ) );
			}	

			void init( std::size_t archiveCapacity = 100 ) {
				m_archive.setCapacity( archiveCapacity );
			}

			template<typename Function>
			void init( const Function & f ) {
				m_parent.setNoObjectives( f.noObjectives() );

				m_archive.grid().reset( 5, f.noObjectives() );

				f.proposeStartingPoint( *m_parent );

				boost::tuple< typename Function::ResultType, typename Function::ResultType > result = m_evaluator( f, *m_parent );
				m_parent.fitness( shark::tag::PenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >( result );
				m_parent.fitness( shark::tag::UnpenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >( result );

				m_archive( m_parent );
			}

			template<typename Function>
			typename Function::SolutionSetType step( const Function & f ) {

				Individual offspring( m_parent );
				Mutator mutator;
				mutator.init( f );
				mutator( offspring );
					
				boost::tuple< typename Function::ResultType, typename Function::ResultType > result = m_evaluator( f, *offspring );
				offspring.fitness( shark::tag::PenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >( result );
				offspring.fitness( shark::tag::UnpenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >( result );

				ParetoDominanceComparator< shark::tag::PenalizedFitness > pdc;
				
				switch( pdc( offspring, m_parent ) ) {
					case 2:
					case 3:
						m_parent = offspring;
						m_archive( m_parent );
						break;
					case -1:
						if( m_archive( offspring ) ) {
							m_parent = check( m_parent, offspring );
						}
						// m_archive( m_parent );
						break;
					case 1:
					case -2:
					case -3:
						break;
				}

				typename Function::SolutionSetType solutionSet;
				BOOST_FOREACH( const Individual & ind, m_archive ) {
					solutionSet.push_back( shark::makeResultSet( *ind, ind.fitness( shark::tag::UnpenalizedFitness() ) ) );
				}
				
				return( solutionSet );
			}

		protected:
			Individual check( const Individual & parent, const Individual & offspring ) {

				int lParent = m_archive.grid().location( parent );
				int lOffspring = m_archive.grid().location( offspring );

				if( lParent == -1 ) {
					return( offspring );
				}

				if( lOffspring == -1 ) {
					return( parent );
				}

				if( m_archive.grid().locationDensity( lOffspring ) < m_archive.grid().locationDensity( lParent ) ) {
					return( offspring );
				}

				return( parent );
			}
			Individual m_parent;
			Archive m_archive;

			shark::moo::PenalizingEvaluator m_evaluator;
		};
	}
}



#endif // SHARK_EA_PAES_H
