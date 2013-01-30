#ifndef SHARK_EA_EVALUATION_COUNT_STOPPING_CONDITION_H
#define SHARK_EA_EVALUATION_COUNT_STOPPING_CONDITION_H

namespace shark {
	/// Maximum number of evaluations stoppping condition.
	struct EvaluationCountStoppingCondition {

		EvaluationCountStoppingCondition( unsigned int maxNoEvaluations = 0 ) : m_maxNoEvaluations( maxNoEvaluations ) {}

		template<typename PropertyTree>
			void configure( const PropertyTree & pTree ) {
				m_maxNoEvaluations = pTree.get<unsigned int>( "MaxNoEvaluations " );
		}

		template<typename SolutionSet>
		bool operator()( unsigned int generationCounter, unsigned int evaluationCounter, const SolutionSet & ) const {
			return( evaluationCounter > m_maxNoEvaluations );
		}

		unsigned int m_maxNoEvaluations;
	};

}

#endif // SHARK_EA_EVALUATION_COUNT_STOPPING_CONDITION_H
