//===========================================================================
/*!
 * 
 *
 * \brief       PenalizingEvaluator


 * 
 *
 * \author      -
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
//===========================================================================

#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_EVALUATION_PENALIZING_EVALUATOR_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_EVALUATION_PENALIZING_EVALUATOR_H

#include <shark/LinAlg/Base.h>
#include <boost/math/special_functions.hpp>
#include <boost/tuple/tuple.hpp>




namespace shark {

	namespace soo {

		/**
		* \brief Penalizing evaluator for scalar objective functions.
		*
		* Evaluates the supplied single-objective function \f$f\f$ for the search point \f$s\f$
		* according to:
		* \f{align*}{
		*   y & = & f( s' )\\
		*   y' & = & f( s' ) + \alpha \vert\vert s - s' \vert\vert_2^2
		* \f}
		* where \f$s'\f$ is the repaired version of \f$s\f$ if \f$s\f$ is not feasible and equal to \f$s\f$ otherwise.
		* The default value of \f$\alpha\f$ is \f$10^{-6}\f$.
		*/
		struct PenalizingEvaluator {

			/**
			* \brief Static constant for accessing the unpenalized fitness in a boost::tuple.
			*/
			static const unsigned int UNPENALIZED_RESULT = 0;

			/**
			* \brief Static constant for accessing the penalized fitness in a boost::tuple.
			*/
			static const unsigned int PENALIZED_RESULT = 1;

			/**
			* \brief Default c'tor, initializes the penalty factor to \f$10^{-6}\f$.
			*/
			PenalizingEvaluator() : m_penaltyFactor( 1E-6 ) {
			}

			/**
			* \brief Evaluates the supplied function on the supplied search point.
			* 
			* Evaluates the supplied single-objective function \f$f\f$ for the search point \f$s\f$
			* according to:
			* \f{align*}{
			*   y & = & f( s' )\\
			*   y' & = & f( s' ) + \alpha \vert\vert s - s' \vert\vert_2^2
			* \f}
			* where \f$s'\f$ is the repaired version of \f$s\f$ if \f$s\f$ is not feasible and equal to \f$s\f$ otherwise.
			* The default value of \f$\alpha\f$ is \f$10^{-6}\f$.

			* \tparam Function Abstracts the objective function type.
			* \tparam SearchPointType Abstract the type of the search point.
			* \param [in] f The function to be evaluated, needs to be constrained.
			* \param [in] s The search point to evaluate the function for.
			* \returns A tuple containing both the penalized and the unpenalized fitness.
			*/
			template<typename Function, typename SearchPointType>
			boost::tuple< typename Function::ResultType, typename Function::ResultType > operator()( const Function & f, const SearchPointType & s ) const {

				if( f.isFeasible( s ) ) {
					typename Function::ResultType fitness = f.eval( s );
					return( boost::make_tuple( fitness, fitness ) );
				}

				if( !f.features().test( Function::IS_CONSTRAINED_FEATURE ) ) {
					// TODO: throw Exception here
				}

				SearchPointType t( s );
				f.closestFeasible( t );

				typename Function::ResultType fitness = f.eval( t );
				typename Function::ResultType penalizedFitness( fitness );
				penalizedFitness += m_penaltyFactor * norm_sqr( t - s ); // TODO: Check

				return( boost::make_tuple( fitness, penalizedFitness ) );
			}

			/**
			* \brief Stores/loads the evaluator's state.
			* \tparam Archive The type of the archive.
			* \param [in,out] archive The archive to use for loading/storing.
			* \param [in] version Currently unused.
			*/
			template<typename Archive>
			void serialize( Archive & archive, const unsigned int version ) {
				archive & m_penaltyFactor;
			}

			double m_penaltyFactor; ///< Penalty factor \f$\alpha\f$, default value: \f$10^{-6}\f$ .

		};
	}

	namespace moo {
		/**
		* \brief Penalizing evaluator for vector-valued objective functions.
		*
		* Evaluates the supplied multi-objective function \f$\vec f\f$ for the search point \f$s\f$
		* according to:
		* \f{align*}{
		*   \vec y & = & \vec f( s' )\\
		*   \vec y' & = & \vec f( s' ) + \alpha \vert\vert s - s' \vert\vert_2^2 \vec 1.
		* \f}
		* where \f$s'\f$ is the repaired version of \f$s\f$ if \f$s\f$ is not feasible and equal to \f$s\f$ otherwise.
		* The default value of \f$\alpha\f$ is \f$10^{-6}\f$.
		*/
		struct PenalizingEvaluator {

			/**
			* \brief Static constant for accessing the unpenalized fitness in a boost::tuple.
			*/
			static const unsigned int UNPENALIZED_RESULT = 0;

			/**
			* \brief Static constant for accessing the penalized fitness in a boost::tuple.
			*/
			static const unsigned int PENALIZED_RESULT = 1;

			/**
			* \brief Default c'tor, initializes the penalty factor to \f$10^{-6}\f$.
			*/
			PenalizingEvaluator() : m_penaltyFactor( 1E-6 ) {
			}

			/**
			* \brief Evaluates the supplied function on the supplied search point.
			* 
			* Evaluates the supplied multi-objective function \f$\vec f\f$ for the search point \f$s\f$
			* according to:
			* \f{align*}{
			*   \vec y & = & \vec f( s' )\\
			*   \vec y' & = & \vec f( s' ) + \alpha \vert\vert s - s' \vert\vert_2^2 \vec 1.
			* \f}
			* where \f$s'\f$ is the repaired version of \f$s\f$ if \f$s\f$ is not feasible and equal to \f$s\f$ otherwise.
			* The default value of \f$\alpha\f$ is \f$10^{-6}\f$.

			* \tparam Function Abstracts the objective function type.
			* \tparam SearchPointType Abstract the type of the search point.
			* \param [in] f The function to be evaluated, needs to be constrained.
			* \param [in] s The search point to evaluate the function for.
			* \returns A tuple containing both the penalized and the unpenalized fitness.
			*/
			template<typename Function, typename SearchPointType>
			boost::tuple< typename Function::ResultType, typename Function::ResultType > operator()( const Function & f, const SearchPointType & s ) const {

				if( f.isFeasible( s ) ) {
					typename Function::ResultType fitness = f.eval( s );
					return( boost::make_tuple( fitness, fitness ) );
				}

				if( !f.features().test( Function::IS_CONSTRAINED_FEATURE ) ) {
					// TODO: throw Exception here
				}

				SearchPointType t( s );
				f.closestFeasible( t );

				typename Function::ResultType fitness = f.eval( t );
				typename Function::ResultType penalizedFitness( fitness );
				double penalty = norm_sqr( t - s );
				for( unsigned int i = 0; i < penalizedFitness.size(); i++ )
					penalizedFitness[i] += m_penaltyFactor * penalty; // TODO: Check

				return( boost::make_tuple( fitness, penalizedFitness ) );
			}

			/**
			* \brief Stores/loads the evaluator's state.
			* \tparam Archive The type of the archive.
			* \param [in,out] archive The archive to use for loading/storing.
			* \param [in] version Currently unused.
			*/
			template<typename Archive>
			void serialize( Archive & archive, const unsigned int version ) {
				archive & m_penaltyFactor;
			}

			double m_penaltyFactor; ///< Penalty factor \f$\alpha\f$, default value: \f$10^{-6}\f$ .

		};

	}

}


#endif
