//===========================================================================
/*!
 * 
 *
 * \brief       NoisyRprop
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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

#ifndef SHARK_ML_OPTIMIZER_NOISYRPROP_H
#define SHARK_ML_OPTIMIZER_NOISYRPROP_H
#include <shark/Core/DLLSupport.h>
#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
namespace shark{

//!
//! \brief Rprop-like algorithm for noisy function evaluations
//!
//! \par
//! The Rprop algorithm (see Rprop.h) is a very robust gradient
//! descent based optimizer for real-valued optimization.
//! However, it can not deal with the presence of noice.
//!
//! \par
//! The NoisyRprop algorithm is a completely novel algorithm
//! which tries to carry over the most important ideas from
//! Rprop to the optimization of noisy problems. It is,
//! of course, slower than the Rprop algorithm, but it can
//! handle noisy problems with noisy gradient information.
//!
class NoisyRprop : public AbstractSingleObjectiveOptimizer<RealVector >
{
public:
	SHARK_EXPORT_SYMBOL NoisyRprop();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Noisy Rprop"; }

	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	//! initialization with default values
	SHARK_EXPORT_SYMBOL void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint);

	//! user defined initialization
	SHARK_EXPORT_SYMBOL void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint,double delta0P);

	//! user defined initialization with
	//! coordinate wise individual step sizes
	SHARK_EXPORT_SYMBOL void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint,const RealVector& delta0P);

	//! optimization step
	SHARK_EXPORT_SYMBOL void step(const ObjectiveFunctionType& objectiveFunction);

protected:
	//! individual step sizes
	RealVector m_delta;

	//! minimal episode length
	unsigned int m_minEpisodeLength;

	//! current episode length
	std::vector<unsigned int> m_episodeLength;

	//! fraction 1 to 100 of the episode length
	std::vector<unsigned int> m_sample;

	//! next 1/100 of the episode length
	std::vector<unsigned int> m_sampleIteration;

	//! iteration within the current episode
	std::vector<unsigned int> m_iteration;

	//! normalized position within the episode
	std::vector<unsigned int> m_position;

	//! "step to the right" event statistic
	std::vector<unsigned int> m_toTheRight;

	//! leftmost visited place
	std::vector<unsigned int> m_leftmost;

	//! rightmost visited place
	std::vector<unsigned int> m_rightmost;

	//! sum of positive derivatives
	RealVector m_leftsum;

	//! sum of negative derivatives
	RealVector m_rightsum;

	//! current error evaluation
	std::vector<unsigned int> m_current;

	//! number of error evaluations per iteration
	std::vector<unsigned int> m_numberOfAverages;

	//! accumulated gradient for each coordinate
	RealVector m_gradient;
};

}

#endif
