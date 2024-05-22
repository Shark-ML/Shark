/*!
 * 
 *
 * \brief       Implements the generational Multi-objective Covariance Matrix Adapation ES.
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <https://shark-ml.github.io/Shark/>
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_MOCMA
#define SHARK_ALGORITHMS_DIRECT_SEARCH_MOCMA

// MOO specific stuff
#include <shark/Algorithms/DirectSearch/Operators/Indicators/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Indicators/AdditiveEpsilonIndicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/CMA/CMAIndividual.h>


#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>

namespace shark {

/// \brief Implements the generational MO-CMA-ES.
///
/// Please see the following papers for further reference:
///	- Igel, Suttorp and Hansen. Steady-state Selection and Efficient Covariance Matrix Update in the Multi-Objective CMA-ES.
