/*!
 * 
 *
 * \brief Implements the frontend for the HypervolumeCalculator algorithms, including the approximations
 *
 *
 * \author     O.Krause
 * \date        2014-2016
 *
 *
 * \par Copyright 1995-2016 Shark Development Team
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMECALCULATOR_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMECALCULATOR_H

#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator2D.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator3D.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculatorMD.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeApproximator.h>

namespace shark {
/// \brief Frontend for hypervolume calculation algorithms in m dimensions.
///
///  Depending on the dimensionality of the problem, one of the specialized algorithms is called.
///  For large dimensionalities for which there are no specialized fast algorithms,
///  either the exponential time or the approximated algorithm is called based on the choice of algorithm
struct HypervolumeCalculator {

	/// \brief Default c'tor.
	HypervolumeCalculator() : m_useApproximation(false) {}
	
	///\brief True if the hypervolume approximation is to be used in dimensions > 3.
	void useApproximation(bool useApproximation){
		m_useApproximation = useApproximation;
	}
	
	double approximationEpsilon()const{
		return m_approximationAlgorithm.epsilon();
	}
	double& approximationEpsilon(){
		return m_approximationAlgorithm.epsilon();
	}
	
	double approximationDelta()const{
		return m_approximationAlgorithm.delta();
	}
	
	double& approximationDelta(){
		return m_approximationAlgorithm.delta();
	}
	
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
		archive & BOOST_SERIALIZATION_NVP(m_useApproximation);
		archive & BOOST_SERIALIZATION_NVP(m_approximationAlgorithm);
	}
	
	/// \brief Executes the algorithm.
	/// \param [in] points The set \f$S\f$ of points for which the following assumption needs to hold: \f$\forall s \in S: \lnot \exists s' \in S: s' \preceq s \f$
	/// \param [in] refPoint The reference point \f$\vec{r} \in \mathbb{R}^n\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$. .
	template<typename Points, typename VectorType>
	double operator()( Points const& points, VectorType const& refPoint){
		SIZE_CHECK( points.begin()->size() == refPoint.size() );
		std::size_t numObjectives = refPoint.size();
		if(numObjectives == 2){
			HypervolumeCalculator2D algorithm;
			return algorithm(points, refPoint);
		}else if(numObjectives == 3){
			HypervolumeCalculator3D algorithm;
			return algorithm(points, refPoint);
		}else if(m_useApproximation){
			return m_approximationAlgorithm(points, refPoint);
		}else{
			HypervolumeCalculatorMD algorithm;
			return algorithm(points, refPoint);
		}
	}

private:
	bool m_useApproximation;
	HypervolumeApproximator m_approximationAlgorithm;
};

}
#endif