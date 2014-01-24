/*!
 * 
 * \file        BoundingBoxCalculator.h
 *
 * \brief       Bounding box calculator.
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_BOUNDING_BOX_CALCULATOR_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_BOUNDING_BOX_CALCULATOR_H

#include <shark/LinAlg/Base.h>

namespace shark {

/**
 * \brief Calculates the bounding of a d-dimensional point set.
 */
template<typename ExtractorType,typename VectorType = shark::RealVector>
struct BoundingBoxCalculator {

	BoundingBoxCalculator(ExtractorType &extractor, VectorType &lowerBound, VectorType &upperBound) : m_extractor(extractor),
		m_lowerBound(lowerBound),
		m_upperBound(upperBound) {
	}

	template<typename Member>
	void operator()(const Member &m) {
		for (unsigned int i = 0; i < m_upperBound.size(); i++) {
			m_lowerBound[i] = std::min(m_lowerBound[i], m_extractor(m)[i]);
			m_upperBound[i] = std::max(m_upperBound[i], m_extractor(m)[i]);
		}
	}

	ExtractorType m_extractor;
	VectorType &m_lowerBound;
	VectorType &m_upperBound;
};
}
#endif // SHARK_ALGORITHMS_DIRECTSEARCH_BOUNDING_BOX_CALCULATOR_H
