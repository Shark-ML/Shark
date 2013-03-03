/**
 *
 * \brief Bounding box calculator.
 *
 *  \author T.Voss, T. Glasmachers, O.Krause
 *  \date 2010-2011
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
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
