//===========================================================================
/*!
 * 
 *
 * \brief       ROC
 * 
 * 
 *
 * \author      O.Krause
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
//===========================================================================
#include <shark/ObjectiveFunctions/ROC.h>

using namespace shark;

//! Compute the threshold for given false acceptance rate,
//! that is, for a given false positive rate.
//! This threshold, used for classification with the underlying
//! model, results in the given false acceptance rate.
double ROC::threshold(double falseAcceptanceRate)const{
	double ii = (1.0 - falseAcceptanceRate) * m_scoreNegative.size();
	int i = (unsigned int)ii;
	if (i >= (int)m_scoreNegative.size()) return 1e100;
	else if (i < 0) return -1e100;
	else if (i == ii || i == (int)m_scoreNegative.size() - 1) return m_scoreNegative[i];

	// linear interpolation
	double rest = ii - i;
	return (1.0 - rest) * m_scoreNegative[i] + rest * m_scoreNegative[i + 1];
}

//! Value of the ROC curve for given false acceptance rate,
//! that is, for a given false positive rate.
double ROC::value(double falseAcceptanceRate)const
{
	double threshold = this->threshold(falseAcceptanceRate);
	size_t i;

	// "verification rate" = 1.0 - "false rejection rate"
	for (i = 0; m_scorePositive[i] < threshold && i < m_scorePositive.size(); i++);
	if (i == 0) return 1.0;
	else if (i == m_scorePositive.size()) return 0.0;

	// linear interpolation
	double sl = m_scorePositive[i - 1];
	double sr = m_scorePositive[i];
	double inter = ((threshold - sl) * i + (sr - threshold) * (i - 1)) / (sr - sl);
	return 1.0 - inter / m_scorePositive.size();
}

//! Computes the equal error rate of the classifier
double ROC::equalErrorRate()const
{
	double threshold;
	int i, c = 0;

	double e1 = 0.0;
	double e2 = 0.0;

	double dc = m_scorePositive.size();
	double di = m_scoreNegative.size();

	for (i = 0; i < (int)m_scoreNegative.size(); i++)
	{
		threshold = m_scoreNegative[i];
		for (; m_scorePositive[c] < threshold && c < (int)m_scorePositive.size(); c++);

		e1 = i / di;			// type 1 error
		e2 = 1.0 - c / dc;		// type 2 error

		if (e1 >= e2) break;
	}
	return 0.5 *(e1 + e2);
}

