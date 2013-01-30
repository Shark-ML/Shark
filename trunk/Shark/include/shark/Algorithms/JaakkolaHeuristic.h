//===========================================================================
/*!
*  \brief Jaakkola's heuristic and related quantities for Gaussian kernel selection
*
*  \author  T. Glasmachers, O. Krause
*  \date    2010, 2012
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
//===========================================================================


#ifndef SHARK_ALGORITHMS_JAAKOLAHEURISTIC_H
#define SHARK_ALGORITHMS_JAAKOLAHEURISTIC_H


#include <shark/Data/Dataset.h>
#include <algorithm>

namespace shark{


/// \brief Jaakkola's heuristic and related quantities for Gaussian kernel selection.
///
/// \par
/// Jaakkola's heuristic method for setting the width parameter of the
/// Gaussian radial basis function kernel is to pick a quantile (usually
/// the median) of the distribution of Euclidean distances between points
/// of different label. The present implementation computes the kernel
/// width \f$ \sigma \f$ and the bandwidth
///    \f[ \gamma = \frac{1}{2 \sigma^2} \f]
/// based on the median or on any other quantile of the empirical
/// distribution.
class JaakkolaHeuristic
{
public:
	/// Constructor
	template<class InputType>
	JaakkolaHeuristic(LabeledData<InputType, unsigned int> const& dataset){
		
		//TODO: this algorithm needs n^2 space!
		typedef typename LabeledData<InputType, unsigned int>::const_element_reference reference;
		BOOST_FOREACH(reference elem1, dataset.elements())
		{
			BOOST_FOREACH(reference elem2, dataset.elements())
			{
				if (elem1.label == elem2.label) continue;
				double dist = distanceSqr(elem1.input, elem2.input);
				m_stat.push_back(dist);
			}
		}
	}

	/// Compute the given quantile (usually the 0.5-quantile)
	/// of the empirical distribution of Euclidean distances
	/// of data pairs with different labels.
	double sigma(double quantile = 0.5)
	{
		std::size_t ic = m_stat.size();
		SHARK_ASSERT(ic > 0);

		std::sort(m_stat.begin(), m_stat.end());

		if (quantile < 0.0)
		{
			// TODO: find minimum
			return std::sqrt(m_stat[0]);
		}
		if (quantile >= 1.0)
		{
			// TODO: find maximum
			return std::sqrt(m_stat[ic-1]);
		}
		else
		{
			// TODO: partial sort!
			double t = quantile * (ic - 1);
			std::size_t i = (std::size_t)floor(t);
			double rest = t - i;
			return ((1.0 - rest) * std::sqrt(m_stat[i]) + rest * std::sqrt(m_stat[i+1]));
		}
	}

	/// Compute the given quantile (usually the median)
	/// of the empirical distribution of Euclidean distances
	/// of data pairs with different labels converted into
	/// a value usable as the gamma parameter of the GaussianRbfKernel.
	double gamma(double quantile = 0.5)
	{
		double s = sigma(quantile);
		return 0.5 / (s * s);
	}

protected:
	/// all pairwise distances
	std::vector<double> m_stat;
};

}
#endif
