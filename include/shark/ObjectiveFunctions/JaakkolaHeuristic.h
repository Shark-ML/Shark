//===========================================================================
/*!
*  \brief Jaakkola's heuristic and related quantities for Gaussian kernel selection
*
*  \author  T. Glasmachers, O. Krause
*  \date    2010
*
*
*  \par Copyright (c) 1999-2010:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
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


#ifndef SHARK_ML_JAAKOLAHEURISTIC_H
#define SHARK_ML_JAAKOLAHEURISTIC_H


#include <shark/Data/Dataset.h>
#include <algorithm>

namespace shark{


//! \brief Jaakkola's heuristic and related quantities for Gaussian kernel selection
class JaakkolaHeuristic
{
public:
	//! Constructor
	template<class InputType>
	JaakkolaHeuristic(LabeledData<InputType,unsigned int> const& dataset)
	{
		//typedef typename LabeledData<InputType,unsigned int>::const_element_reference iterator;
		typedef typename LabeledData<InputType,unsigned int>::const_element_range Elements;
		Elements elements = dataset.elements();
		for(typename Elements::iterator it = elements.begin(); it != elements.end(); ++it){
			//for (iterator it = dataset.elemBegin(); it != dataset.elemEnd(); it++) {
			typename Elements::iterator itIn = it;
			itIn++;
			for (; itIn != elements.end(); itIn++) {
				if (itIn->label == it->label) continue;
				double dist = shark::distance(it->input,itIn->input);
				m_stat.push_back(dist);
			}
		}
		std::sort(m_stat.begin(), m_stat.end());
	}

	//! Compute the given quantile (usually the 0.5-quantile)
	//! of the empirical distribution of euklidean distances
	//! of data pairs with different labels.
	double sigma(double quantile = 0.5)
	{
		size_t ic = m_stat.size();
		if (ic == 0)
			return 1.0;

		if (quantile < 0.0)
			return std::sqrt(m_stat[0]);
		if (quantile >= 1.0)
			return std::sqrt(m_stat[ic-1]);

		double t = quantile * (ic - 1);
		size_t i = (size_t)floor(t);
		double rest = t - i;
		return ((1.0 - rest) * std::sqrt(m_stat[i]) + rest * std::sqrt(m_stat[i+1]));
	}

	//! Compute the given quantile (usually the 0.5-quantile)
	//! of the empirical distribution of euklidean distances
	//! of data pairs with different labels converted into
	//! a value usable as the gamma parameter of the GaussianRbfKernel.
	double gamma(double quantile = 0.5)
	{
		double s = sigma(quantile);
		return 0.5 / (s * s);
	}


protected:
	std::vector<double> m_stat;
};

}
#endif
