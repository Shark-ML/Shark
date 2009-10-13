//======================================================================
/*!
 *
 *  \file SelectionMOO-Impl.h
 *
 *  \brief Implementation of indicator-based, multi-objective selection
 * 
 *  \author Thomas Vossü &lt;thomas.voss@rub.de&gt;
 *
 *  \par
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
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */


#ifndef _SELECTIONMOO_IMPL_H_
#define _SELECTIONMOO_IMPL_H_

#include <MOO-EALib/SelectionMOO.h>
#include <MOO-EALib/MOOMeasures.h>
#include <MOO-EALib/PopulationMOO.h>


template<typename Indicator_T>
void IndicatorBasedSelectionStrategy<Indicator_T>::operator()(PopulationMOO & pop)
{
	unsigned m_maxMOORank = pop.maxMOORank();
	unsigned noObjectives = 0;
	unsigned i, j;

	for (i = 0; i < pop.size(); i++)
	{
		noObjectives = Shark::max(noObjectives, static_cast<unsigned>(pop[i].getMOOFitnessValues(m_bUnpenalizedFitness).size()));		// required for Macintosh compiler
	}

	ObjectiveSort objSort(m_bUnpenalizedFitness, noObjectives);
	std::sort(pop.begin(), pop.end(), objSort);

	std::vector<unsigned> p(0); p.reserve(pop.size());

	for (i = 1; i <= m_maxMOORank; i++)
	{
		p.clear();

		for (j = 0; j < pop.size(); j++)
		{
			if (pop[j].getMOORank() == i)
			{
				p.push_back(j);
			}
		}

		if (p.size() <= 1)
			continue;

		unsigned pSize = p.size();

		std::vector<std::vector<double> > a(p.size());

		for (j = 0; j < a.size(); j++)
		{
			a[j] = pop[p[j]].getMOOFitnessValues(m_bUnpenalizedFitness);
		}

		MinMax minMax = calc_boundary_elements(pop,
											   p,
											   noObjectives,
											   m_bUnpenalizedFitness
											  );
		for (j = 0; j < noObjectives; j++)
		{
			if (m_bAscending)
				pop[p[minMax.first[j]]].setMOOShare(std::numeric_limits<double>::max());
			else
				pop[p[minMax.second[j]]].setMOOShare(std::numeric_limits<double>::max());

		}

		while (p.size() > 1)
		{
			std::vector<double> indicatorValues(p.size(), std::numeric_limits<double>::max());
			for (j = 0; j < a.size(); j++)
			{
				if (pop[p[j]].getMOOShare() == std::numeric_limits<double>::max())
					continue;

				indicatorValues[j] = m_binaryQualityIndicator(a, j);
			}

			std::vector<double>::iterator min = std::min_element(indicatorValues.begin(), indicatorValues.end());

			if (*min == std::numeric_limits<double>::max())
				break;

			pop[p[std::distance(indicatorValues.begin(), min)]].setMOOShare(pSize - p.size());
			p.erase(p.begin() + std::distance(indicatorValues.begin(), min));
			a.erase(a.begin() + std::distance(indicatorValues.begin(), min));
		}

		for (j = 0; j < p.size(); j++)
			pop[p[j]].setMOOShare(pSize);

	}
}


#endif

