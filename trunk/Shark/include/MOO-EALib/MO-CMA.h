//===========================================================================
/*!
 *  \file MO-CMA.h
 *
 *  \brief CMA-ES for multi-objective optimization
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
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
//===========================================================================


#ifndef _MO_CMA_H_
#define _MO_CMA_H_


#include <SharkDefs.h>
#include <MOO-EALib/PopulationMOO.h>
#include <EALib/SearchAlgorithm.h>
#include <EALib/ChromosomeCMA.h>

//! \brief CMA-ES for multi-objective optimization
class MOCMASearch : public EvolutionaryAlgorithm<double*>
{
public:
	MOCMASearch();
	~MOCMASearch();

	void init(ObjectiveFunctionVS<double>& fitness, unsigned int mu, unsigned int lambda);
	void init(ObjectiveFunctionVS<double>& fitness, double stepsize, unsigned int mu, unsigned int lambda);
	void run();
	void bestSolutions(std::vector<double*>& points);
	void bestSolutionsFitness(Array<double>& fitness);

	inline double penaltyFactor() const { return m_penaltyFactor; }
	inline void setPenaltyFactor(double factor) { m_penaltyFactor = factor; }

	void parents(PopulationMOO& parents) const;

protected:
	void eval(IndividualMOO& ind);

	ObjectiveFunctionVS<double>* m_fitness;
	unsigned int m_objectives;
	PopulationMOO* m_pop;
	double m_penaltyFactor;
};


#endif
