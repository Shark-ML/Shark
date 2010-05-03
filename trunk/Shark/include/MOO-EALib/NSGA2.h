/*!
 *  \file NSGA2.cpp
 *
 *  \author T. Glasmachers
 *
 *  \brief NSGA-2 algorithm for multi-objective optimization
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


#ifndef _NSGA2_H_
#define _NSGA2_H_


#include <SharkDefs.h>
#include <MOO-EALib/PopulationMOO.h>
#include <EALib/SearchAlgorithm.h>
#include <EALib/ChromosomeCMA.h>

//! \brief NSGA-2 algorithm for multi-objective optimization
class NSGA2Search : public EvolutionaryAlgorithm<double*>
{
public:
	NSGA2Search();
	~NSGA2Search();

	void init(ObjectiveFunctionVS<double>& fitness, unsigned int mu, double nm = 20.0, double nc = 20.0, double pc = 0.9);
	void run();
	void bestSolutions(std::vector<double*>& points);
	void bestSolutionsFitness(Array<double>& fitness);

	inline double penaltyFactor() const { return m_penaltyFactor; }
	inline void setPenaltyFactor(double factor) { m_penaltyFactor = factor; }

	inline PopulationMOO& parents() { return *m_parents; }
	inline const PopulationMOO& parents() const { return *m_parents; }

protected:
	void eval(IndividualMOO& ind);

	unsigned int m_objectives;
	double m_nm;
	double m_nc;
	double m_pc;
	double m_pm;

	ObjectiveFunctionVS<double>* m_fitness;
	PopulationMOO* m_parents;
	PopulationMOO* m_offspring;
	std::vector<double> m_lower;
	std::vector<double> m_upper;
	double m_penaltyFactor;
};


#endif
