//===========================================================================
/*!
 *  \file OnePlusOneES.h
 *
 *  \brief simple 1+1-ES
 *
 *  \author Tobias Glasmachers
 *  \date 2008
 *
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


#ifndef _OnePlusOneES_H_
#define _OnePlusOneES_H_


#include <SharkDefs.h>
#include <EALib/SearchAlgorithm.h>
#include <EALib/IndividualT.h>
#include <LinAlg/VecMat.h>

//! \brief simple 1+1-ES
class OnePlusOneES : public SearchAlgorithm<double*>
{
public:
	OnePlusOneES();
	~OnePlusOneES();

	enum eStepSizeControl
	{
		SelfAdaptation,
		OneFifth,
		SymmetricOneFifth,
	};

	void init(eStepSizeControl mode, ObjectiveFunctionVS<double>& fitness);
	void init(eStepSizeControl mode, ObjectiveFunctionVS<double>& fitness, const Array<double>& start, double stepsize);
	void run();
	void bestSolutions(std::vector<double*>& points);
	void bestSolutionsFitness(Array<double>& fitness);

	inline unsigned int dimension() const { return m_parent[0].size(); }
	inline double stepsize() const { return m_parent[1][0]; }
	inline const IndividualT<double>& parent() const { return m_parent; }

protected:
	void SampleUnitVector(Vector& v);

	eStepSizeControl m_mode;
	ObjectiveFunctionVS<double>* m_fitness;
	IndividualT<double> m_parent;

	void DoSelfAdaptation();
	void DoOneFifth();
	void DoSymmetricOneFifth();

	double m_logStepSizeStdDev;
	double m_logStepSizeShift;
};


#endif
