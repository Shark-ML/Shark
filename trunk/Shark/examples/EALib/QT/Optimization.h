//===========================================================================
/*!
 *  \file Optimization.h
 *
 *  \author  T. Glasmachers
 *  \date    2008
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-27974<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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


#ifndef _Optimization_H_
#define _Optimization_H_


#include <EALib/ObjectiveFunction.h>
#include <EALib/ObjectiveFunctions.h>
#include <EALib/MultiObjectiveFunctions.h>
#include <EALib/SearchAlgorithm.h>
#include <EALib/CMA.h>
#include <EALib/ElitistCMA.h>
#include <EALib/OnePlusOneES.h>
#include <MOO-EALib/MO-CMA.h>
#include <MOO-EALib/NSGA2.h>
#include "config.h"


// description of a property
class PropertyDesc
{
private:
	PropertyDesc(const char* name, bool scalar = false, bool logscale = false, bool observable = true);
	~PropertyDesc();

	char* m_name;
	bool m_scalar;
	bool m_logscale;
	bool m_observable;

public:
	inline const char* name() const { return m_name; }
	inline bool isScalar() const { return m_scalar; }
	inline bool isLogScale() const { return m_logscale; }
	inline bool isObservable() const { return m_observable; }

	static PropertyDesc sooFitness;
	static PropertyDesc mooFitness;
	static PropertyDesc crowdingDistance;
	static PropertyDesc epsilonIndicator;
	static PropertyDesc hypervolumeIndicator;
	static PropertyDesc population;
	static PropertyDesc position;
	static PropertyDesc covariance;
	static PropertyDesc covarianceConditioning;
};


class EncapsulatedProblem
{
public:
	EncapsulatedProblem();
	~EncapsulatedProblem();

	inline Configuration* getConfiguration() const { return m_configuration; }
	void setConfiguration(Configuration* config);
	void Init();
	inline ObjectiveFunctionVS<double>* getObjectiveFunction() const { return m_objectiveFunction; }
	bool getObservation(const char* name, Array<double>& value) const;

protected:
	ObjectiveFunctionVS<double>* m_objectiveFunction;
	Configuration* m_configuration;
};


class EncapsulatedSearchAlgorithm
{
public:
	EncapsulatedSearchAlgorithm();
	~EncapsulatedSearchAlgorithm();

	inline Configuration* getConfiguration() const { return m_configuration; }
	void setConfiguration(Configuration* config);
	void Init(EncapsulatedProblem& problem);
	inline SearchAlgorithm<double*>* getAlgorithm() const { return m_algorithm; }
	bool getObservation(PropertyDesc* property, Array<double>& value) const;

	inline unsigned int properties() const
	{ return m_property.size(); }
	inline PropertyDesc* property(unsigned int index) const
	{ return m_property[index]; }

protected:
	inline bool isCMA() const
	{
		const CMASearch* p = dynamic_cast<const CMASearch*>(m_algorithm);
		return (p != NULL);
	}
	inline const CMASearch& getCMA() const
	{
		const CMASearch* p = dynamic_cast<const CMASearch*>(m_algorithm);
		if (p == NULL) throw SHARKEXCEPTION("[EncapsulatedSearchAlgorithm::getCMA] internal error");
		return *p;
	}

	inline bool isElitistCMA() const
	{
		const CMAElitistSearch* p = dynamic_cast<const CMAElitistSearch*>(m_algorithm);
		return (p != NULL);
	}
	inline const CMAElitistSearch& getElitistCMA() const
	{
		const CMAElitistSearch* p = dynamic_cast<const CMAElitistSearch*>(m_algorithm);
		if (p == NULL) throw SHARKEXCEPTION("[EncapsulatedSearchAlgorithm::getElitistCMA] internal error");
		return *p;
	}

	inline bool isOnePlusOneES() const
	{
		const OnePlusOneES* p = dynamic_cast<const OnePlusOneES*>(m_algorithm);
		return (p != NULL);
	}
	inline const OnePlusOneES& getOnePlusOneES() const
	{
		const OnePlusOneES* p = dynamic_cast<const OnePlusOneES*>(m_algorithm);
		if (p == NULL) throw SHARKEXCEPTION("[EncapsulatedSearchAlgorithm::OnePlusOneES] internal error");
		return *p;
	}

	inline bool isMoCma() const
	{
		const MOCMASearch* p = dynamic_cast<const MOCMASearch*>(m_algorithm);
		return (p != NULL);
	}
	inline const MOCMASearch& getMoCma() const
	{
		const MOCMASearch* p = dynamic_cast<const MOCMASearch*>(m_algorithm);
		if (p == NULL) throw SHARKEXCEPTION("[EncapsulatedSearchAlgorithm::getMoCma] internal error");
		return *p;
	}

	inline bool isNSGA2() const
	{
		const NSGA2Search* p = dynamic_cast<const NSGA2Search*>(m_algorithm);
		return (p != NULL);
	}
	inline const NSGA2Search& getNSGA2() const
	{
		const NSGA2Search* p = dynamic_cast<const NSGA2Search*>(m_algorithm);
		if (p == NULL) throw SHARKEXCEPTION("[EncapsulatedSearchAlgorithm::getNSGA2] internal error");
		return *p;
	}

	SearchAlgorithm<double*>* m_algorithm;
	Configuration* m_configuration;
	std::vector<PropertyDesc*> m_property;
	EncapsulatedProblem* m_problem;
};


#endif
