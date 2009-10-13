//===========================================================================
/*!
 *  \file Optimization.cpp
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


#include <MOO-EALib/Hypervolume.h>
#include "Optimization.h"
#include "Canyon.h"


PropertyDesc::PropertyDesc(const char* name, bool scalar, bool logscale, bool observable)
{
	m_name = (char*)(void*)malloc(strlen(name) + 1);
	strcpy(m_name, name);

	m_scalar = scalar;
	m_logscale = logscale;
	m_observable = observable;
}

PropertyDesc::~PropertyDesc()
{
	free(m_name);
}


// static objects
PropertyDesc PropertyDesc::sooFitness("fitness", true, true);
PropertyDesc PropertyDesc::mooFitness("fitness vector", false, true, false);
PropertyDesc PropertyDesc::crowdingDistance("crowding distance", true, true);
PropertyDesc PropertyDesc::epsilonIndicator("epsilon indicator", true, true);
PropertyDesc PropertyDesc::hypervolumeIndicator("hypervolume indicator", true, true);
PropertyDesc PropertyDesc::population("population", false, false, false);
PropertyDesc PropertyDesc::position("position", false, false, false);
PropertyDesc PropertyDesc::covariance("covariance matrix", false, false, false);
PropertyDesc PropertyDesc::covarianceConditioning("conditioning of the covariance matrix", true, true);


////////////////////////////////////////////////////////////


EncapsulatedProblem::EncapsulatedProblem()
{
	m_objectiveFunction = NULL;
	m_configuration = NULL;
}

EncapsulatedProblem::~EncapsulatedProblem()
{
	if (m_objectiveFunction != NULL) delete m_objectiveFunction;
// 	if (m_configuration != NULL) delete m_configuration;
}


void EncapsulatedProblem::setConfiguration(Configuration* config)
{
// 	if (m_configuration != NULL) delete m_configuration;
	m_configuration = config;
}

void EncapsulatedProblem::Init()
{
	if (m_objectiveFunction != NULL)
	{
		delete m_objectiveFunction;
		m_objectiveFunction = NULL;
	}

	const PropertyNode& node = (*m_configuration)["objective-function"];
	const PropertyNode& f = node.getSelectedNode();

	// SINGLE OBJECTIVE FUNCTIONS
	if (strcmp(node.getSelected(), "sphere") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new Sphere(dim);
	}
	else if (strcmp(node.getSelected(), "paraboloid") == 0)
	{
		int dim = f["dimension"].getInt();
		double cond = f["conditioning"].getDouble();
		m_objectiveFunction = new Paraboloid(dim, cond);
	}
	else if (strcmp(node.getSelected(), "tablet") == 0)
	{
		int dim = f["dimension"].getInt();
		double cond = f["conditioning"].getDouble();
		m_objectiveFunction = new Tablet(dim, cond);
	}
	else if (strcmp(node.getSelected(), "cigar") == 0)
	{
		int dim = f["dimension"].getInt();
		double cond = f["conditioning"].getDouble();
		m_objectiveFunction = new Cigar(dim, cond);
	}
	else if (strcmp(node.getSelected(), "twoaxis") == 0)
	{
		int dim = f["dimension"].getInt();
		double cond = f["conditioning"].getDouble();
		m_objectiveFunction = new Twoaxis(dim, cond);
	}
	else if (strcmp(node.getSelected(), "Ackley") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new Ackley( dim );
	}
	else if (strcmp(node.getSelected(), "rastrigin") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new Rastrigin( dim );
	}
	else if (strcmp(node.getSelected(), "Griewangk") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new Griewangk( dim );
	}
	else if (strcmp(node.getSelected(), "Rosenbrock") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new Rosenbrock( dim );
	}
	else if (strcmp(node.getSelected(), "Rosenbrock-rotated") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new RosenbrockRotated( dim );
	}
	else if (strcmp(node.getSelected(), "diff-pow") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new DiffPow(dim);
	}
	else if (strcmp(node.getSelected(), "diff-pow-rotated") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new DiffPowRotated(dim);
	}
	else if (strcmp(node.getSelected(), "Schwefel") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new Schwefel( dim );
	}
	else if (strcmp(node.getSelected(), "Schwefel-ellipsoid") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new SchwefelEllipsoid(dim);
	}
	else if (strcmp(node.getSelected(), "Schwefel-ellipsoid-rotated") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new SchwefelEllipsoidRotated(dim);
	}
	else if (strcmp(node.getSelected(), "random") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new RandomFitness(dim);
	}

	// SPECIAL CANYON FUNCTION
	else if (strcmp(node.getSelected(), "canyon") == 0)
	{
		m_objectiveFunction = new Canyon();
	}

	// MULTI OBJECTIVE FUNCTIONS
	else if (strcmp(node.getSelected(), "BhinKorn") == 0)
	{
		m_objectiveFunction = new BhinKorn();
	}
	else if (strcmp(node.getSelected(), "Schaffer") == 0)
	{
		m_objectiveFunction = new Schaffer();
	}
	else if (strcmp(node.getSelected(), "ZDT-1") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZDT1(dim);
	}
	else if (strcmp(node.getSelected(), "ZDT-2") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZDT2(dim);
	}
	else if (strcmp(node.getSelected(), "ZDT-3") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZDT3(dim);
	}
	else if (strcmp(node.getSelected(), "ZDT-4") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZDT4(dim);
	}
// 	else if (strcmp(node.getSelected(), "ZDT-5") == 0)
// 	{
// 		int dim = f["dimension"].getInt();
// 		m_objectiveFunction = new ZDT5(dim);
// 	}
	else if (strcmp(node.getSelected(), "ZDT-6") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZDT6(dim);
	}
	else if (strcmp(node.getSelected(), "IHR-1") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new IHR1(dim);
	}
	else if (strcmp(node.getSelected(), "IHR-2") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new IHR2(dim);
	}
	else if (strcmp(node.getSelected(), "IHR-3") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new IHR3(dim);
	}
	else if (strcmp(node.getSelected(), "IHR-6") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new IHR6(dim);
	}
	else if (strcmp(node.getSelected(), "ZZJ07-F1") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZZJ07_F1(dim);
	}
	else if (strcmp(node.getSelected(), "ZZJ07-F2") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZZJ07_F2(dim);
	}
	else if (strcmp(node.getSelected(), "ZZJ07-F3") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZZJ07_F3(dim);
	}
	else if (strcmp(node.getSelected(), "ZZJ07-F4") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZZJ07_F4(dim);
	}
	else if (strcmp(node.getSelected(), "ZZJ07-F5") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZZJ07_F5(dim);
	}
	else if (strcmp(node.getSelected(), "ZZJ07-F6") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZZJ07_F6(dim);
	}
	else if (strcmp(node.getSelected(), "ZZJ07-F7") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZZJ07_F7(dim);
	}
	else if (strcmp(node.getSelected(), "ZZJ07-F8") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZZJ07_F8(dim);
	}
	else if (strcmp(node.getSelected(), "ZZJ07-F9") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZZJ07_F9(dim);
	}
	else if (strcmp(node.getSelected(), "ZZJ07-F10") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new ZZJ07_F10(dim);
	}
	else if (strcmp(node.getSelected(), "LZ06-F1") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new LZ06_F1(dim);
	}
	else if (strcmp(node.getSelected(), "LZ06-F2") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new LZ06_F2(dim);
	}
	else if (strcmp(node.getSelected(), "LZ07-F1") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new LZ07_F1(dim);
	}
	else if (strcmp(node.getSelected(), "LZ07-F2") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new LZ07_F2(dim);
	}
	else if (strcmp(node.getSelected(), "LZ07-F3") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new LZ07_F3(dim);
	}
	else if (strcmp(node.getSelected(), "LZ07-F4") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new LZ07_F4(dim);
	}
	else if (strcmp(node.getSelected(), "LZ07-F5") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new LZ07_F5(dim);
	}
	else if (strcmp(node.getSelected(), "LZ07-F6") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new LZ07_F6(dim);
	}
	else if (strcmp(node.getSelected(), "LZ07-F7") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new LZ07_F7(dim);
	}
	else if (strcmp(node.getSelected(), "LZ07-F8") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new LZ07_F8(dim);
	}
	else if (strcmp(node.getSelected(), "LZ07-F9") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new LZ07_F9(dim);
	}
	else if (strcmp(node.getSelected(), "ELLI-1") == 0)
	{
		int dim = f["dimension"].getInt();
		double cond = f["conditioning"].getDouble();
		m_objectiveFunction = new ELLI1(dim, cond);
	}
	else if (strcmp(node.getSelected(), "ELLI-2") == 0)
	{
		int dim = f["dimension"].getInt();
		double cond = f["conditioning"].getDouble();
		m_objectiveFunction = new ELLI2(dim, cond);
	}
	else if (strcmp(node.getSelected(), "CIGTAB-1") == 0)
	{
		int dim = f["dimension"].getInt();
		double cond = f["conditioning"].getDouble();
		m_objectiveFunction = new CIGTAB1(dim, cond);
	}
	else if (strcmp(node.getSelected(), "CIGTAB-2") == 0)
	{
		int dim = f["dimension"].getInt();
		double cond = f["conditioning"].getDouble();
		m_objectiveFunction = new CIGTAB2(dim, cond);
	}
	else if (strcmp(node.getSelected(), "DTLZ-1") == 0)
	{
		int dim = f["dimension"].getInt();
		int obj = f["objectives"].getInt();
		m_objectiveFunction = new DTLZ1(dim, obj);
	}
	else if (strcmp(node.getSelected(), "DTLZ-2") == 0)
	{
		int dim = f["dimension"].getInt();
		int obj = f["objectives"].getInt();
		m_objectiveFunction = new DTLZ2(dim, obj);
	}
	else if (strcmp(node.getSelected(), "DTLZ-3") == 0)
	{
		int dim = f["dimension"].getInt();
		int obj = f["objectives"].getInt();
		m_objectiveFunction = new DTLZ3(dim, obj);
	}
	else if (strcmp(node.getSelected(), "DTLZ-4") == 0)
	{
		int dim = f["dimension"].getInt();
		int obj = f["objectives"].getInt();
		m_objectiveFunction = new DTLZ4(dim, obj);
	}
	else if (strcmp(node.getSelected(), "DTLZ-5") == 0)
	{
		int dim = f["dimension"].getInt();
		int obj = f["objectives"].getInt();
		m_objectiveFunction = new DTLZ5(dim, obj);
	}
	else if (strcmp(node.getSelected(), "DTLZ-6") == 0)
	{
		int dim = f["dimension"].getInt();
		int obj = f["objectives"].getInt();
		m_objectiveFunction = new DTLZ6(dim, obj);
	}
	else if (strcmp(node.getSelected(), "DTLZ-7") == 0)
	{
		int dim = f["dimension"].getInt();
		int obj = f["objectives"].getInt();
		m_objectiveFunction = new DTLZ7(dim, obj);
	}
	else if (strcmp(node.getSelected(), "Superspheres") == 0)
	{
		int dim = f["dimension"].getInt();
		m_objectiveFunction = new Superspheres(dim);
	}
	else throw SHARKEXCEPTION("[EncapsulatedProblem::Init] unknown objective function");
}

bool EncapsulatedProblem::getObservation(const char* name, Array<double>& value) const
{
	return false;
}


////////////////////////////////////////////////////////////


EncapsulatedSearchAlgorithm::EncapsulatedSearchAlgorithm()
{
	m_algorithm = NULL;
	m_configuration = NULL;
}

EncapsulatedSearchAlgorithm::~EncapsulatedSearchAlgorithm()
{
	if (m_algorithm != NULL) delete m_algorithm;
// 	if (m_configuration != NULL) delete m_configuration;
}


void EncapsulatedSearchAlgorithm::setConfiguration(Configuration* config)
{
// 	if (m_configuration != NULL) delete m_configuration;
	m_configuration = config;
}

void EncapsulatedSearchAlgorithm::Init(EncapsulatedProblem& problem)
{
	if (m_algorithm != NULL)
	{
		delete m_algorithm;
		m_algorithm = NULL;
	}

	m_property.clear();

	const PropertyNode& node = (*m_configuration)["search-strategy"];
	const PropertyNode& algo = node.getSelectedNode();
	if (strcmp(node.getSelected(), "CMA-ES") == 0)
	{
		CMA::RecombType recomb = CMA::superlinear;
		CMA::UpdateType update = CMA::rankmu;

		const char* rec = algo["recombination-type"].getSelected();
		if (strcmp(rec, "equal") == 0) recomb = CMA::equal;
		else if (strcmp(rec, "linear") == 0) recomb = CMA::linear;
		else if (strcmp(rec, "superlinear") == 0) recomb = CMA::superlinear;

		const char* upd = algo["update-type"].getSelected();
		if (strcmp(upd, "rank-1") == 0) update = CMA::rankone;
		else if (strcmp(upd, "rank-mu") == 0) update = CMA::rankmu;

		CMASearch* search = new CMASearch();
		m_algorithm = search;
		search->init(*problem.getObjectiveFunction(), recomb, update);

		m_property.push_back(&PropertyDesc::sooFitness);
		m_property.push_back(&PropertyDesc::population);
		m_property.push_back(&PropertyDesc::position);
		m_property.push_back(&PropertyDesc::covariance);
		m_property.push_back(&PropertyDesc::covarianceConditioning);
	}
	else if (strcmp(node.getSelected(), "1+1-CMA-ES") == 0)
	{
		CMAElitistSearch* search = new CMAElitistSearch();
		m_algorithm = search;
		search->init(*problem.getObjectiveFunction());

		m_property.push_back(&PropertyDesc::sooFitness);
		m_property.push_back(&PropertyDesc::population);
		m_property.push_back(&PropertyDesc::position);
		m_property.push_back(&PropertyDesc::covariance);
// 		m_property.push_back(&PropertyDesc::covarianceConditioning);
	}
	else if (strcmp(node.getSelected(), "1+1-ES") == 0)
	{
		OnePlusOneES* search = new OnePlusOneES();
		m_algorithm = search;

		OnePlusOneES::eStepSizeControl mode = OnePlusOneES::SelfAdaptation;
		const char* ssc = algo["step-size-control"].getSelected();
		if (strcmp(ssc, "self-adaptation") == 0) mode = OnePlusOneES::SelfAdaptation;
		else if (strcmp(ssc, "one-fifth-rule") == 0) mode = OnePlusOneES::OneFifth;

		search->init(mode, *problem.getObjectiveFunction());

		m_property.push_back(&PropertyDesc::sooFitness);
		m_property.push_back(&PropertyDesc::population);
		m_property.push_back(&PropertyDesc::position);
		m_property.push_back(&PropertyDesc::covariance);
	}
	else if (strcmp(node.getSelected(), "NSGA-2") == 0)
	{
		NSGA2Search* search = new NSGA2Search();
		m_algorithm = search;
		search->init(*problem.getObjectiveFunction(),
				algo["mu"].getInt(),
				algo["nm"].getDouble(),
				algo["nc"].getDouble(),
				algo["pc"].getDouble()
			);

		m_property.push_back(&PropertyDesc::mooFitness);
		m_property.push_back(&PropertyDesc::population);
		m_property.push_back(&PropertyDesc::hypervolumeIndicator);
	}
	else if (strcmp(node.getSelected(), "MO-CMA-ES") == 0)
	{
		MOCMASearch* search = new MOCMASearch();
		m_algorithm = search;
		search->init(*problem.getObjectiveFunction(), algo["mu"].getInt(), algo["lambda"].getInt());

		m_property.push_back(&PropertyDesc::mooFitness);
		m_property.push_back(&PropertyDesc::population);
		m_property.push_back(&PropertyDesc::hypervolumeIndicator);
	}
	else throw SHARKEXCEPTION("[EncapsulatedSearchAlgorithm::Init] unknown algorithm");

	m_problem = &problem;
}

bool EncapsulatedSearchAlgorithm::getObservation(PropertyDesc* property, Array<double>& value) const
{
	if (property == &PropertyDesc::sooFitness)
	{
		value.resize(1, false);
		value(0) = m_algorithm->bestSolutionFitness();
		return true;
	}
	else if (property == &PropertyDesc::mooFitness)
	{
		m_algorithm->bestSolutionsFitness(value);
		return true;
	}
	else if (property == &PropertyDesc::covarianceConditioning)
	{
		if (isCMA())
		{
			Array<double> lambda = getCMA().getCMA().getLambda();
			value.resize(1, false);
			value(0) = lambda(0) / lambda(lambda.dim(0) - 1);
			return true;
		}
		else if (isElitistCMA())
		{
			// TODO
		}
	}
	else if (property == &PropertyDesc::population)
	{
		if (isCMA())
		{
			const PopulationT<double>& pop = *getCMA().parents();
			int i, ic = pop.size();
			int d, dim = pop[0][0].size();
			value.resize(ic, dim, false);
			for (i=0; i<ic; i++)
			{
				for (d=0; d<dim; d++)
				{
					value(i, d) = pop[i][0][d];
				}
			}
			return true;
		}
		else if (isElitistCMA())
		{
			const PopulationCT<ChromosomeCMA>& pop = *getElitistCMA().parents();
			int d, dim = pop[0][0].size();
			value.resize(1, dim, false);
			for (d=0; d<dim; d++) value(0, d) = pop[0][0][d];
			return true;
		}
		else if (isOnePlusOneES())
		{
			const IndividualT<double>& p = getOnePlusOneES().parent();
			int d, dim = p[0].size();
			value.resize(1, dim, false);
			for (d=0; d<dim; d++) value(0, d) = p[0][d];
			return true;
		}
		else if (isMoCma())
		{
			PopulationMOO pop;
			getMoCma().parents(pop);
			int i, ic = pop.size();
			int d, dim = pop[0][0].size();
			value.resize(ic, dim, false);
			for (i=0; i<ic; i++)
			{
				for (d=0; d<dim; d++)
				{
					value(i, d) = (static_cast<ChromosomeT<double>&>(pop[i][0]))[d];
				}
			}
			return true;
		}
		else if (isNSGA2())
		{
			const PopulationMOO& pop = getNSGA2().parents();
			int i, ic = pop.size();
			int d, dim = pop[0][0].size();
			value.resize(ic, dim, false);
			for (i=0; i<ic; i++)
			{
				for (d=0; d<dim; d++)
				{
					value(i, d) = (static_cast<const ChromosomeT<double>&>(pop[i][0]))[d];
				}
			}
			return true;
		}
	}
	else if (property == &PropertyDesc::position)
	{
		if (isCMA())
		{
			const PopulationT<double>& pop = *getCMA().parents();
			int i, ic = pop.size();
			int d, dim = pop[0][0].size();
			value.resize(dim, false);
			for (d=0; d<dim; d++)
			{
				double mean = 0.0;
				for (i=0; i<ic; i++) mean += pop[i][0][d];
				mean /= ic;
				value(d) = mean;
			}
			return true;
		}
		else if (isElitistCMA())
		{
			const PopulationCT<ChromosomeCMA>& pop = *getElitistCMA().parents();
			int d, dim = pop[0][0].size();
			value.resize(dim, false);
			for (d=0; d<dim; d++) value(d) = pop[0][0][d];
			return true;
		}
		else if (isOnePlusOneES())
		{
			const IndividualT<double>& p = getOnePlusOneES().parent();
			int d, dim = p[0].size();
			value.resize(dim, false);
			for (d=0; d<dim; d++) value(d) = p[0][d];
			return true;
		}
	}
	else if (property == &PropertyDesc::covariance)
	{
		if (isCMA())
		{
			double sigma = getCMA().getCMA().getSigma();
			value = (sigma * sigma) * getCMA().getCMA().getC();
			return true;
		}
		else if (isElitistCMA())
		{
			const PopulationCT<ChromosomeCMA>& pop = *getElitistCMA().parents();
			const IndividualCT<ChromosomeCMA>& ind = pop[0];
			const ChromosomeCMA& chrom = ind[0];
			double sigma = chrom.getSigma();
			value = (sigma * sigma) * chrom.getC();
			return true;
		}
		else if (isOnePlusOneES())
		{
			double stepsize = getOnePlusOneES().stepsize();
			int d, dim = getOnePlusOneES().dimension();
			value.resize(dim, dim, false);
			value = 0.0;
			for (d=0; d<dim; d++) value(d, d) = stepsize;
			return true;
		}
	}
// 	else if (property == &PropertyDesc::crowdingDistance)
// 	{
// 	}
	else if (property == &PropertyDesc::epsilonIndicator)
	{
	}
	else if (property == &PropertyDesc::hypervolumeIndicator)
	{
		value.resize(1, false);
		value(0) = 0.0;
		std::vector<double> referencePoint;
		ObjectiveFunctionVS<double>* of = m_problem->getObjectiveFunction();
		if (! of->nadirFitness(referencePoint)) return false;

		Array<double> fit;
		m_algorithm->bestSolutionsFitness(fit);
		if (fit.ndim() != 2 || fit.nelem() == 0) return false;

		value(0) = hypervolume(&fit(0, 0), &referencePoint[0], fit.dim(1), fit.dim(0));
		return true;
	}

	return false;
}
