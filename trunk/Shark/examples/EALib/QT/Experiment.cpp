//===========================================================================
/*!
 *  \file Experiment.cpp
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


#include <math.h>

#include <QFile>
#include <QTime>

#include "Experiment.h"
#include "Optimization.h"
#include "Wizard.h"
#include "View.h"
#include "ControlWidget.h"


TrialHistory::TrialHistory(Experiment* experiment)
{
	m_experiment = experiment;

	int i, ic = m_experiment->m_recording.size();
	for (i=0; i<ic; i++)
	{
		m_property.push_back(new PropertyHistory(m_experiment->m_recording[i]));
	}
}

TrialHistory::~TrialHistory()
{
	int i, ic = m_property.size();
	for (i=0; i<ic; i++) delete m_property[i];
}


void TrialHistory::OnChanged(int evals)
{
	int i, ic = m_property.size();
	for (i=0; i<ic; i++) m_property[i]->OnChanged(m_experiment->m_algorithm, evals);
}

int TrialHistory::getObservationIndex(PropertyDesc* property) const
{
	int i, ic = m_property.size();
	for (i=0; i<ic; i++) if (m_property[i]->m_property == property) return i;
	return -1;
}


////////////////////////////////////////////////////////////


TrialHistory::PropertyHistory::PropertyHistory(PropertyDesc* property)
{
	m_property = property;
}

TrialHistory::PropertyHistory::~PropertyHistory()
{
}


void TrialHistory::PropertyHistory::OnChanged(EncapsulatedSearchAlgorithm& algo, int evals)
{
	Array<double> data;
	algo.getObservation(m_property, data);
	m_evals.append_elem(evals);
	m_values.append_rows(data);
}

void TrialHistory::PropertyHistory::observation(double evals, Array<double>& values) const
{
	int ec = m_evals.dim(0);
	if (ec == 0) return;

	if (m_evals(0) >= evals) values = m_values[0];
	else if (m_evals(ec -1) <= evals) values = m_values[ec - 1];
	else
	{
		// binary search: find the largest e with m_evals[e] <= evals
		unsigned int stepsize = 0;
		unsigned int b = 16;
		for (b = 16; b > 0; b >>= 1)
		{
			if (ec >= (1 << (stepsize + b))) stepsize += b;
		}
		stepsize = (1 << stepsize);
		int i, e = 0;
		while (stepsize > 0)
		{
			i = e + stepsize;
			if (i < ec && m_evals(i) < evals) e = i;
			stepsize >>= 1;
		}

		// interpolate
		int before = m_evals(e);
		int after = m_evals(e+1);
		double delta = after - before;
		double pos = evals - before;
		double f2 = pos / delta;
		double f1 = 1.0 - f2;
		if (pos < 0.0) pos = 0.0;

		if (m_property->isLogScale())
		{
			values = exp(f1 * log(m_values[e]) + f2 * log(m_values[e + 1]));
		}
		else
		{
			values = f1 * m_values[e] + f2 * m_values[e + 1];
		}
	}
}


////////////////////////////////////////////////////////////


ExperimentThread::ExperimentThread(Experiment* experiment, ManualControlWidget* controlWidget)
{
	m_experiment = experiment;
	m_controlWidget = controlWidget;
	m_numberOfViews = 0;
}

ExperimentThread::~ExperimentThread()
{
}


void ExperimentThread::registerView(View* view)
{
	connect(this, SIGNAL(reset()), view, SLOT(reset()), Qt::QueuedConnection);
	connect(this, SIGNAL(changed(int)), view, SLOT(changed(int)), Qt::QueuedConnection);
	connect(this, SIGNAL(destroy()), view, SLOT(destroy()), Qt::QueuedConnection);
	connect(view, SIGNAL(done()), this, SLOT(viewDone()), Qt::QueuedConnection);
	connect(view, SIGNAL(destroyed()), this, SLOT(viewDestroyed()), Qt::QueuedConnection);

	m_numberOfViews++;
}

void ExperimentThread::ResetViews()
{
	m_waitForViews = m_numberOfViews;
	emit reset();
	while (m_waitForViews > 0) usleep(1);
}

void ExperimentThread::NotifyViews(int evals)
{
	m_waitForViews = m_numberOfViews;
	emit changed(evals);
	while (m_waitForViews > 0) usleep(1);
}

void ExperimentThread::viewDone()
{
	m_waitForViews--;
}

void ExperimentThread::viewDestroyed()
{
	m_numberOfViews--;
	m_waitForViews--;
}

void ExperimentThread::run()
{
	// loop through the trials
	int milliseconds = (unsigned int)floor(1000.0 / m_experiment->m_iterPerSecond);
	QTime timer;

	bool successful = true;

	int t, tc = m_experiment->m_trials;
	for (t=0; t<tc && successful; t++)
	{
		TrialHistory* history = new TrialHistory(m_experiment);
		m_experiment->m_trial.push_back(history);

		ResetViews();

		int e = 0;
		while (successful)
		{
			// perform one iteration
			timer.start(); // start QT timer
			unsigned int start = m_experiment->m_problem.getObjectiveFunction()->timesCalled();
			m_experiment->m_algorithm.getAlgorithm()->run();
			unsigned int finish = m_experiment->m_problem.getObjectiveFunction()->timesCalled();
			e += finish - start;

			if (finish == start)
			{
				exit(1);
			}

			// record the history
			history->OnChanged(e);

			// update the online views
			NotifyViews(e);

			// check stopping condition
			if (e >= m_experiment->m_totalEvaluations) break;

			// check wait condition
			if (m_experiment->m_controlMode == 0)
			{
				// manual control
				while (true)
				{
					if (m_controlWidget->isStop())
					{
						successful = false;
						break;
					}
					if (m_controlWidget->isNext()) break;
					usleep(10000);
				}
			}
			else if (m_experiment->m_controlMode == 1)
			{
				// timed control
				int delta = timer.elapsed();
				if (delta <= milliseconds)
				{
					int wait = milliseconds -  delta;
					if (wait > 0)
					{
						msleep(wait);
					}
				}
			}
		}

		if (t < tc - 1)
		{
			m_experiment->m_problem.Init();
			m_experiment->m_algorithm.Init(m_experiment->m_problem);
		}
	}

	if (m_controlWidget != NULL)
	{
		emit destroyControlWidget();
	}

	if (successful)
	{
		emit experimentFinished(m_experiment);
	}
	else
	{
		// close all views and delete the experiment
		emit destroy();
		delete m_experiment;
	}
}


////////////////////////////////////////////////////////////


// static
const char problemSooDef[4096] = 
"{ root branch"
"  { objective-function select paraboloid"
"    { sphere branch"
"      { dimension int 2 1 1000 }"
"    }"
"    { paraboloid branch"
"      { dimension int 2 1 1000 }"
"      { conditioning double 1000.0 1.0 1000000.0 log }"
"    }"
"    { tablet branch"
"      { dimension int 2 1 1000 }"
"      { conditioning double 1000.0 1.0 1000000.0 log }"
"    }"
"    { cigar branch"
"      { dimension int 2 1 1000 }"
"      { conditioning double 1000.0 1.0 1000000.0 log }"
"    }"
"    { twoaxis branch"
"      { dimension int 2 1 1000 }"
"      { conditioning double 1000.0 1.0 1000000.0 log }"
"    }"
"    { Ackley branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { rastrigin branch"
"      { dimension int 10 1 1000 }"
"    }"
"    { Griewangk branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { Rosenbrock branch"
"      { dimension int 29 1 1000 }"
"    }"
"    { Rosenbrock-rotated branch"
"      { dimension int 29 1 1000 }"
"    }"
"    { diff-pow branch"
"      { dimension int 2 1 1000 }"
"    }"
"    { diff-pow-rotated branch"
"      { dimension int 2 1 1000 }"
"    }"
// "    { Schwefel branch"
// "      { dimension int 2 1 1000 }"
// "    }"
"    { Schwefel-ellipsoid branch"
"      { dimension int 2 1 1000 }"
"    }"
"    { Schwefel-ellipsoid-rotated branch"
"      { dimension int 2 1 1000 }"
"    }"
// "    { random branch"
// "      { dimension int 2 1 1000 }"
// "    }"
"    { canyon const }"
"  }"
"}";

// static
const char problemMooDef[4096] = 
"{ root branch"
"  { objective-function select ELLI-1"
"    { ELLI-1 branch"
"      { dimension int 10 1 1000 }"
"      { conditioning double 1000.0 1.0 1000000.0 log }"
"    }"
"    { ELLI-2 branch"
"      { dimension int 10 1 1000 }"
"      { conditioning double 1000.0 1.0 1000000.0 log }"
"    }"
"    { CIGTAB-1 branch"
"      { dimension int 10 1 1000 }"
"      { conditioning double 1000.0 1.0 1000000.0 log }"
"    }"
"    { CIGTAB-2 branch"
"      { dimension int 10 1 1000 }"
"      { conditioning double 1000.0 1.0 1000000.0 log }"
"    }"
// "    { BhinKorn const }"
"    { Schaffer const }"
"    { ZDT-1 branch"
"      { dimension int 10 1 1000 }"
"    }"
"    { ZDT-2 branch"
"      { dimension int 10 1 1000 }"
"    }"
"    { ZDT-3 branch"
"      { dimension int 10 1 1000 }"
"    }"
"    { ZDT-4 branch"
"      { dimension int 10 1 1000 }"
"    }"
// "    { ZDT-5 branch"
// "      { dimension int 11 1 1000 }"
// "    }"
"    { ZDT-6 branch"
"      { dimension int 10 1 1000 }"
"    }"
"    { IHR-1 branch"
"      { dimension int 10 1 1000 }"
"    }"
"    { IHR-2 branch"
"      { dimension int 10 1 1000 }"
"    }"
"    { IHR-3 branch"
"      { dimension int 10 1 1000 }"
"    }"
"    { IHR-6 branch"
"      { dimension int 10 1 1000 }"
"    }"
"    { ZZJ07-F1 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { ZZJ07-F2 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { ZZJ07-F3 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { ZZJ07-F4 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { ZZJ07-F5 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { ZZJ07-F6 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { ZZJ07-F7 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { ZZJ07-F8 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { ZZJ07-F9 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { ZZJ07-F10 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { LZ06-F1 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { LZ06-F2 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { LZ07-F1 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { LZ07-F2 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { LZ07-F3 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { LZ07-F4 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { LZ07-F5 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { LZ07-F6 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { LZ07-F7 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { LZ07-F8 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { LZ07-F9 branch"
"      { dimension int 30 1 1000 }"
"    }"
"    { DTLZ-1 branch"
"      { dimension int 30 1 1000 }"
"      { objectives int 3 1 10 }"
"    }"
"    { DTLZ-2 branch"
"      { dimension int 30 1 1000 }"
"      { objectives int 3 1 10 }"
"    }"
"    { DTLZ-3 branch"
"      { dimension int 30 1 1000 }"
"      { objectives int 3 1 10 }"
"    }"
"    { DTLZ-4 branch"
"      { dimension int 30 1 1000 }"
"      { objectives int 3 1 10 }"
"    }"
"    { DTLZ-5 branch"
"      { dimension int 30 1 1000 }"
"      { objectives int 3 1 10 }"
"    }"
"    { DTLZ-6 branch"
"      { dimension int 30 1 1000 }"
"      { objectives int 3 1 10 }"
"    }"
"    { DTLZ-7 branch"
"      { dimension int 30 1 1000 }"
"      { objectives int 3 1 10 }"
"    }"
"    { Superspheres branch"
"      { dimension int 3 1 1000 }"
"    }"
"  }"
"}";

// static
const char algorithmSooDef[4096] = 
"{ root branch "
"  { search-strategy select CMA-ES"
"    { CMA-ES branch"
"      { update-type select rank-mu"
"        { rank-1 const }"
"        { rank-mu const }"
"      }"
"      { recombination-type select superlinear"
"        { equal const }"
"        { linear const }"
"        { superlinear const }"
"      }"
"    }"
"    { 1+1-CMA-ES const }"
"    { 1+1-ES branch"
"      { step-size-control select self-adaptation"
"        { self-adaptation const }"
"        { one-fifth-rule const }"
"      }"
"    }"
"  }"
"}";

// static
const char algorithmMooDef[4096] = 
"{ root branch "
"  { search-strategy select MO-CMA-ES"
"    { NSGA-2 branch"
"      { mu int 100 1 10000 }"
"      { nm double 20 0 1000000 }"
"      { nc double 20 0 1000000 }"
"      { pc double 0.9 0 1 }"
"    }"
"    { MO-CMA-ES branch"
"      { mu int 100 1 10000 }"
"      { lambda int 100 1 10000 }"
"    }"
"  }"
"}";


////////////////////////////////////////////////////////////


Experiment::Experiment()
: m_confProblemSoo(problemSooDef)
, m_confProblemMoo(problemMooDef)
, m_confAlgorithmSoo(algorithmSooDef)
, m_confAlgorithmMoo(algorithmMooDef)
{
	m_name = "new experiment";
	m_seed = 0;
	m_trials = 1;
	m_totalEvaluations = 1000;
	m_controlMode = 2;
	m_iterPerSecond = 10;
	m_isMOO = false;

	m_online2dLandscape = false;
	m_online3dLandscape = false;
	m_onlineMoFitness = false;
}

Experiment::Experiment(Experiment* tmpl)
: m_confProblemSoo(tmpl->m_confProblemSoo)
, m_confProblemMoo(tmpl->m_confProblemMoo)
, m_confAlgorithmSoo(tmpl->m_confAlgorithmSoo)
, m_confAlgorithmMoo(tmpl->m_confAlgorithmMoo)
{
	m_name = "variation of " + tmpl->m_name;
	m_seed = tmpl->m_seed;
	m_trials = tmpl->m_trials;
	m_isMOO = tmpl->m_isMOO;
	m_totalEvaluations = tmpl->m_totalEvaluations;
	m_controlMode = tmpl->m_controlMode;
	m_iterPerSecond = tmpl->m_iterPerSecond;
	m_online2dLandscape = tmpl->m_online2dLandscape;
	m_online3dLandscape = tmpl->m_online3dLandscape;
	m_onlineMoFitness = tmpl->m_onlineMoFitness;
	m_onlineProperty = tmpl->m_onlineProperty;
	m_recording = tmpl->m_recording;

	if (m_isMOO)
	{
		m_problem.setConfiguration(&m_confProblemMoo);
		m_algorithm.setConfiguration(&m_confAlgorithmMoo);
	}
	else
	{
		m_problem.setConfiguration(&m_confProblemSoo);
		m_algorithm.setConfiguration(&m_confAlgorithmSoo);
	}

// 	m_problem.Init();
// 	m_algorithm.Init(m_problem);
}

Experiment::~Experiment()
{
}


// Return a processed recording for the property
// identified by index after the given number of
// evaluations. The mode parameter controls the
// type of processing.
double Experiment::recording(int index, double evals, int mode) const
{
	if (! m_recording[index]->isScalar()) throw SHARKEXCEPTION("[Experiment::recording] property is not scalar");
	bool l = m_recording[index]->isLogScale();

	// collect one value from each trial
	// (which may already be interpolated)
	int i, ic = m_trial.size();
	std::vector<double> val(ic);
	Array<double> tmp;
	for (i=0; i<ic; i++)
	{
		m_trial[i]->observation(index, evals, tmp);
		val[i] = tmp(0);
	}

	// apply the mode and return the value
	switch (mode)
	{
		case MODE_MEAN:
		{
			double m = 0.0;
			if (l)
			{
				// geometric mean
				for (i=0; i<ic; i++) m += log(val[i]);
				return exp(m / (double)ic);
			}
			else
			{
				// arithmetic mean
				for (i=0; i<ic; i++) m += val[i];
				return m / (double)ic;
			}
		}
		case MODE_MEDIAN:
		{
			std::sort(val.begin(), val.end());
			if (ic & 1) return val[ic / 2];
			else return 0.5 * (val[ic / 2 - 1] + val[ic / 2]);
		}
		case MODE_MINIMUM:
		{
			double m = 1e100;
			for (i=0; i<ic; i++) if (val[i] < m) m = val[i];
			return m;
		}
		case MODE_MAXIMUM:
		{
			double m = -1e100;
			for (i=0; i<ic; i++) if (val[i] > m) m = val[i];
			return m;
		}
		default:
		{
			throw SHARKEXCEPTION("[Experiment::recording] unknown mode value");
			return 0.0;		// dead code
		}
	}
}

QString Experiment::description() const
{
	QString ret;
	char tmp[4096];

	ret += "name: " + m_name + "\n";
	ret += "seed: " + QString::number(m_seed) + "\n";
	ret += "number of trials: " + QString::number(m_trials) + "\n";
	ret += "total number of fitness evaluations: " + QString::number(m_totalEvaluations) + "\n";
	ret += "problem:\n";
	(*m_problem.getConfiguration())[(int)0].description(tmp, 2);
	ret += tmp;
	ret += "algorithm:\n";
	(*m_algorithm.getConfiguration())[(int)0].description(tmp, 2);
	ret += tmp;
	if (m_controlMode == 0)
	{
		ret += "manual control\n";
	}
	if (m_controlMode == 1)
	{
		sprintf(tmp, "timed experiment: %g iterations per second\n", m_iterPerSecond);
		ret += tmp;
	}
	if (m_controlMode == 2)
	{
		ret += "full speed experiment\n";
	}

	return ret;
}

bool Experiment::ExportRecordings()
{
	QString filename = name() + ".data";
	QFile file(filename);
	if (! file.open(QIODevice::WriteOnly)) return false;

	QString content = description() + "\n";

	int r, rc = m_recording.size();
	int t, tc = m_trial.size();
	if (rc > 0)
	{
		for (t=0; t<tc; t++)
		{
			content += "\nTrial " + QString::number(t+1) + ":\n";
			const TrialHistory& tr = trial(t);
			const TrialHistory::PropertyHistory* ph0 = tr.propertyHistory(0);
			int e, ec = ph0->m_evals.dim(0);
			for (e=0; e<ec; e++)
			{
				content += "  time: " + QString::number(ph0->m_evals(e)) + "\n";
				for (r=0; r<rc; r++)
				{
					const TrialHistory::PropertyHistory* ph = tr.propertyHistory(r);
					content += "    property '" + QString(m_recording[r]->name()) + "':\n";
					const Array<double>& data = ph->m_values[e];
					if (data.ndim() == 0)
					{
						content += "      " + QString::number(data()) + "\n";
					}
					else if (data.ndim() == 1)
					{
						content += "     ";
						int i, ic = data.dim(0);
						for (i=0; i<ic; i++) content += " " + QString::number(data(i));
						content += "\n";
					}
					else if (data.ndim() == 2)
					{
						int i, ic = data.dim(0);
						int j, jc = data.dim(1);
						for (i=0; i<ic; i++)
						{
							content += "     ";
							for (j=0; j<jc; j++) content += " " + QString::number(data(i, j));
							content += "\n";
						}
					}
				}
			}
		}
	}
	else content += "[no recordings available]\n";

	file.write(content.toAscii());

	file.close();
	return true;
}
