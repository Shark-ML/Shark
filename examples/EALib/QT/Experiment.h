//===========================================================================
/*!
 *  \file Experiment.h
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


#ifndef _Experiment_H_
#define _Experiment_H_


#define MODE_MEAN 1
#define MODE_MEDIAN 2
#define MODE_MAXIMUM 3
#define MODE_MINIMUM 4


#include <QThread>
#include <Array/Array.h>

#include "Optimization.h"


class View;
class EncapsulatedSearchAlgorithm;
class Experiment;
class ExperimentWizard;
class ManualControlWidget;


// The TrialHistory saves a number of observations
// and the corresponding "time" index in terms of fitness
// evaluations. It is able to return interpolated values
// for intermediate "times".
class TrialHistory
{
public:
	TrialHistory(Experiment* experiment);
	~TrialHistory();


	class PropertyHistory
	{
	public:
		PropertyHistory(PropertyDesc* property);
		~PropertyHistory();

		void OnChanged(EncapsulatedSearchAlgorithm& algo, int evals);

		void observation(double evals, Array<double>& values) const;

		PropertyDesc* m_property;
		Array<int> m_evals;
		Array<double> m_values;
	};


	inline int properties() const
	{ return m_property.size(); }

	inline void observation(int index, double evals, Array<double>& values) const
	{ m_property[index]->observation(evals, values); }

	int getObservationIndex(PropertyDesc* property) const;

	inline const PropertyHistory* propertyHistory(int index) const
	{ return m_property[index]; }

	void OnChanged(int evals);

protected:
	Experiment* m_experiment;
	std::vector<PropertyHistory*> m_property;
};


class ExperimentThread : public QThread
{
	Q_OBJECT

signals:
	void reset();
	void changed(int evals);
	void destroy();
	void destroyControlWidget();
	void experimentFinished(Experiment* experiment);

public slots:
	void viewDone();
	void viewDestroyed();

public:
	ExperimentThread(Experiment* experiment, ManualControlWidget* controlWidget = NULL);
	~ExperimentThread();

	void registerView(View* view);
	void ResetViews();
	void NotifyViews(int evals);
	void run();

protected:
	ManualControlWidget* m_controlWidget;
	Experiment* m_experiment;
	int m_numberOfViews;
	int m_waitForViews;
};


// An Experiment object is a collection of the
// following things:
// (*) It keeps a description of all information
//     necessary to reproduce the experiment.
// (*) It holds all objects necessary to carry
//     out the experiment, including a QThread
//     object and online view widgets. These
//     objects are only valid during the experiment.
// (*) It collects result information in order
//     to enable statistical evaluation. These
//     information are, of course, only available
//     after the experiment.
class Experiment
{
public:
	Experiment();
	Experiment(Experiment* tmpl);
	~Experiment();


	inline QString name() const
	{ return m_name; }

	inline unsigned int trials() const
	{ return m_trial.size(); }

	inline const TrialHistory& trial(unsigned int index) const
	{ return *m_trial[index]; }

	inline const TrialHistory& currentTrial() const
	{ return *m_trial[m_trial.size() - 1]; }

	inline int firstEval() const
	{
		int ret = 0;
		int i, ic = m_trial.size();
		for (i=0; i<ic; i++)
		{
			int fe = m_trial[i]->propertyHistory(0)->m_evals(0);
			if (fe > ret) ret = fe;
		}
		return ret;
	}

	inline int recordingPropertyIndex(PropertyDesc* property) const
	{
		int i, ic = m_recording.size();
		for (i=0; i<ic; i++) if (m_recording[i] == property) return i;
		return -1;
	}

	inline bool isRecorded(PropertyDesc* property) const
	{ return (recordingPropertyIndex(property) >= 0); }

	double recording(int index, double evals, int mode = MODE_MEDIAN) const;

	QString description() const;

	bool ExportRecordings();

// protected:
	// experiment description and objects
	QString m_name;
	int m_seed;
	int m_trials;
	int m_totalEvaluations;
	bool m_isMOO;
	EncapsulatedProblem m_problem;
	EncapsulatedSearchAlgorithm m_algorithm;

	// configurations
	Configuration m_confProblemSoo;
	Configuration m_confProblemMoo;
	Configuration m_confAlgorithmSoo;
	Configuration m_confAlgorithmMoo;

	int m_controlMode;
	double m_iterPerSecond;

	// views
	bool m_online2dLandscape;
	bool m_online3dLandscape;
	bool m_onlineMoFitness;
	std::vector<PropertyDesc*> m_onlineProperty;

	// experimental results
	std::vector<PropertyDesc*> m_recording;
	std::vector<TrialHistory*> m_trial;
};


#endif
