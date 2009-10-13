//===========================================================================
/*!
 *  \file Wizard.cpp
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


#include <QMessageBox>
#include <QCoreApplication>

#include "Wizard.h"
#include "View.h"
#include "ValueView.h"
#include "LandscapeView2D.h"
#include "LandscapeView3D.h"
#include "MoFitnessView.h"
#include "MainWidget.h"
#include "ControlWidget.h"


////////////////////////////////////////////////////////////


MyWizardPage::MyWizardPage(ExperimentWizard* parent)
: QWizardPage(parent)
{
	m_wizard = parent;
}


////////////////////////////////////////////////////////////


EW_NameSeedTrials::EW_NameSeedTrials(ExperimentWizard* parent)
: MyWizardPage(parent)
, c_lname("experiment name")
, c_lseed("seed for the random number generator")
, c_ltrials("number of independent trials")
, c_name(NULL)
, c_seed(NULL)
, c_trials(NULL)
, val_seed(0, 2147483647, &c_seed)
, val_trials(1, 10000, &c_trials)
{
	c_layout.addWidget(&c_lname, 0, 0);
	c_layout.addWidget(&c_name, 0, 1);
	c_layout.addWidget(&c_lseed, 1, 0);
	c_layout.addWidget(&c_seed, 1, 1);
	c_layout.addWidget(&c_ltrials, 2, 0);
	c_layout.addWidget(&c_trials, 2, 1);

	setTitle(tr("General Properties"));
	setLayout(&c_layout);

	c_seed.setValidator(&val_seed);
	c_trials.setValidator(&val_trials);

	c_lname.setToolTip(tr("The name of the experiment will be used to refer to this experiment later on.\nYou should give a descriptive name, contrasting this experiment to other experiments."));
	c_lseed.setToolTip(tr("The seed is used to initialize\nthe random number generator."));
	c_ltrials.setToolTip(tr("This experiment will be repeated for the given number of trials.\nIn each trial, the seed of the random number generator is increased by one."));
}

void EW_NameSeedTrials::initializePage()
{
	c_name.setText(m_wizard->m_experiment->m_name);
	c_seed.setText(QString::number(m_wizard->m_experiment->m_seed));
	c_trials.setText(QString::number(m_wizard->m_experiment->m_trials));
}

bool EW_NameSeedTrials::validatePage()
{
	QString s; int pos;

	s = c_seed.text();
	if (val_seed.validate(s, pos) != QValidator::Acceptable)
	{
		QMessageBox::critical(this,
				"Invalid Value",
				"seed must be a valid integer",
				QMessageBox::Ok);
		return false;
	}

	s = c_trials.text();
	if (val_trials.validate(s, pos) != QValidator::Acceptable)
	{
		QMessageBox::critical(this,
				"Invalid Value",
				"number of trials must be in the range [1, 10000]",
				QMessageBox::Ok);
		return false;
	}

	m_wizard->m_experiment->m_name = c_name.text();
	m_wizard->m_experiment->m_seed = c_seed.text().toInt();
	m_wizard->m_experiment->m_trials = c_trials.text().toInt();

	return true;
}


////////////////////////////////////////////////////////////


EW_SoMooType::EW_SoMooType(ExperimentWizard* parent)
: MyWizardPage(parent)
, c_soo("single-objective problem")
, c_moo("multi-objective problem")
{
	c_layout.addWidget(&c_soo);
	c_layout.addWidget(&c_moo);

	setTitle(tr("Problem Type"));
	setLayout(&c_layout);

	c_soo.setToolTip(tr("This experiment will be a single objective\noptimization (SOO) experiment."));
	c_moo.setToolTip(tr("This experiment will be a multi objective\noptimization (MOO) experiment."));
}

void EW_SoMooType::initializePage()
{
	c_soo.setChecked(! m_wizard->m_experiment->m_isMOO);
	c_moo.setChecked(m_wizard->m_experiment->m_isMOO);
}

bool EW_SoMooType::validatePage()
{
	m_wizard->m_experiment->m_isMOO = c_moo.isChecked();

	if (m_wizard->m_experiment->m_isMOO)
	{
		m_wizard->m_experiment->m_problem.setConfiguration(&m_wizard->m_experiment->m_confProblemMoo);
		m_wizard->m_experiment->m_algorithm.setConfiguration(&m_wizard->m_experiment->m_confAlgorithmMoo);
	}
	else
	{
		m_wizard->m_experiment->m_problem.setConfiguration(&m_wizard->m_experiment->m_confProblemSoo);
		m_wizard->m_experiment->m_algorithm.setConfiguration(&m_wizard->m_experiment->m_confAlgorithmSoo);
	}

	return true;
}


////////////////////////////////////////////////////////////


EW_Objective::EW_Objective(ExperimentWizard* parent)
: MyWizardPage(parent)
{
	setTitle(tr("Objective Function"));

	c_layout = NULL;
	c_objective = NULL;
}


void EW_Objective::initializePage()
{
	c_objective = new ConfigurationView(m_wizard->m_experiment->m_problem.getConfiguration());
	c_objective->setToolTip(tr("Choose an objective function and fix its parameters."));
	c_layout = new QVBoxLayout();
	c_layout->addWidget(c_objective);
	setLayout(c_layout);
}

void EW_Objective::cleanupPage()
{
	delete c_layout;
	delete c_objective;
	c_layout = NULL;
	c_objective = NULL;
}

bool EW_Objective::validatePage()
{
	m_wizard->m_experiment->m_problem.Init();

	return true;
}


////////////////////////////////////////////////////////////


EW_Algorithm::EW_Algorithm(ExperimentWizard* parent)
: MyWizardPage(parent)
{
	setTitle(tr("Search Algorithm"));

	c_layout = NULL;
	c_algorithm = NULL;
}


void EW_Algorithm::initializePage()
{
	c_algorithm = new ConfigurationView(m_wizard->m_experiment->m_algorithm.getConfiguration());
	c_algorithm->setToolTip(tr("Choose a search algorithm and fix its strategy parameters."));
	c_layout = new QVBoxLayout();
	c_layout->addWidget(c_algorithm);
	setLayout(c_layout);
}

void EW_Algorithm::cleanupPage()
{
	delete c_layout;
	delete c_algorithm;
	c_layout = NULL;
	c_algorithm = NULL;
}

bool EW_Algorithm::validatePage()
{
	m_wizard->m_experiment->m_algorithm.Init(m_wizard->m_experiment->m_problem);

	return true;
}


////////////////////////////////////////////////////////////


EW_Views::EW_Views(ExperimentWizard* parent)
: MyWizardPage(parent)
, c_2dLandscape("2D fitness landscape")
, c_3dLandscape("3D fitness landscape")
, c_paretoFront("pareto front")
{
	setTitle(tr("Select the Views for Online Monitoring"));
	setLayout(&c_layout);

	c_2dLandscape.setToolTip(tr("Bird's view on the fitness landscape for\ntwo-dimensional search spaces."));
	c_3dLandscape.setToolTip(tr("3D view - camera following the search\nalgorithm through the fitness landscape\nfor two-dimensional search spaces.\nBest suited for the canyon objective.\n"));
	c_paretoFront.setToolTip(tr("Visualization of the front\nof currently pareto optimal solutions."));
}


void EW_Views::initializePage()
{
	if (m_wizard->m_experiment->m_isMOO)
	{
		c_layout.addWidget(&c_paretoFront);
		c_paretoFront.setCheckState(Qt::Checked);
		c_paretoFront.setChecked(m_wizard->m_experiment->m_onlineMoFitness);
	}
	else
	{
		c_layout.addWidget(&c_2dLandscape);
		c_layout.addWidget(&c_3dLandscape);
		c_2dLandscape.setChecked(m_wizard->m_experiment->m_online2dLandscape);
		c_3dLandscape.setChecked(m_wizard->m_experiment->m_online3dLandscape);
	}

	// create checkboxes for all properties
	int i, ic = m_wizard->m_experiment->m_algorithm.properties();
	int j, jc = m_wizard->m_experiment->m_onlineProperty.size();
	for (i=0; i<ic; i++)
	{
		PropertyDesc* prop = m_wizard->m_experiment->m_algorithm.property(i);
		if (prop->isObservable())
		{
			// search for this property
			for (j=0; j<jc; j++)
			{
				if (m_wizard->m_experiment->m_onlineProperty[j] == prop) break;
			}
			bool check = (j < jc);
// 			if (jc == 0 && prop == &PropertyDesc::sooFitness) check = true;
			QCheckBox* box = new QCheckBox(prop->name());
			box->setChecked(check);
			box->setToolTip(tr("Monitor the property '") + prop->name() + tr("'\nonline during the optimization."));
			c_layout.addWidget(box);
			c_checkbox.push_back(box);
		}
	}
}

void EW_Views::cleanupPage()
{
	int i, ic = c_checkbox.size();
	for (i=0; i<ic; i++) delete c_checkbox[i];
	c_checkbox.clear();
}

bool EW_Views::validatePage()
{
	if (m_wizard->m_experiment->m_isMOO)
	{
		m_wizard->m_experiment->m_onlineMoFitness = c_paretoFront.isChecked();
	}
	else
	{
		m_wizard->m_experiment->m_online2dLandscape = c_2dLandscape.isChecked();
		m_wizard->m_experiment->m_online3dLandscape = c_3dLandscape.isChecked();
	}

	m_wizard->m_experiment->m_onlineProperty.clear();
	int i, ic = m_wizard->m_experiment->m_algorithm.properties();
	int j = 0;
	for (i=0; i<ic; i++)
	{
		PropertyDesc* prop = m_wizard->m_experiment->m_algorithm.property(i);
		if (prop->isObservable())
		{
			if (c_checkbox[j]->isChecked())
			{
				m_wizard->m_experiment->m_onlineProperty.push_back(prop);
			}
			j++;
		}
	}

	return true;
}


////////////////////////////////////////////////////////////


EW_RecordProperties::EW_RecordProperties(ExperimentWizard* parent)
: MyWizardPage(parent)
{
	setTitle(tr("Recording of Properties"));
	setLayout(&c_layout);
}


void EW_RecordProperties::initializePage()
{
	// create checkboxes for all properties
	int i, ic = m_wizard->m_experiment->m_algorithm.properties();
	for (i=0; i<ic; i++)
	{
		PropertyDesc* prop = m_wizard->m_experiment->m_algorithm.property(i);
		QCheckBox* box = new QCheckBox(prop->name());
		box->setChecked(prop == &PropertyDesc::sooFitness);
		box->setToolTip(tr("Record the property '") + prop->name() + tr("'\nfor later statistical evaluations."));
		c_layout.addWidget(box);
		c_checkbox.push_back(box);
	}
}

void EW_RecordProperties::cleanupPage()
{
	int i, ic = c_checkbox.size();
	for (i=0; i<ic; i++) delete c_checkbox[i];
	c_checkbox.clear();
}

bool EW_RecordProperties::validatePage()
{
	m_wizard->m_experiment->m_recording.clear();
	int i, ic = c_checkbox.size();
	for (i=0; i<ic; i++)
	{
		if (c_checkbox[i]->isChecked())
		{
			PropertyDesc* prop = m_wizard->m_experiment->m_algorithm.property(i);
			m_wizard->m_experiment->m_recording.push_back(prop);
		}
	}

	return true;
}


////////////////////////////////////////////////////////////


EW_FlowControl::EW_FlowControl(ExperimentWizard* parent)
: MyWizardPage(parent)
, c_manual("manual control")
, c_timed("timed experiment")
, c_fullspeed("full speed experiment")
, c_levals("number of fitness evaluations per trial")
, c_evals("10000")
, c_lips("iterations per second")
, c_ips("10")
, val_evals(1, 1000000, &c_evals)
, val_ips(1.0, 100.0, 20, &c_ips)
{
	setTitle(tr("Flow Control"));
	setLayout(&c_layout);

	c_layout.addWidget(&c_levals, 0, 1);
	c_layout.addWidget(&c_evals, 0, 2);
	c_layout.addWidget(&c_manual, 1, 0);
	c_layout.addWidget(&c_timed, 2, 0);
	c_layout.addWidget(&c_lips, 2, 1);
	c_layout.addWidget(&c_ips, 2, 2);
	c_layout.addWidget(&c_fullspeed, 3, 0);

	c_evals.setValidator(&val_evals);
	c_ips.setValidator(&val_ips);

	c_levals.setToolTip(tr("The number of allowed fitness evaluations\nis used as a stopping criterion.\nA trial ends as soon as the number\nof fitness evaluations performed by the\nsearch algorithm exceeds this value\n(after a full iteration)."));
	c_manual.setToolTip(tr("Open a window with a 'next' and a\n'stop' button for manual control\nof search algorithm iterations."));
	c_timed.setToolTip(tr("Perform the optimization (at most)\nat the given speed, in order\nto facilitate online monitoring."));
	c_fullspeed.setToolTip(tr("Perform the optimization at\nthe maximum possible speed."));
}


void EW_FlowControl::initializePage()
{
	c_evals.setText(QString::number(m_wizard->m_experiment->m_totalEvaluations));
	c_manual.setChecked(m_wizard->m_experiment->m_controlMode == 0);
	c_timed.setChecked(m_wizard->m_experiment->m_controlMode == 1);
	c_fullspeed.setChecked(m_wizard->m_experiment->m_controlMode == 2);
	c_ips.setText(QString::number(m_wizard->m_experiment->m_iterPerSecond));
}

bool EW_FlowControl::validatePage()
{
	m_wizard->m_experiment->m_controlMode = 0;
	if (c_timed.isChecked()) m_wizard->m_experiment->m_controlMode = 1;
	if (c_fullspeed.isChecked()) m_wizard->m_experiment->m_controlMode = 2;

	QString s; int pos;

	s = c_evals.text();
	if (val_evals.validate(s, pos) != QValidator::Acceptable)
	{
		QMessageBox::critical(this,
				"Invalid Value",
				"number of fitness evaluations must be in the range [1, 1000000]",
				QMessageBox::Ok);
		return false;
	}

	s = c_ips.text();
	if (val_ips.validate(s, pos) != QValidator::Acceptable)
	{
		QMessageBox::critical(this,
				"Invalid Value",
				"fitness evaluations per second must be in the range [1, 100]",
				QMessageBox::Ok);
		return false;
	}

	m_wizard->m_experiment->m_totalEvaluations = c_evals.text().toInt();
	m_wizard->m_experiment->m_iterPerSecond = c_ips.text().toDouble();

	return true;
}

int EW_FlowControl::nextId() const
{
	return -1;
}


////////////////////////////////////////////////////////////


ExperimentWizard::ExperimentWizard(Experiment* experiment, QWidget* parent)
: QWizard(parent)
{
	m_experiment = experiment;

	m_NameSeedEvals = new EW_NameSeedTrials(this);
	m_SoMooType = new EW_SoMooType(this);
	m_Objective = new EW_Objective(this);
	m_Algorithm = new EW_Algorithm(this);
	m_Views = new EW_Views(this);
	m_RecordProperties = new EW_RecordProperties(this);
	m_FlowControl = new EW_FlowControl(this);

	setPage(0, m_NameSeedEvals);
	setPage(1, m_SoMooType);
	setPage(2, m_Objective);
	setPage(3, m_Algorithm);
	setPage(4, m_Views);
	setPage(5, m_RecordProperties);
	setPage(6, m_FlowControl);

	setStartId(0);

	setWindowTitle(tr("Experiment Setup Wizard"));

	connect(this, SIGNAL(finished(int)), this, SLOT(OnFinished(int)));

	setAttribute(Qt::WA_DeleteOnClose);
}

ExperimentWizard::~ExperimentWizard()
{
	delete m_NameSeedEvals;
	delete m_SoMooType;
	delete m_Objective;
	delete m_Algorithm;
	delete m_Views;
	delete m_RecordProperties;
	delete m_FlowControl;
}


QSize ExperimentWizard::sizeHint() const
{
	return QSize(600, 400);
}

// perpare the experiment, in particular
// prepare all GUI elements, because QT
// is obviously not able to handle GUI in
// multiple threads
void ExperimentWizard::OnFinished(int resultcode)
{
	if (resultcode != QDialog::Accepted)
	{
		delete m_experiment;
		close();
		return;
	}

	// prepare the experiment
	ManualControlWidget* control = NULL;
	if (m_experiment->m_controlMode == 0) control = new ManualControlWidget(m_experiment->m_name);

	ExperimentThread* thread = new ExperimentThread(m_experiment, control);
	if (control != NULL) connect(thread, SIGNAL(destroyControlWidget()), control, SLOT(onDestroy()), Qt::QueuedConnection);
	connect(thread, SIGNAL(experimentFinished(Experiment*)), MainWidget::mainWidget()->model(), SLOT(AddExperiment(Experiment*)), Qt::QueuedConnection);

	Rng::seed(m_experiment->m_seed);

	// create the views
	if (m_experiment->m_online2dLandscape)
	{
		LandscapeView2D* v = new LandscapeView2D(m_experiment->m_name, m_experiment);
		thread->registerView(v);
		v->show();
	}
	if (m_experiment->m_online3dLandscape)
	{
		LandscapeView3D* v = new LandscapeView3D(m_experiment->m_name, m_experiment);
		thread->registerView(v);
		v->show();
	}
	if (m_experiment->m_onlineMoFitness)
	{
		MoFitnessView* v = new MoFitnessView(m_experiment->m_name, m_experiment);
		thread->registerView(v);
		v->show();
	}

	int i, ic = m_experiment->m_onlineProperty.size();
	for (i=0; i<ic; i++)
	{
		ValueView* v = new ValueView(m_experiment->m_name, m_experiment, m_experiment->m_onlineProperty[i]);
		thread->registerView(v);
		v->show();
	}

	close();
	if (control != NULL) control->show();

	// start the experiment
	thread->start();
}
