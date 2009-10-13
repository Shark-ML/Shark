//===========================================================================
/*!
 *  \file Wizard.h
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


#ifndef _Wizard_H_
#define _Wizard_H_


// The ExperimentWizard is used to build
// objects of type Optimization and Experiment.


#include <Qt>
#include <QWidget>
#include <QWizard>
#include <QWizardPage>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QRadioButton>
#include <QCheckBox>
#include <QComboBox>
#include <QIntValidator>
#include <QDoubleValidator>
#include <QScrollArea>
#include "ConfigCtrl.h"
#include "Optimization.h"
#include "Experiment.h"


class ExperimentWizard;


class MyWizardPage : public QWizardPage
{
	Q_OBJECT

public:
	MyWizardPage(ExperimentWizard* parent);

protected:
	ExperimentWizard* m_wizard;
};


class EW_NameSeedTrials : public MyWizardPage
{
	Q_OBJECT

public:
	EW_NameSeedTrials(ExperimentWizard* parent);

	void initializePage();
	bool validatePage();

	QGridLayout c_layout;
	QLabel c_lname;
	QLabel c_lseed;
	QLabel c_ltrials;
	QLineEdit c_name;
	QLineEdit c_seed;
	QLineEdit c_trials;

	QIntValidator val_seed;
	QIntValidator val_trials;
};


class EW_SoMooType : public MyWizardPage
{
	Q_OBJECT

public:
	EW_SoMooType(ExperimentWizard* parent);

	void initializePage();
	bool validatePage();

	QVBoxLayout c_layout;
	QRadioButton c_soo;
	QRadioButton c_moo;
};


class EW_Objective : public MyWizardPage
{
	Q_OBJECT

public:
	EW_Objective(ExperimentWizard* parent);

	void initializePage();
	void cleanupPage();
	bool validatePage();

	QVBoxLayout* c_layout;
	ConfigurationView* c_objective;
};


class EW_Algorithm : public MyWizardPage
{
	Q_OBJECT

public:
	EW_Algorithm(ExperimentWizard* parent);

	void initializePage();
	void cleanupPage();
	bool validatePage();

	QVBoxLayout* c_layout;
	ConfigurationView* c_algorithm;
};


class EW_Views : public MyWizardPage
{
	Q_OBJECT

public:
	EW_Views(ExperimentWizard* parent);

	void initializePage();
	void cleanupPage();
	bool validatePage();

	QVBoxLayout c_layout;
	std::vector<QCheckBox*> c_checkbox;

	// SOO
	QCheckBox c_2dLandscape;
	QCheckBox c_3dLandscape;

	// MOO
	QCheckBox c_paretoFront;
};


class EW_RecordProperties : public MyWizardPage
{
	Q_OBJECT

public:
	EW_RecordProperties(ExperimentWizard* parent);

	void initializePage();
	void cleanupPage();
	bool validatePage();

	QVBoxLayout c_layout;
	std::vector<QCheckBox*> c_checkbox;
};


class EW_FlowControl : public MyWizardPage
{
	Q_OBJECT

public:
	EW_FlowControl(ExperimentWizard* parent);

	void initializePage();
	bool validatePage();
	int nextId() const;

	QGridLayout c_layout;
	QRadioButton c_manual;
	QRadioButton c_timed;
	QRadioButton c_fullspeed;
	QLabel c_levals;
	QLineEdit c_evals;
	QLabel c_lips;
	QLineEdit c_ips;

	QIntValidator val_evals;
	QDoubleValidator val_ips;
};


class ExperimentWizard : public QWizard
{
	Q_OBJECT

public slots:
	void OnFinished(int resultcode);

public:
	ExperimentWizard(Experiment* experiment, QWidget* parent = NULL);
	~ExperimentWizard();

	QSize sizeHint() const;

	Experiment* m_experiment;

	EW_NameSeedTrials* m_NameSeedEvals;
	EW_SoMooType* m_SoMooType;
	EW_Objective* m_Objective;
	EW_Algorithm* m_Algorithm;
	EW_Views* m_Views;
	EW_RecordProperties* m_RecordProperties;
	EW_FlowControl* m_FlowControl;
};


#endif
