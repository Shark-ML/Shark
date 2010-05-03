//===========================================================================
/*!
 *  \file MainWidget.cpp
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


#include "MainWidget.h"
#include "Optimization.h"
#include "Experiment.h"
#include "Wizard.h"
#include "ExperimentPropertiesWidget.h"
#include "ComparisonView.h"
#include "TestWidget.h"
#include "UTest.h"

#include <QApplication>
#include <QMessageBox>
#include <QMenu>


ExperimentListModel::ExperimentListModel(QObject* parent)
: QAbstractItemModel(parent)
{
}

ExperimentListModel::~ExperimentListModel()
{
}


void ExperimentListModel::AddExperiment(Experiment* experiment)
{
	m_list.push_back(experiment);
	reset();
}

void ExperimentListModel::DeleteExperiment(unsigned int index)
{
	if (index >= m_list.size()) return;

	Experiment* experiment = m_list[index];
	delete experiment;
	m_list.erase(m_list.begin() + index);
	reset();
}

void ExperimentListModel::ExportExperimentRecordings(unsigned int index)
{
	if (index >= m_list.size()) return;

	if (m_list[index]->ExportRecordings())
	{
		QMessageBox::information(NULL,
				"Data Export",
				"The recordings have been written to the file\n"
				+ m_list[index]->name() + ".data"
			);
	}
	else
	{
		QMessageBox::critical(NULL, "Data Export", "DATA EXPORT FAILED");
	}
}

QVariant ExperimentListModel::data(const QModelIndex& index, int role) const
{
	if (! index.isValid()) return QVariant();

	if (role == Qt::DisplayRole)
	{
		Experiment* e = m_list[index.row()];
		if (index.column() == 0)
		{
			return QVariant(e->m_name);
		}
		else if (index.column() == 1)
		{
			return QVariant(e->m_trials);
		}
		else if (index.column() == 2)
		{
			return QVariant(e->m_totalEvaluations);
		}
	}

	return QVariant();
}

Qt::ItemFlags ExperimentListModel::flags(const QModelIndex& index) const
{
	if (! index.isValid()) return 0;

	Qt::ItemFlags ret = Qt::ItemIsEnabled | Qt::ItemIsSelectable;

	return ret;
}

QVariant ExperimentListModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role == Qt::DisplayRole)
	{
		if (orientation == Qt::Horizontal)
		{
			if (section == 0) return tr("experiment name");
			else if (section == 1) return tr("# trials");
			else if (section == 2) return tr("# fitness evaluations");
		}
		else
		{
			return QVariant(section + 1);
		}
	}

	return QVariant();
}

QModelIndex ExperimentListModel::index(int row, int column, const QModelIndex& parent) const
{
	if (! hasIndex(row, column, parent)) return QModelIndex();

	if (parent.isValid()) return QModelIndex();
	else return createIndex(row, column);
}

QModelIndex ExperimentListModel::parent(const QModelIndex& index) const
{
	return QModelIndex();
}

int ExperimentListModel::rowCount(const QModelIndex& parent) const
{
	if (parent.isValid()) return 0;
	else return m_list.size();
}

int ExperimentListModel::columnCount(const QModelIndex& parent) const
{
	return 3;
}


////////////////////////////////////////////////////////////


ExperimentListView::ExperimentListView(QWidget* parent)
: QTableView(parent)
{
	m_itemUnderContextMenu = -1;

	setSelectionBehavior(QAbstractItemView::SelectRows);
	setSelectionMode(QAbstractItemView::MultiSelection);
}

ExperimentListView::~ExperimentListView()
{
}


void ExperimentListView::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton)
	{
		QModelIndex item = indexAt(event->pos());
		if (item.isValid())
		{
			event->accept();
			selectionModel()->select(item, QItemSelectionModel::Toggle | QItemSelectionModel::Rows);
		}
		else event->ignore();
	}
	else if (event->button() == Qt::RightButton)
	{
		QModelIndex item = indexAt(event->pos());
		if (item.isValid())
		{
			event->accept();
			m_itemUnderContextMenu = item.row();
			QMenu* context = new QMenu();
			QAction* action;
			action = context->addAction("new experiment from this template");
			connect(action, SIGNAL(triggered(bool)), parentWidget(), SLOT(NewExperimentFromTemplate(bool)));
			action = context->addAction("view configuration summary");
			connect(action, SIGNAL(triggered(bool)), parentWidget(), SLOT(ViewConfigurationSummary(bool)));
			action = context->addAction("discard");
			connect(action, SIGNAL(triggered(bool)), parentWidget(), SLOT(Discard(bool)));
			action = context->addAction("export recordings to file");
			connect(action, SIGNAL(triggered(bool)), parentWidget(), SLOT(ExportRecordingsToFile(bool)));
			context->popup(event->globalPos());
		}
		else event->ignore();
	}
	else event->ignore();
}


////////////////////////////////////////////////////////////


// static
MainWidget* MainWidget::m_this = NULL;


MainWidget::MainWidget(QWidget* parent)
: QWidget(parent)
, buttonStart("New Experiment ...", this)
, labelExperiments("List of Experiments", this)
, eTable(this)
, widgetBottom(this)
, comboProperty(&widgetBottom)
, comboMode(&widgetBottom)
, buttonPlot("Comparison Plot", &widgetBottom)
, buttonTest("Mann-Whitney U-test", &widgetBottom)
{
	if (m_this != NULL) throw SHARKEXCEPTION("MainWidget is a singleton");
	m_this = this;

	setWindowTitle("SHARK - Direct Search Demonstration and Test Environment");

	setLayout(&layout);
	layout.addWidget(&buttonStart);
	layout.addWidget(&labelExperiments);
	layout.addWidget(&eTable);
	layout.addWidget(&widgetBottom);
	widgetBottom.setLayout(&layoutBottom);
	layoutBottom.addWidget(&comboProperty);
	layoutBottom.addWidget(&comboMode);
	layoutBottom.addWidget(&buttonPlot);
	layoutBottom.addWidget(&buttonTest);

	labelExperiments.setAlignment(Qt::AlignHCenter);
	eTable.setModel(&experiments);
	eTable.setColumnWidth(0, 250);
	eTable.setColumnWidth(1, 140);
	eTable.setColumnWidth(2, 180);
	comboProperty.addItem("fitness");
// 	comboProperty.addItem("crowding distance");
// 	comboProperty.addItem("epsilon indicator");
	comboProperty.addItem("hypervolume indicator");
	comboMode.addItem("mean");
	comboMode.addItem("median");
	comboMode.addItem("minimum");
	comboMode.addItem("maximum");

	connect(&buttonStart, SIGNAL(pressed()), this, SLOT(NewExperiment()));
	connect(&buttonPlot, SIGNAL(pressed()), this, SLOT(ComparisonPlot()));
	connect(&buttonTest, SIGNAL(pressed()), this, SLOT(PerformTest()));

	eTable.setToolTip(tr("This table lists all finished experiments.\nExperiments can be selected by clicking on them.\nWith the buttons below it is possible to compare\nthe results of selected experiments by means\nof comparison plots and statistical test."));
	comboProperty.setToolTip(tr("Select a property for plotting or testing."));
	comboMode.setToolTip(tr("Select a mode to combine the results of\ndifferent trials for comparison plots."));
	buttonStart.setToolTip(tr("Wizard for the configuration\nof a new experiment."));
	buttonPlot.setToolTip(tr("Open a comparison plot of all selected\nexperiments for the chosen property and mode."));
	buttonTest.setToolTip(tr("Compare the final test results of the\nselected experiments with a Mann-Whitney U-test."));
}


// static
MainWidget* MainWidget::mainWidget()
{
	return m_this;
}

void MainWidget::NewExperiment()
{
	Experiment* experiment = new Experiment();
	ExperimentWizard* wizard = new ExperimentWizard(experiment, this);
	wizard->show();
}

void MainWidget::NewExperimentFromTemplate(bool dummy)
{
	int index = eTable.getItemUnderContextMenu();
	Experiment* tmplExperiment = experiments.getExperiment(index);
	if (tmplExperiment == NULL) return;

	Experiment* experiment = new Experiment(tmplExperiment);
	ExperimentWizard* wizard = new ExperimentWizard(experiment, this);
	wizard->show();
}

void MainWidget::ViewConfigurationSummary(bool dummy)
{
	int index = eTable.getItemUnderContextMenu();
	Experiment* experiment = experiments.getExperiment(index);
	if (experiment == NULL) return;

	ExperimentPropertiesWidget* w = new ExperimentPropertiesWidget(experiment, NULL);
	w->show();
}

void MainWidget::Discard(bool dummy)
{
	int index = eTable.getItemUnderContextMenu();
	experiments.DeleteExperiment(index);
}

void MainWidget::ExportRecordingsToFile(bool dummy)
{
	int index = eTable.getItemUnderContextMenu();
	experiments.ExportExperimentRecordings(index);
}

void MainWidget::ComparisonPlot()
{
	// create list of selected experiments
	std::vector<Experiment*> selected;

	QItemSelectionModel* ism = eTable.selectionModel();
	unsigned int i, ic = experiments.getNumberOfExperiments();
	for (i=0; i<ic; i++)
	{
		if (ism->isRowSelected(i, QModelIndex()))
		{
			selected.push_back(experiments.getExperiment(i));
		}
	}

	ic = selected.size();
	if (ic == 0)
	{
		QMessageBox::critical(this, "invalid selection",
				"You have not selected any experiments.\n"
				"Click on experiments to select them.\n"
				"(The list is initially empty. In this case\n"
				"you may which to start some experiments using\n"
				"the button at the top.)"
			);
		return;
	}

	// determine the property
	PropertyDesc* prop = NULL;
	switch (comboProperty.currentIndex())
	{
		case 0:
			prop = &PropertyDesc::sooFitness;
			break;
		case 1:
// 			prop = &PropertyDesc::crowdingDistance;
// 			break;
// 		case 2:
// 			prop = &PropertyDesc::epsilonIndicator;
// 			break;
// 		case 3:
			prop = &PropertyDesc::hypervolumeIndicator;
			break;
	}
	if (prop == NULL)
	{
		QMessageBox::critical(this, "invalid property",
				"Please select a property to plot."
			);
		return;
	}

	// determine the mode
	int mode = 0;
	switch (comboMode.currentIndex())
	{
		case 0:
			mode = MODE_MEAN;
			break;
		case 1:
			mode = MODE_MEDIAN;
			break;
		case 2:
			mode = MODE_MINIMUM;
			break;
		case 3:
			mode = MODE_MAXIMUM;
			break;
	}
	if (mode == 0)
	{
		QMessageBox::critical(this, "invalid mode",
				"Please select a mode to combine trials for plotting."
			);
		return;
	}

	// filter the list
	std::vector<Experiment*> toplot;
	std::vector<Experiment*> notavail;
	for (i=0; i<ic; i++)
	{
		Experiment* ex = selected[i];
		if (ex->isRecorded(prop)) toplot.push_back(ex);
		else notavail.push_back(ex);
	}
	if (toplot.size() < ic)
	{
		if (toplot.size() == 0)
		{
			QMessageBox::critical(this, "property not recorded",
					"The property '" + comboProperty.currentText() + "' has not been\n"
					"recorded for any of the selected experiments.\n"
					"Please select at least one valid experiment."
				);
			return;
		}

		QString msg;
		msg = "The property '" + comboProperty.currentText() + "' has not been recorded\nfor the following experiments:\n";
		for (i=0; i<notavail.size(); i++) msg += "    " + notavail[i]->name() + "\n";
		msg += "Do you want to continue anyway?";
		if (QMessageBox::question(this, "property not recorded", msg,
				QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
		{
			return;
		}
	}
	ic = toplot.size();

	// launch a plot window
	ComparisonView* cv = new ComparisonView(prop, mode, NULL);
	for (i=0; i<ic; i++) cv->AddExperiment(toplot[i]);
	cv->show();
}

void MainWidget::PerformTest()
{
	// create list of selected experiments
	std::vector<Experiment*> selected;

	QItemSelectionModel* ism = eTable.selectionModel();
	unsigned int i, j, ic = experiments.getNumberOfExperiments();
	for (i=0; i<ic; i++)
	{
		if (ism->isRowSelected(i, QModelIndex()))
		{
			selected.push_back(experiments.getExperiment(i));
		}
	}

	ic = selected.size();
	if (ic < 2)
	{
		QMessageBox::critical(this, "invalid selection",
				"You have selected less than two experiments.\n"
				"Click on experiments to select them.\n"
				"(The list is initially empty. In this case\n"
				"you may which to start some experiments using\n"
				"the button at the top.)"
			);
		return;
	}

	// determine the property
	PropertyDesc* prop = NULL;
	switch (comboProperty.currentIndex())
	{
		case 0:
			prop = &PropertyDesc::sooFitness;
			break;
		case 1:
// 			prop = &PropertyDesc::crowdingDistance;
// 			break;
// 		case 2:
// 			prop = &PropertyDesc::epsilonIndicator;
// 			break;
// 		case 3:
			prop = &PropertyDesc::hypervolumeIndicator;
			break;
	}
	if (prop == NULL)
	{
		QMessageBox::critical(this, "invalid property",
				"Please select a property to which the test should be applied."
			);
		return;
	}

	// filter the list
	std::vector<Experiment*> totest;
	std::vector<Experiment*> notavail;
	for (i=0; i<ic; i++)
	{
		Experiment* ex = selected[i];
		if (ex->isRecorded(prop)) totest.push_back(ex);
		else notavail.push_back(ex);
	}
	if (totest.size() < ic)
	{
		if (totest.size() < 2)
		{
			QMessageBox::critical(this, "property not recorded",
					"The property '" + comboProperty.currentText() + "' has not\n"
					"been recorded for at least two experiments.\n"
					"Please select at least two valid experiments."
				);
			return;
		}

		QString msg;
		msg = "The property '" + comboProperty.currentText() + "' has not been recorded\nfor the following experiments:\n";
		for (i=0; i<notavail.size(); i++) msg += "    " + notavail[i]->name() + "\n";
		msg += "Do you want to continue anyway?";
		if (QMessageBox::question(this, "property not recorded", msg,
				QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
		{
			return;
		}
	}
	ic = totest.size();

	// perform the tests
	Array<double> p(ic, ic);
	std::vector<double> data_i;
	std::vector<double> data_j;
	double twosided, A_leftOf_B, B_leftOf_A;
	for (i=0; i<ic; i++)
	{
		Experiment* ex_i = totest[i];
		GetLastValues(ex_i, prop, data_i);
		for (j=0; j<i; j++)
		{
			Experiment* ex_j = totest[j];
			GetLastValues(ex_j, prop, data_j);
			UTest(data_i, data_j, twosided, A_leftOf_B, B_leftOf_A);
			p(i, j) = A_leftOf_B;
			p(j, i) = B_leftOf_A;
		}
		p(i, i) = 0.5;
	}

	// launch a result window
	TestWidget* tw = new TestWidget(prop->name(), totest, p, NULL);
	tw->show();
}

QSize MainWidget::sizeHint() const
{
	return QSize(650, 500);
}

void MainWidget::GetLastValues(Experiment* experiment, PropertyDesc* property, std::vector<double>& values)
{
	int t, tc = experiment->trials();
	values.resize(tc);
	Array<double> val;
	for (t=0; t<tc; t++)
	{
		const TrialHistory& history = experiment->trial(t);
		history.observation(history.getObservationIndex(property),
				experiment->m_totalEvaluations,
				val);
		values[t] = val(0);
	}
}
