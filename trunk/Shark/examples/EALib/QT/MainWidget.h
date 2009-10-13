//===========================================================================
/*!
 *  \file MainWidget.h
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


#ifndef _MainWidget_H_
#define _MainWidget_H_


#include <Qt>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QAbstractItemModel>
#include <QTableView>
#include <QMouseEvent>


class Experiment;
class ExperimentWizard;
class PropertyDesc;


class ExperimentListModel : public QAbstractItemModel
{
	Q_OBJECT

public slots:
	void AddExperiment(Experiment* experiment);

public:
	ExperimentListModel(QObject* parent = NULL);
	~ExperimentListModel();

	QVariant data(const QModelIndex& index, int role) const;
	Qt::ItemFlags flags(const QModelIndex& index) const;
	QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
	QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const;
	QModelIndex parent(const QModelIndex& index) const;
	int rowCount(const QModelIndex& parent = QModelIndex()) const;
	int columnCount(const QModelIndex& parent = QModelIndex()) const;

	inline unsigned int getNumberOfExperiments()
	{
		return m_list.size();
	}

	inline Experiment* getExperiment(unsigned int index)
	{
		if (index >= m_list.size()) return NULL;
		return m_list[index];
	}

	void DeleteExperiment(unsigned int index);
	void ExportExperimentRecordings(unsigned int index);

protected:
	std::vector<Experiment*> m_list;
};


class ExperimentListView : public QTableView
{
public:
	ExperimentListView(QWidget* parent = NULL);
	~ExperimentListView();

	inline int getItemUnderContextMenu()
	{
		return m_itemUnderContextMenu;
	}

protected:
	void mousePressEvent(QMouseEvent* event);

	int m_itemUnderContextMenu;
};


class MainWidget : public QWidget
{
	Q_OBJECT

public:
	MainWidget(QWidget* parent = NULL);
	static MainWidget* mainWidget();
	inline ExperimentListModel* model() { return &experiments; }

	QSize sizeHint() const;

public slots:
	void NewExperiment();
	void NewExperimentFromTemplate(bool dummy);
	void ViewConfigurationSummary(bool dummy);
	void Discard(bool dummy);
	void ExportRecordingsToFile(bool dummy);
	void ComparisonPlot();
	void PerformTest();

protected:
	void GetLastValues(Experiment* experiment, PropertyDesc* property, std::vector<double>& values);

	static MainWidget* m_this;

	ExperimentListModel experiments;

	QVBoxLayout layout;
	QPushButton buttonStart;
	QLabel labelExperiments;
	ExperimentListView eTable;

	QWidget widgetBottom;
	QHBoxLayout layoutBottom;
	QComboBox comboProperty;
	QComboBox comboMode;
	QPushButton buttonPlot;
	QPushButton buttonTest;
};


#endif
