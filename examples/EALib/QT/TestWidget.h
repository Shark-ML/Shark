//===========================================================================
/*!
 *  \file TestWidget.h
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


#ifndef _TestWidget_H_
#define _TestWidget_H_


#include <SharkDefs.h>
#include <Array/Array.h>

#include <Qt>
#include <QWidget>
#include <QAbstractItemModel>
#include <QTableView>


class Experiment;


class TestListModel : public QAbstractItemModel
{
	Q_OBJECT

public:
	TestListModel(const std::vector<Experiment*>& list, const Array<double>& pMatrix);
	~TestListModel();

	QVariant data(const QModelIndex& index, int role) const;
	Qt::ItemFlags flags(const QModelIndex& index) const;
	QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
	QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const;
	QModelIndex parent(const QModelIndex& index) const;
	int rowCount(const QModelIndex& parent = QModelIndex()) const;
	int columnCount(const QModelIndex& parent = QModelIndex()) const;

protected:
	std::vector<Experiment*> m_list;
	Array<double> m_pMatrix;
};


// widget for the presentation of test results
class TestWidget : public QTableView
{
	Q_OBJECT

public:
	TestWidget(QString property, const std::vector<Experiment*>& list, const Array<double>& pMatrix, QWidget* parent = NULL);

	QSize sizeHint() const;

protected:
	int size;
	TestListModel model;
};


#endif
