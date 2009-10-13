//===========================================================================
/*!
 *  \file TestWidget.cpp
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


#include "TestWidget.h"
#include "Experiment.h"


TestListModel::TestListModel(const std::vector<Experiment*>& list, const Array<double>& pMatrix)
: QAbstractItemModel(NULL)
{
	m_list = list;
	m_pMatrix = pMatrix;
}

TestListModel::~TestListModel()
{
}


QVariant TestListModel::data(const QModelIndex& index, int role) const
{
	if (! index.isValid()) return QVariant();

	if (role == Qt::DisplayRole)
	{
		if (index.column() == index.row()) return QVariant("---");
		else return QVariant(m_pMatrix(index.row(), index.column()));
	}

	return QVariant();
}

Qt::ItemFlags TestListModel::flags(const QModelIndex& index) const
{
	if (! index.isValid()) return 0;

	Qt::ItemFlags ret = Qt::ItemIsEnabled | Qt::ItemIsSelectable;

	return ret;
}

QVariant TestListModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role == Qt::DisplayRole)
	{
		if (orientation == Qt::Horizontal)
		{
			return m_list[section]->name();
		}
		else
		{
			return m_list[section]->name();
		}
	}

	return QVariant();
}

QModelIndex TestListModel::index(int row, int column, const QModelIndex& parent) const
{
	if (! hasIndex(row, column, parent)) return QModelIndex();

	if (parent.isValid()) return QModelIndex();
	else return createIndex(row, column);
}

QModelIndex TestListModel::parent(const QModelIndex& index) const
{
	return QModelIndex();
}

int TestListModel::rowCount(const QModelIndex& parent) const
{
	if (parent.isValid()) return 0;
	else return m_list.size();
}

int TestListModel::columnCount(const QModelIndex& parent) const
{
	if (parent.isValid()) return 0;
	else return m_list.size();
}


////////////////////////////////////////////////////////////


TestWidget::TestWidget(QString property, const std::vector<Experiment*>& list, const Array<double>& pMatrix, QWidget* parent)
: QTableView(parent)
, model(list, pMatrix)
{
	setWindowTitle("U-Test based comparison of " + property);

	setModel(&model);
	size = list.size();
	int i;
	for (i=0; i<size; i++) setColumnWidth(i, 150);

	setToolTip(tr(
			"The table lists the p-values resulting from pairwise U-Tests\n"
			"of the selected experiments. A small p-value indicates that\n"
			"with high probability the values obtained in the experiment\n"
			"in this row are lower than the values obtained in the experiment\n"
			"in this column.\n"
			"The p-value is (an upper bound on) the probability to obtain\n"
			"this relation by pure chance (given the results). Usually, a\n"
			"p-value of 0.05 or lower is considered sufficiently significant.\n"
			"Other standard values are 0.01 and 0.001 for highly significant\n"
			"results."
		));
}


QSize TestWidget::sizeHint() const
{
	return QSize(280 + 150 * size, 130 + 30 * size);
}
