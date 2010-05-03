//===========================================================================
/*!
 *  \file ConfigCtrl.h
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


#ifndef _ConfigCtrl_H_
#define _ConfigCtrl_H_


//
// QT-4 widget, using the Model/View architecture,
// for display and editing of configuration settings.
//


#include <Qt>
#include <QWidget>
#include <QComboBox>
#include <QLineEdit>
#include <QSlider>
#include <QDoubleValidator>
#include <QAbstractItemModel>
#include <QItemDelegate>
#include <QTreeView>

#include "config.h"


class MyComboBox : public QComboBox
{
	Q_OBJECT

public:
	MyComboBox(QWidget* parent = NULL);
	~MyComboBox();
};


class DoubleEditor : public QWidget
{
	Q_OBJECT

public:
	DoubleEditor(QWidget* parent, double value, double minimum, double maximum, bool logarithmic);
	~DoubleEditor();

	inline double getValue()
	{
		return value;
	}

	void setValue(double value);

public slots:
	void OnLineEditChanged();
	void OnSliderChanged();

signals:
	void commitData(QWidget* editor);

protected:
	void resizeEvent(QResizeEvent* event);

	QLineEdit lineedit;
	QSlider slider;
	QDoubleValidator validator;

	double value;
	double minimum;
	double maximum;
	bool logarithmic;
};


// encapsulation of a configuration,
// conforming to the QT "Item Model" interface
class ConfigurationModel : public QAbstractItemModel
{
	Q_OBJECT

public:
	ConfigurationModel(Configuration* data, QObject* parent = NULL);
	~ConfigurationModel();

	QVariant data(const QModelIndex& index, int role) const;
	bool setData(const QModelIndex& index, const QVariant& value, int role);
	Qt::ItemFlags flags(const QModelIndex& index) const;
	QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
	QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const;
	QModelIndex parent(const QModelIndex& index) const;
	int rowCount(const QModelIndex& parent = QModelIndex()) const;
	int columnCount(const QModelIndex& parent = QModelIndex()) const;

protected:
	Configuration* root;
};


// QT delegate, providing editor controls
class ConfigurationDelegate : public QItemDelegate
{
	Q_OBJECT

public:
	ConfigurationDelegate(QObject* parent = NULL);
	~ConfigurationDelegate();

	QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const;
	void setEditorData(QWidget* editor, const QModelIndex& index) const;
	void setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const;
	void updateEditorGeometry(QWidget* editor, const QStyleOptionViewItem& option, const QModelIndex& index) const;

public slots:
	void onCommitData(QWidget* editor);
};


// special tree view for a configuration
class ConfigurationView : public QTreeView
{
	Q_OBJECT

public:
	ConfigurationView(Configuration* data, QWidget* parent = NULL);
	~ConfigurationView();

	QSize sizeHint() const;

protected:
	ConfigurationModel model;
	ConfigurationDelegate delegate;
};


#endif
