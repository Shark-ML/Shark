//===========================================================================
/*!
 *  \file ConfigCtrl.cpp
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


#include "ConfigCtrl.h"

#include <QSpinBox>


MyComboBox::MyComboBox(QWidget* parent)
: QComboBox(parent)
{
	connect(this, SIGNAL(currentIndexChanged(int)), this, SLOT(close()));
}

MyComboBox::~MyComboBox()
{
}


////////////////////////////////////////////////////////////


DoubleEditor::DoubleEditor(QWidget* parent, double value, double minimum, double maximum, bool logarithmic)
: QWidget(parent)
, lineedit(this)
, slider(this)
, validator(minimum, maximum, 20, &lineedit)
{
	this->value = value;
	this->minimum = minimum;
	this->maximum = maximum;
	this->logarithmic = logarithmic;

	setFocusPolicy(Qt::StrongFocus);

	lineedit.setValidator(&validator);
	QString txt;
	txt.setNum(value);
	lineedit.setText(txt);

	slider.setOrientation(Qt::Horizontal);
	slider.setRange(0, 100);
	slider.setTickPosition(QSlider::TicksBelow);
	slider.setTickInterval(10);
	if (logarithmic) slider.setValue((int)Shark::round(100.0 *(log(value) - log(minimum)) / (log(maximum) - log(minimum))));
	else slider.setValue((int)Shark::round(100.0 *(value - minimum) / (maximum - minimum)));

	QRect rect = geometry();
	if (rect.width() < 200) rect.setWidth(200);
	setGeometry(rect);
	lineedit.setGeometry(0, 0, 100, rect.height() - 8);
	slider.setGeometry(100, 0, rect.width() - 100, rect.height() - 8);

	connect(&lineedit, SIGNAL(editingFinished()), this, SLOT(OnLineEditChanged()));
	connect(&slider, SIGNAL(valueChanged(int)), this, SLOT(OnSliderChanged()));
}

DoubleEditor::~DoubleEditor()
{
}


void DoubleEditor::setValue(double value)
{
	QString txt;
	txt.setNum(value);
	lineedit.setText(txt);

	if (logarithmic) slider.setValue((int)Shark::round(100.0 *(log(value) - log(minimum)) / (log(maximum) - log(minimum))));
	else slider.setValue((int)Shark::round(100.0 *(value - minimum) / (maximum - minimum)));
}

void DoubleEditor::OnLineEditChanged()
{
	QString txt = lineedit.text();
	value = txt.toDouble();

	if (logarithmic) slider.setValue((int)Shark::round(100.0 * (log(value) - log(minimum)) / (log(maximum) - log(minimum))));
	else slider.setValue((int)Shark::round(100.0 * (value - minimum) / (maximum - minimum)));

	emit commitData(this);
}

void DoubleEditor::OnSliderChanged()
{
	int pos = slider.value();
	if (logarithmic) value = exp(log(minimum) + pos * (log(maximum) - log(minimum)) / 100.0);
	else value = minimum + pos * (maximum - minimum) / 100.0;

	QString txt;
	txt.setNum(value);
	lineedit.setText(txt);

	emit commitData(this);
}

void DoubleEditor::resizeEvent(QResizeEvent* event)
{
	QRect rect = geometry();
	if (rect.width() < 200) rect.setWidth(200);
	setGeometry(rect);
	lineedit.setGeometry(0, 0, 100, childrenRect().height());
	slider.setGeometry(100, 0, rect.width() - 100, childrenRect().height());
}


////////////////////////////////////////////////////////////


ConfigurationModel::ConfigurationModel(Configuration* data, QObject* parent)
: QAbstractItemModel(parent)
{
	this->root = data;
}

ConfigurationModel::~ConfigurationModel()
{
}


QVariant ConfigurationModel::data(const QModelIndex& index, int role) const
{
	if (! index.isValid()) return QVariant();

	if (role == Qt::DisplayRole)
	{
		PropertyNode* node = (PropertyNode*)index.internalPointer();
		if (index.column() == 0)
		{
			return tr(node->getName());
		}
		else if (index.column() == 1)
		{
			QString str;
			switch (node->getType())
			{
				case PropertyNode::etBool:
					if (node->getBool()) str = "true"; else str = "false";
					break;
				case PropertyNode::etInt:
					str.setNum(node->getInt());
					break;
				case PropertyNode::etDouble:
					str.setNum(node->getDouble());
					break;
				case PropertyNode::etString:
					str = node->getString();
					break;
				case PropertyNode::etSelectOne:
					str = node->getSelected();
					break;
				case PropertyNode::etBranch:
					str = "";
					break;
				case PropertyNode::etConst:
					str = "";
					break;
			}
			return str;
		}
	}
	else if (role == Qt::EditRole)
	{
		PropertyNode* node = (PropertyNode*)index.internalPointer();
		if (index.column() == 1)
		{
			QString str;
			switch (node->getType())
			{
				case PropertyNode::etBool:
					return node->getBool();
				case PropertyNode::etInt:
					return node->getInt();
				case PropertyNode::etDouble:
					return node->getDouble();
				case PropertyNode::etString:
					return node->getString();
				case PropertyNode::etSelectOne:
					return node->getSelectedIndex();
				default:
					return QVariant();
			}
		}
	}

	return QVariant();
}

bool ConfigurationModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
	if (index.isValid() && role == Qt::EditRole)
	{
		PropertyNode* node = (PropertyNode*)index.internalPointer();
		QString str;
		switch (node->getType())
		{
			case PropertyNode::etBool:
				node->set(value.toBool());
				break;
			case PropertyNode::etInt:
				node->set(value.toInt());
				break;
			case PropertyNode::etDouble:
				node->set(value.toDouble());
				break;
			case PropertyNode::etString:
				node->set(qPrintable(value.toString()));
				break;
			case PropertyNode::etSelectOne:
			{
				node->set((*node)[value.toInt()]);
				dataChanged(this->index(0, 0, index), this->index(node->size() - 1, 1, index));
				break;
			}
			case PropertyNode::etBranch:
				return false;
			case PropertyNode::etConst:
				return false;
		}
		emit dataChanged(index, index);
		return true;
	}
	else return false;
}

Qt::ItemFlags ConfigurationModel::flags(const QModelIndex& index) const
{
	if (! index.isValid()) return 0;

	Qt::ItemFlags ret = Qt::ItemIsSelectable;

	PropertyNode* node = (PropertyNode*)index.internalPointer();
	PropertyNode* cand = node;
	bool enabled = true;
	while (cand != root)
	{
		PropertyNode* parent = cand->getParent();
		if (parent->getType() == PropertyNode::etSelectOne)
		{
			if (&parent->getSelectedNode() != cand)
			{
				enabled = false;
				break;
			}
		}
		cand = parent;
	}
	if (enabled) ret |= Qt::ItemIsEnabled;

	if (index.column() == 1 && (enabled)) ret |= Qt::ItemIsEditable;

	return ret;
}

QVariant ConfigurationModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role == Qt::DisplayRole)
	{
		if (section == 0) return tr("property");
		else if (section == 1) return tr("value");
	}

	return QVariant();
}

QModelIndex ConfigurationModel::index(int row, int column, const QModelIndex& parent) const
{
	if (! hasIndex(row, column, parent)) return QModelIndex();

	PropertyNode* parentNode;
	if (! parent.isValid()) parentNode = root;
	else parentNode = (PropertyNode*)parent.internalPointer();

	if (column < 0 || column >= 2) return QModelIndex();
	if (row < 0 || row >= parentNode->size()) return QModelIndex();

	PropertyNode* childNode = &(*parentNode)[row];
	return createIndex(row, column, childNode);
}

QModelIndex ConfigurationModel::parent(const QModelIndex& index) const
{
	if (! index.isValid()) return QModelIndex();

	PropertyNode* childNode = (PropertyNode*)index.internalPointer();
	PropertyNode* parentNode = childNode->getParent();

	if (parentNode == root) return QModelIndex();

	// find the row of the parent
	PropertyNode* grandParentNode = parentNode->getParent();
	int row;
	for (row=0; row<grandParentNode->size(); row++) if (&(*grandParentNode)[row] == parentNode) break;

	return createIndex(row, 0, parentNode);
}

int ConfigurationModel::rowCount(const QModelIndex& parent) const
{
	PropertyNode* node;
	if (! parent.isValid()) node = root;
	else node = (PropertyNode*)parent.internalPointer();

	if (node->getType() == PropertyNode::etSelectOne)
	{
		return node->size();

// 		PropertyNode& sub = node->getSelectedNode();
// 		if (sub.getType() == PropertyNode::etConst) return 0;
// 		else return 1;
	}
	else if (node->getType() == PropertyNode::etBranch)
	{
		return node->size();
	}
	else return 0;
}

int ConfigurationModel::columnCount(const QModelIndex& parent) const
{
	return 2;
}


////////////////////////////////////////////////////////////


ConfigurationDelegate::ConfigurationDelegate(QObject* parent)
: QItemDelegate(parent)
{
}

ConfigurationDelegate::~ConfigurationDelegate()
{
}


QWidget* ConfigurationDelegate::createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
	if (! index.isValid()) return NULL;

	PropertyNode* node = (PropertyNode*)index.internalPointer();
	PropertyNode::eType type = node->getType();
	if (type == PropertyNode::etBool)
	{
		MyComboBox* editor = new MyComboBox(parent);
		editor->addItem("false");
		editor->addItem("true");
		editor->setEditable(false);
		if (node->getBool()) editor->setCurrentIndex(1); else editor->setCurrentIndex(0);
		return editor;
	}
	else if (type == PropertyNode::etInt)
	{
		QSpinBox* editor = new QSpinBox(parent);
		editor->setMinimum(node->getIntMin());
		editor->setMaximum(node->getIntMax());
		return editor;	
	}
	else if (type == PropertyNode::etDouble)
	{
		DoubleEditor* editor = new DoubleEditor(parent, node->getDouble(), node->getDoubleMin(), node->getDoubleMax(), node->isLogarithmic());
		connect(editor, SIGNAL(commitData(QWidget*)), this, SLOT(onCommitData(QWidget*)));
		return editor;
	}
	else if (type == PropertyNode::etString)
	{
		QLineEdit* editor = new QLineEdit(parent);
		editor->setText(node->getString());
		return editor;
	}
	else if (type == PropertyNode::etSelectOne)
	{
		MyComboBox* editor = new MyComboBox(parent);
		int i, ic = node->size();
		for (i=0; i<ic; i++) editor->addItem((*node)[i].getName());
		editor->setEditable(false);
		editor->setCurrentIndex(node->getSelectedIndex());
		return editor;
	}
	else return NULL;
}

void ConfigurationDelegate::setEditorData(QWidget* editor, const QModelIndex& index) const
{
	if (! index.isValid()) return;

	PropertyNode* node = (PropertyNode*)index.internalPointer();
	PropertyNode::eType type = node->getType();
	if (type == PropertyNode::etBool)
	{
		MyComboBox* ed = (MyComboBox*)editor;
		bool value = index.model()->data(index, Qt::EditRole).toBool();
		if (value) ed->setCurrentIndex(1); else ed->setCurrentIndex(0);
	}
	else if (type == PropertyNode::etInt)
	{
		QSpinBox* ed = (QSpinBox*)editor;
		int value = index.model()->data(index, Qt::EditRole).toInt();
		ed->setValue(value);
	}
	else if (type == PropertyNode::etDouble)
	{
		DoubleEditor* ed = (DoubleEditor*)editor;
		double value = index.model()->data(index, Qt::EditRole).toDouble();
		ed->setValue(value);
	}
	else if (type == PropertyNode::etString)
	{
		QLineEdit* ed = (QLineEdit*)editor;
		QString value = index.model()->data(index, Qt::EditRole).toString();
		ed->setText(value);
	}
	else if (type == PropertyNode::etSelectOne)
	{
		MyComboBox* ed = (MyComboBox*)editor;
		int value = index.model()->data(index, Qt::EditRole).toInt();
		ed->setCurrentIndex(value);
	}
	else return;
}

void ConfigurationDelegate::setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const
{
	if (! index.isValid()) return;

	PropertyNode* node = (PropertyNode*)index.internalPointer();
	PropertyNode::eType type = node->getType();
	if (type == PropertyNode::etBool)
	{
		MyComboBox* ed = (MyComboBox*)editor;
		bool value = (ed->currentIndex() != 0);
		model->setData(index, value, Qt::EditRole);
	}
	else if (type == PropertyNode::etInt)
	{
		QSpinBox* ed = (QSpinBox*)editor;
		ed->interpretText();
		int value = ed->value();
		model->setData(index, value, Qt::EditRole);
	}
	else if (type == PropertyNode::etDouble)
	{
		DoubleEditor* ed = (DoubleEditor*)editor;
		double value = ed->getValue();
		model->setData(index, value, Qt::EditRole);
	}
	else if (type == PropertyNode::etString)
	{
		QLineEdit* ed = (QLineEdit*)editor;
		QString value = ed->text();
		model->setData(index, value, Qt::EditRole);
	}
	else if (type == PropertyNode::etSelectOne)
	{
		MyComboBox* ed = (MyComboBox*)editor;
		int value = ed->currentIndex();
		model->setData(index, value, Qt::EditRole);
	}
	else return;
}

void ConfigurationDelegate::updateEditorGeometry(QWidget* editor, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
	if (! index.isValid()) return;

	PropertyNode* node = (PropertyNode*)index.internalPointer();
	PropertyNode::eType type = node->getType();
	if (type == PropertyNode::etBool)
	{
		MyComboBox* ed = (MyComboBox*)editor;
		ed->setGeometry(option.rect);
	}
	else if (type == PropertyNode::etInt)
	{
		QSpinBox* ed = (QSpinBox*)editor;
		ed->setGeometry(option.rect);
	}
	else if (type == PropertyNode::etDouble)
	{
		DoubleEditor* ed = (DoubleEditor*)editor;
		ed->setGeometry(option.rect);
	}
	else if (type == PropertyNode::etString)
	{
		QLineEdit* ed = (QLineEdit*)editor;
		ed->setGeometry(option.rect);
	}
	else if (type == PropertyNode::etSelectOne)
	{
		MyComboBox* ed = (MyComboBox*)editor;
		ed->setGeometry(option.rect);
	}
	else return;
}

void ConfigurationDelegate::onCommitData(QWidget* editor)
{
	emit commitData(editor);
}


////////////////////////////////////////////////////////////


ConfigurationView::ConfigurationView(Configuration* data, QWidget* parent)
: QTreeView(parent)
, model(data)
, delegate()
{
	setModel(&model);
	setItemDelegate(&delegate);

	connect(this, SIGNAL(pressed(const QModelIndex&)), this, SLOT(edit(const QModelIndex&)));

	setColumnWidth(0, 280);
	setColumnWidth(1, 280);

	expandAll();
}

ConfigurationView::~ConfigurationView()
{
}


QSize ConfigurationView::sizeHint() const
{
	return QSize(570, 300);
}
