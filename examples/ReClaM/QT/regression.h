//===========================================================================
/*!
 *  \file regression.h
 *
 *  \author  T. Glasmachers
 *  \date    2007
 *
 *  \par Copyright (c) 1999-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-27974<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  \par Project:
 *      ReClaM
 * 
 *  \par File and Revision:
 *      $RCSfile: regression.h,v $<BR>
 *      $Revision: 1.2 $<BR>
 *      $Date: 2007/09/21 18:02:01 $
 *
 *  \par Changes:
 *      $Log: regression.h,v $
 *      Revision 1.2  2007/09/21 18:02:01  leninga
 *      complete header
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of ReClaM. This library is free software;
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


#ifndef _regression_H_
#define _regression_H_


#include <Qt>
#include <QWidget>
#include <QApplication>
#include <QPainter>
#include <QPaintEvent>
#include <QImage>
#include <QLabel>
#include <QComboBox>
#include <QSlider>
#include <QPushButton>
#include <QScrollArea>

#include <ReClaM/Dataset.h>
#include <ReClaM/KernelFunction.h>
#include <ReClaM/Svm.h>
#include <ReClaM/MeanSquaredError.h>


class FrameWidget;


class NoisySinc : public DataSource
{
public:
	NoisySinc(double noise = 0.1);
	~NoisySinc();

	bool GetData(Array<double>& data, Array<double>& target, int count);

protected:
	double noise;
};


class Property
{
public:
	Property(const char* name, Model* model, unsigned int param, double m, double M, bool l = false);

	double getValue();
	void setValue(double value);

	const char* name;
	Model* model;
	unsigned int param;
	double minVal;
	double maxVal;
	bool logarithmic;
};


class Doc
{
public:
	Doc();
	~Doc();

	void Train();
	void GenerateDataset();
	void ClearDataset();
	void AddDataPoint(double point, double label);

	void Set(int kernel, int method);

	Dataset* dataset;
	KernelFunction* kernelFunction;
	Model* predictiveModel;
	Model* metaModel;

	LinearKernel kernelL;
	PolynomialKernel kernelP;
	RBFKernel kernelG;

	SVM* svm;

	RegularizationNetwork* regnet;
	Epsilon_SVM* esvm;

	MeanSquaredError mse;
};


class RegressionWidget : public QWidget
{
	Q_OBJECT

public:
	RegressionWidget(Doc* doc, FrameWidget* parent);

	void Draw();

protected:
	void paintEvent(QPaintEvent* event);
	void mousePressEvent(QMouseEvent* event);

	FrameWidget* frame;
	Doc* doc;
	QImage image;
};


class PropertyEdit : public QSlider
{
	Q_OBJECT

public:
	PropertyEdit(Property* prop, QWidget* parent, QLabel* label);
	~PropertyEdit();

public slots:
	void OnChanged(int value);

protected:
	void setLabelText();

	Property* property;
	QLabel* label;
};


class PropertiesWidget : public QWidget
{
	Q_OBJECT

public:
	PropertiesWidget(QWidget* parent = NULL);
	~PropertiesWidget();

	void Clear();
	void Append(Property* prop);

protected:
	void paintEvent(QPaintEvent* event);

	std::vector<Property*> property;
	std::vector<QLabel*> name;
	std::vector<PropertyEdit*> editor;
};


class PropertiesBar : public QScrollArea
{
	Q_OBJECT

public:
	PropertiesBar(QWidget* parent = NULL);

	void Clear();
	void Append(Property* prop);

protected:
	PropertiesWidget inner;
};


class FrameWidget : public QWidget
{
	Q_OBJECT

public:
	FrameWidget(QWidget* parent = NULL);

public slots:
	void OnClearDataset();
	void OnGenerateDataset();
	void OnChange(int value);
	void OnCompute();

protected:
	Doc doc;

	// widgets
	RegressionWidget wOutput;
	QPushButton wButtonClearDataset;
	QPushButton wButtonGenerateDataset;
	QLabel wLabelKernel;
	QComboBox wKernel;
	PropertiesBar wBarKernel;
	QLabel wLabelMethod;
	QComboBox wMethod;
	PropertiesBar wBarMethod;
	QPushButton wButtonCompute;
};


#endif
