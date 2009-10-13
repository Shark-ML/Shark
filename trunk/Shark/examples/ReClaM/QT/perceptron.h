//===========================================================================
/*!
 *  \file perceptron.h
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
 *      $RCSfile: perceptron.h,v $<BR>
 *      $Revision: 1.2 $<BR>
 *      $Date: 2007/09/21 18:02:01 $
 *
 *  \par Changes:
 *      $Log: perceptron.h,v $
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


#ifndef _perceptron_H_
#define _perceptron_H_


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
#include <QLineEdit>

#include <ReClaM/Dataset.h>
#include <ReClaM/LinearModel.h>


class FrameWidget;


class SeparableDist : public DataSource
{
public:
	SeparableDist();
	~SeparableDist();

	bool GetData(Array<double>& data, Array<double>& target, int count);
};


class Doc
{
public:
	Doc();
	~Doc();

	void Reset();
	void TrainStep();
	void GenerateDataset(int points);

	Dataset* dataset;
	LinearFunction predictor;

	Array<double> w;
	Array<double> w_old;
	int update;

	SeparableDist dist;
};


class ClassificationWidget : public QWidget
{
	Q_OBJECT

public:
	ClassificationWidget(Doc* doc, FrameWidget* parent);

	void Draw();

protected:
	void paintEvent(QPaintEvent* event);

	FrameWidget* frame;
	Doc* doc;
	QImage image;
};


class FrameWidget : public QWidget
{
	Q_OBJECT

public:
	FrameWidget(QWidget* parent = NULL);

public slots:
	void OnGenerateDataset();
	void OnStep();
	void OnReset();

protected:
	Doc doc;

	// widgets
	ClassificationWidget wOutput;
	QLineEdit wLineEditDatasetSize;
	QPushButton wButtonGenerateDataset;
	QPushButton wButtonStep;
	QPushButton wButtonReset;
};


#endif
