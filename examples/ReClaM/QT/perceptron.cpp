//===========================================================================
/*!
 *  \file perceptron.cpp
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
 *      $RCSfile: perceptron.cpp,v $<BR>
 *      $Revision: 1.3 $<BR>
 *      $Date: 2007/10/18 15:30:40 $
 *
 *  \par Changes:
 *      $Log: perceptron.cpp,v $
 *      Revision 1.3  2007/10/18 15:30:40  christian_igel
 *      exception changed
 *
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


#include "perceptron.h"
#include <Array/Array.h>
#include <Rng/GlobalRng.h>
#include <ReClaM/ClassificationError.h>
#include <math.h>


using namespace std;


SeparableDist::SeparableDist()
{
	dataDim = 2;
	targetDim = 1;
}

SeparableDist::~SeparableDist()
{
}


bool SeparableDist::GetData(Array<double>& data, Array<double>& target, int count)
{
	data.resize(count, dataDim, false);
	target.resize(count, 1, false);

	int c;
	for (c = 0; c < count; c++)
	{
		double x = Rng::uni(-0.5, 0.5);
		double y = Rng::uni(-0.5, 0.5);
		double label = (x > 0.0) ? 1.0 : -1.0;
		data(c, 0) = x;
		data(c, 1) = y;
		target(c, 0) = label;
	}

	return true;
}


////////////////////////////////////////////////////////////


Doc::Doc()
: predictor(2)
{
	dataset = NULL;
	GenerateDataset(20);
	Reset();
}

Doc::~Doc()
{
	delete dataset;
}


void Doc::Reset()
{
	w.resize(2, false);
	w_old.resize(2, false);
	w = 0.0;
	w_old = 0.0;
	update = -1;

	predictor.setParameter(0, 0.0);
	predictor.setParameter(1, 0.0);
}


void Doc::TrainStep()
{
	const Array<double>& x = dataset->getTrainingData();
	const Array<double>& y = dataset->getTrainingTarget();

	int ic = x.dim(0);
	int i = (update + 1) % ic;

	int p, pc = 2;
	for (p=0; p<pc; p++) predictor.setParameter(p, w(p));

	Array<double> out;
	do
	{
		predictor.model(x[i], out);
		if (out(0) * y(i, 0) <= 0.0)
		{
			for (p=0; p<pc; p++)
			{
				w_old(p) = w(p);
				w(p) += y(i, 0) * x(i, p);
				predictor.setParameter(p, w(p));
				update = i;
			}
			break;
		}
		i = (i+1) % ic;
	}
	while (i != update);
}

void Doc::GenerateDataset(int points)
{
	if (dataset != NULL) delete dataset;
	dataset = new Dataset(dist, points, 0);
}


////////////////////////////////////////////////////////////


ClassificationWidget::ClassificationWidget(Doc* doc, FrameWidget* parent)
: QWidget(parent)
, image(400, 400, QImage::Format_RGB32)
{
	this->doc = doc;
	this->frame = parent;
}


void ClassificationWidget::Draw()
{
	LinearFunction* model = &doc->predictor;
	Dataset* dataset = doc->dataset;

	const Array<double>& input = dataset->getTrainingData();
	const Array<double>& target = dataset->getTrainingTarget();

	int x, y;

	// compute the model prediction
	Array<double> point(2);
	Array<double> output;
	Array<double> value(400, 400);
	for (y = 0; y < 400; y++)
	{
		point(1) = 0.005 * (y - 200);
		for (x = 0; x < 400; x++)
		{
			point(0) = 0.005 * (x - 200);
			model->model(point, output);
			value(x, y) = output(0);
		}
	}

	// output the solution
	for (y = 0; y < 400; y++)
	{
		for (x = 0; x < 400; x++)
		{
			if (x == 200 || y == 200) image.setPixel(x, y, 0xff000000);
			else
			{
				if (value(x, y) > 0.0) image.setPixel(x, y, 0xff000080);
				else if (value(x, y) < 0.0) image.setPixel(x, y, 0xff800000);
				else image.setPixel(x, y, 0xff404040);
			}
		}
	}

	// output old vector, new vector and update vector
	int i;
	for (i=0; i<800; i++)
	{
		x = (int)(200 + 0.25 * i * doc->w_old(0));
		y = (int)(200 + 0.25 * i * doc->w_old(1));
		if (x >= 0 && y >= 0 && x < 400 && y < 400) image.setPixel(x, y, 0xffffffff);
		x = (int)(200 + 0.25 * i * doc->w(0));
		y = (int)(200 + 0.25 * i * doc->w(1));
		if (x >= 0 && y >= 0 && x < 400 && y < 400) image.setPixel(x, y, 0xffffff00);
	}
	if (doc->update != -1)
	{
		for (i=0; i<800; i++)
		{
			x = (int)(200 + 0.25 * i * input(doc->update, 0));
			y = (int)(200 + 0.25 * i * input(doc->update, 1));
			if (x >= 0 && y >= 0 && x < 400 && y < 400) image.setPixel(x, y, 0xff808080);
		}
	}

	// output the dataset
	int ic = input.dim(0);
	unsigned int color;
	for (i = 0; i < ic; i++)
	{
		x = (int)(200.0 * input(i, 0) + 200);
		y = (int)(200.0 * input(i, 1) + 200);
		if (target(i, 0) > 0.0) color = 0xff4040ff;
		else color = 0xffff4040;
		image.setPixel(x, y, color);
		if (x < 399) image.setPixel(x + 1, y, color);
		if (x > 0) image.setPixel(x - 1, y, color);
		if (y < 399) image.setPixel(x, y + 1, color);
		if (y > 0) image.setPixel(x, y - 1, color);
	}

	// inform QT that the widget needs to be redrawn
	update();
}

void ClassificationWidget::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);
	painter.drawImage(event->rect(), image);
}


////////////////////////////////////////////////////////////


FrameWidget::FrameWidget(QWidget* parent)
		: QWidget(parent)
		, wOutput(&doc, this)
		, wLineEditDatasetSize("20", this)
		, wButtonGenerateDataset("generate dataset", this)
		, wButtonStep("Update", this)
		, wButtonReset("Reset", this)
{
	setWindowTitle("Perceptron example");
	setFixedSize(700, 400);

	wOutput.setGeometry(300, 0, 400, 400);
	wLineEditDatasetSize.setGeometry(10, 10, 280, 20);
	wButtonGenerateDataset.setGeometry(10, 40, 280, 20);
	wButtonStep.setGeometry(10, 70, 280, 20);
	wButtonReset.setGeometry(10, 100, 280, 20);

	QObject::connect(&wButtonGenerateDataset, SIGNAL(clicked()), this, SLOT(OnGenerateDataset()));
	QObject::connect(&wButtonStep, SIGNAL(clicked()), this, SLOT(OnStep()));
	QObject::connect(&wButtonReset, SIGNAL(clicked()), this, SLOT(OnReset()));
}


void FrameWidget::OnGenerateDataset()
{
	bool ok;
	int sz = wLineEditDatasetSize.text().toInt(&ok);
	if (! ok) sz = 20;
	doc.GenerateDataset(sz);
	doc.Reset();
	wOutput.Draw();
}

void FrameWidget::OnStep()
{
	doc.TrainStep();
	wOutput.Draw();
}

void FrameWidget::OnReset()
{
	doc.Reset();
	wOutput.Draw();
}


////////////////////////////////////////////////////////////


int main(int argc, char** argv)
{
	int ret = EXIT_FAILURE;
	try
	{
		QApplication app(argc, argv);

		FrameWidget frame;
		frame.show();

		ret = app.exec();
	}
	catch (SharkException e)
	{
		cout << "SharkException: " << e.what() << endl;
	}

	return ret;
}
