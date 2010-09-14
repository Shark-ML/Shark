//===========================================================================
/*!
 *  \file regression.cpp
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


#include "regression.h"
#include <Array/Array.h>
#include <Rng/GlobalRng.h>
#include <math.h>


using namespace std;


NoisySinc::NoisySinc(double noise)
{
	this->noise = noise;

	dataDim = 1;
	targetDim = 1;
}

NoisySinc::~NoisySinc()
{}


bool NoisySinc::GetData(Array<double>& data, Array<double>& target, int count)
{
	data.resize(count, dataDim, false);
	target.resize(count, 1, false);

	int c;
	for (c = 0; c < count; c++)
	{
		double x = Rng::uni(-12.0, 12.0);
		data(c, 0) = x / 6.0;
		target(c, 0) = 2.0 * sin(x) / x + Rng::uni(-noise, noise) - 0.5;
	}

	return true;
}


////////////////////////////////////////////////////////////


Property::Property(const char* name, Model* model, unsigned int param, double m, double M, bool l)
{
	this->name = name;
	this->model = model;
	this->param = param;
	this->minVal = m;
	this->maxVal = M;
	this->logarithmic = l;
}


double Property::getValue()
{
	return model->getParameter(param);
}

void Property::setValue(double value)
{
	model->setParameter(param, value);
}


////////////////////////////////////////////////////////////


Doc::Doc()
		: kernelP(2, 1.0)
		, kernelG(1.0)
{
	svm = NULL;
	regnet = NULL;
	esvm = NULL;

	Array<double> a;
	dataset = new Dataset();
	dataset->CreateFromArrays(a, a, a, a);

	Set(2, 0);
}

Doc::~Doc()
{
	delete dataset;
}


void Doc::Set(int kernel, int method)
{
	if (kernel == 0)
	{
		kernelFunction = &kernelL;
	}
	else if (kernel == 1)
	{
		kernelFunction = &kernelP;
	}
	else if (kernel == 2)
	{
		kernelFunction = &kernelG;
	}

	if (method != -1)
	{
		if (svm != NULL) delete svm;
		svm = NULL;
		if (esvm != NULL) delete esvm;
		esvm = NULL;
		if (regnet != NULL) delete regnet;
		regnet = NULL;
	}
	if (method == 0)
	{
		svm = new SVM(kernelFunction, false);
		regnet = new RegularizationNetwork(svm, 0.1);
		predictiveModel = svm;
		metaModel = regnet;
	}
	else if (method == 1)
	{
		svm = new SVM(kernelFunction, false);
		esvm = new Epsilon_SVM(svm, 100.0, 0.1);
		predictiveModel = svm;
		metaModel = esvm;
	}
}

void Doc::Train()
{
	if (dataset->getTrainingData().ndim() == 0) return;

	if (regnet != NULL)
	{
		SVM_Optimizer opt;
		opt.init(*regnet);
		opt.optimize(*svm, dataset->getTrainingData(), dataset->getTrainingTarget(), true);
	}
	else if (esvm != NULL)
	{
		SVM_Optimizer opt;
		opt.init(*esvm);
		opt.optimize(*svm, dataset->getTrainingData(), dataset->getTrainingTarget(), true);
	}
}

void Doc::GenerateDataset()
{
	NoisySinc dist;
	delete dataset;
	dataset = new Dataset();
	dataset->CreateFromSource(dist, 20, 0);
}

void Doc::ClearDataset()
{
	delete dataset;
	Array<double> a;
	dataset = new Dataset();
	dataset->CreateFromArrays(a, a, a, a);
}

void Doc::AddDataPoint(double point, double label)
{
	Array<double> td = dataset->getTrainingData();
	Array<double> tt = dataset->getTrainingTarget();
	Array<double> a;
	int size = (td.ndim() == 0) ? 0 : td.dim(0);

	td.resize(size + 1, 1, true);
	tt.resize(size + 1, 1, true);
	td(size, 0) = point;
	tt(size, 0) = label;

	delete dataset;
	dataset = new Dataset();
	dataset->CreateFromArrays(td, tt, a, a);
}


////////////////////////////////////////////////////////////


RegressionWidget::RegressionWidget(Doc* doc, FrameWidget* parent)
		: QWidget(parent)
		, image(400, 400, QImage::Format_RGB32)
{
	this->doc = doc;
	this->frame = parent;
}


void RegressionWidget::Draw()
{
	Model* model = doc->predictiveModel;
	Dataset* dataset = doc->dataset;

	const Array<double>& input = dataset->getTrainingData();
	const Array<double>& target = dataset->getTrainingTarget();
	Array<int> value(400);
	int x, y;

	image.fill(0xffc0c0c0);
	QPainter painter(&image);
	painter.setPen(QPen(QColor(160, 160, 160)));
	painter.drawLine(100, 0, 100, 400);
	painter.drawLine(200, 0, 200, 400);
	painter.drawLine(300, 0, 300, 400);
	painter.drawLine(0, 100, 400, 100);
	painter.drawLine(0, 200, 400, 200);
	painter.drawLine(0, 300, 400, 300);
	painter.setPen(QPen(QColor(128, 128, 128)));
	for (x = 0; x < 400; x++)
	{
		double xx = (x - 200.0) * 0.06;
		double v = 2.0 * sin(xx) / xx - 0.5;
		value(x) = (int)Shark::round(200.0 - 100.0 * v);
	}
	for (x = 1; x < 400; x++) painter.drawLine(x - 1, value(x - 1), x, value(x));
	painter.setPen(QPen(QColor(0, 0, 0)));

	if (input.ndim() == 0)
	{
		painter.drawText(100, 200, "Click to insert a training example");
	}
	else
	{
		if (model != NULL)
		{
			// compute the model prediction
			Array<double> point(1);
			Array<double> output;
			for (x = 0; x < 400; x++)
			{
				point(0) = 0.01 * (x - 200);
				model->model(point, output);
				value(x) = (int)Shark::round(200.0 - 100.0 * output(0));
			}

			// output the solution
			for (x = 1; x < 400; x++) painter.drawLine(x - 1, value(x - 1), x, value(x));
		}

		// output the dataset
		int i, ic = input.dim(0);
		QBrush brush(QColor(255, 255, 0));
		for (i = 0; i < ic; i++)
		{
			x = (int)Shark::round(200.0 + 100.0 * input(i, 0));
			y = (int)Shark::round(200.0 - 100.0 * target(i, 0));
			painter.fillRect(x - 5, y - 5, 11, 11, brush);
			painter.drawRect(x - 5, y - 5, 11, 11);
			painter.drawLine(x - 1, y, x + 1, y);
			painter.drawLine(x, y - 1, x, y + 1);
		}
	}

	// inform QT that the widget needs to be redrawn
	update();
}

void RegressionWidget::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);
	painter.drawImage(event->rect(), image);
}

void RegressionWidget::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton)
	{
		double x = 0.01 * (event->x() - 200);
		double y = 0.01 * (200 - event->y());
		doc->AddDataPoint(x, y);
		frame->OnCompute();
	}
}


////////////////////////////////////////////////////////////


PropertyEdit::PropertyEdit(Property* prop, QWidget* parent, QLabel* label)
		: QSlider(Qt::Horizontal, parent)
{
	setFixedSize(120, 20);

	property = prop;
	this->label = label;

	setRange(0, 100);
	setTickPosition(QSlider::TicksBelow);
	setTickInterval(10);

	double v = property->getValue();
	if (property->logarithmic)
	{
		setValue((int)(100.0 *(log(v) - log(property->minVal)) / (log(property->maxVal) - log(property->minVal))));
	}
	else
	{
		setValue((int)(100.0 *(v - property->minVal) / (property->maxVal - property->minVal)));
	}

	connect(this, SIGNAL(valueChanged(int)), this, SLOT(OnChanged(int)));

	setLabelText();
}

PropertyEdit::~PropertyEdit()
{}


void PropertyEdit::OnChanged(int value)
{
	if (property->logarithmic)
	{
		double v = exp(0.01 * value * (log(property->maxVal) - log(property->minVal)) + log(property->minVal));
		property->setValue(v);
	}
	else
	{
		double v = 0.01 * value * (property->maxVal - property->minVal) + property->minVal;
		property->setValue(v);
	}
	setLabelText();
}

void PropertyEdit::setLabelText()
{
	if (label != NULL)
	{
		QString text = property->name;
		QString num; num.setNum(property->getValue());
		text += ": ";
		text += num;
		label->setText(text);
	}
}


////////////////////////////////////////////////////////////


PropertiesWidget::PropertiesWidget(QWidget* parent)
		: QWidget(parent)
{}

PropertiesWidget::~PropertiesWidget()
{}


void PropertiesWidget::Clear()
{
	int i, ic = property.size();
	for (i = 0; i < ic; i++)
	{
		delete editor[i];
		delete name[i];
		delete property[i];
	}
	property.clear();
	name.clear();
	editor.clear();

	resize(280, 10);
}

void PropertiesWidget::Append(Property* prop)
{
	int index = property.size();
	resize(280, 21*index + 22);

	QString text = prop->name;
	QString num; num.setNum(prop->getValue());
	text += ": ";
	text += num;
	QLabel* l = new QLabel(text, this);
	l->setGeometry(5, 21*index + 1, 145, 20);
	l->show();

	PropertyEdit* e = new PropertyEdit(prop, this, l);
	e->setGeometry(155, 21*index + 1, 120, 20);
	e->show();

	property.push_back(prop);
	name.push_back(l);
	editor.push_back(e);

	update();
}

void PropertiesWidget::paintEvent(QPaintEvent* event)
{
	QWidget::paintEvent(event);

	QPainter painter(this);

	int i, ic = property.size();
	for (i = 0; i <= ic; i++)
	{
		painter.setPen(QPen(Qt::gray));
		painter.drawLine(0, 21*i, 280, 21*i);
		painter.setPen(QPen(Qt::black));
	}
}


////////////////////////////////////////////////////////////


PropertiesBar::PropertiesBar(QWidget* parent)
		: QScrollArea(parent)
		, inner(this)
{}


void PropertiesBar::Clear()
{
	inner.Clear();
}

void PropertiesBar::Append(Property* prop)
{
	inner.Append(prop);
}


////////////////////////////////////////////////////////////


FrameWidget::FrameWidget(QWidget* parent)
		: QWidget(parent)
		, wOutput(&doc, this)
		, wButtonClearDataset("clear dataset", this)
		, wButtonGenerateDataset("generate dataset", this)
		, wLabelKernel("kernel", this)
		, wKernel(this)
		, wBarKernel(this)
		, wLabelMethod("method", this)
		, wMethod(this)
		, wBarMethod(this)
		, wButtonCompute("compute solution !", this)
{
	setWindowTitle("ReClaM regression example");
	setFixedSize(700, 400);

	wOutput.setGeometry(300, 0, 400, 400);
	wButtonClearDataset.setGeometry(10, 10, 280, 20);
	wButtonGenerateDataset.setGeometry(10, 40, 280, 20);
	wLabelKernel.setGeometry(10, 70, 280, 20);
	wKernel.setGeometry(10, 90, 280, 20);
	wBarKernel.setGeometry(10, 110, 280, 100);
	wLabelMethod.setGeometry(10, 220, 280, 20);
	wMethod.setGeometry(10, 240, 280, 20);
	wBarMethod.setGeometry(10, 260, 280, 100);
	wButtonCompute.setGeometry(10, 370, 280, 20);

	wKernel.addItem("linear kernel");
	wKernel.addItem("polynomial kernel");
	wKernel.addItem("Gaussian kernel");

	wMethod.addItem("Regularization Network");
	wMethod.addItem("epsilon-SVM");

	QObject::connect(&wButtonClearDataset, SIGNAL(clicked()), this, SLOT(OnClearDataset()));
	QObject::connect(&wButtonGenerateDataset, SIGNAL(clicked()), this, SLOT(OnGenerateDataset()));
	QObject::connect(&wKernel, SIGNAL(currentIndexChanged(int)), this, SLOT(OnChange(int)));
	QObject::connect(&wMethod, SIGNAL(currentIndexChanged(int)), this, SLOT(OnChange(int)));
	QObject::connect(&wButtonCompute, SIGNAL(clicked()), this, SLOT(OnCompute()));

	wKernel.setCurrentIndex(2);
	wMethod.setCurrentIndex(0);
}


void FrameWidget::OnClearDataset()
{
	doc.ClearDataset();
	OnCompute();
}

void FrameWidget::OnGenerateDataset()
{
	doc.GenerateDataset();
	OnCompute();
}

void FrameWidget::OnChange(int value)
{
	value = 0;

	int kernel = wKernel.currentIndex();
	int method = wMethod.currentIndex();

	doc.Set(kernel, method);

	wBarKernel.Clear();
	if (kernel == 0)
	{}
	else if (kernel == 1)
	{
		wBarKernel.Append(new Property("degree", &doc.kernelP, 0, 2.0, 7.0));
		wBarKernel.Append(new Property("offset", &doc.kernelP, 1, 0.0, 10.0));
	}
	else if (kernel == 2)
	{
		wBarKernel.Append(new Property("gamma", &doc.kernelG, 0, 0.01, 100.0, true));
	}

	wBarMethod.Clear();
	if (method == 0)
	{
		wBarMethod.Append(new Property("gamma", doc.regnet, 0, 0.0001, 10.00, true));
	}
	else if (method == 1)
	{
		wBarMethod.Append(new Property("C", doc.esvm, 0, 0.1, 10000.0, true));
		wBarMethod.Append(new Property("epsilon", doc.esvm, 1, 0.001, 1.0, true));
	}

	OnCompute();
}

void FrameWidget::OnCompute()
{
	QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
	doc.Train();
	wOutput.Draw();
	QApplication::restoreOverrideCursor();
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
	catch (const SharkException& e)
	{
		cout << "SharkException: " << e.what() << endl;
	}

	return ret;
}
