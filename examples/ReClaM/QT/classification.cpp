//===========================================================================
/*!
 *  \file classification.cpp
 *
 *  \author  T. Glasmachers, C. Igel
 *  \date    2007, 2010
 *
 *  \par Copyright (c) 1999-2010:
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
 *      $RCSfile: classification.cpp,v $<BR>
 *      $Revision: 1.3 $<BR>
 *      $Date: 2007/10/18 15:29:33 $
 *
 *  \par Changes:
 *      $Log: classification.cpp,v $
 *      Revision 1.3  2007/10/18 15:29:33  christian_igel
 *      changed exception
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


#include "classification.h"
#include <Array/Array.h>
#include <Rng/GlobalRng.h>
#include <ReClaM/ClassificationError.h>
#include <math.h>


using namespace std;


OverlappingDist::OverlappingDist()
{
	dataDim = 2;
	targetDim = 1;
}

OverlappingDist::~OverlappingDist()
{
}


bool OverlappingDist::GetData(Array<double>& data, Array<double>& target, int count)
{
	data.resize(count, dataDim, false);
	target.resize(count, 1, false);

	int c;
	for (c = 0; c < count; c++)
	{
		if (Rng::discrete(0, 1) == 1)
		{
			// positive class
			double x = Rng::uni(0.0, 1.0);
			double radius = 1.3 * (1.0 - x * x);
			double angle = Rng::uni(0.0, 6.283185307179586477);
			data(c, 0) = 2.5 + radius * cos(angle);
			data(c, 1) = 2.0 + radius * sin(angle);
			target(c, 0) = 1.0;
		}
		else
		{
			// negative class
			double x = Rng::uni(0.0, 1.0);
			double radius = 1.3 * (1.0 - x * x);
			double angle = Rng::uni(0.0, 6.283185307179586477);
			data(c, 0) = 1.5 + radius * cos(angle);
			data(c, 1) = 2.0 + radius * sin(angle);
			target(c, 0) = -1.0;
		}
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
	km = NULL;
	knn = NULL;
	csvm = NULL;
	perceptron = NULL;

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
		if (km != NULL) delete km;
		km = NULL;
		if (knn != NULL) delete knn;
		knn = NULL;
		if (csvm != NULL) delete csvm;
		csvm = NULL;
		if (perceptron != NULL) delete perceptron;
		perceptron = NULL;
	}
	if (method == 0)
	{
		perceptron = new Perceptron();
		svm = new SVM(kernelFunction, false);
		predictiveModel = svm;
		metaModel = svm;
	}
	else if (method == 1)
	{
		knn = new KernelNearestNeighbor(kernelFunction, 1);
		predictiveModel = knn;
		metaModel = knn;
	}
	else if (method == 2)
	{
		km = new KernelMeanClassifier(kernelFunction);
		predictiveModel = km;
		metaModel = km;
	}
	else if (method == 3)
	{
		svm = new SVM(kernelFunction, false);
		csvm = new C_SVM(svm, 100.0, 100.0);
		predictiveModel = svm;
		metaModel = csvm;
	}
}

void Doc::Train()
{
	if (dataset->getTrainingData().ndim() == 0) return;

	if (perceptron != NULL)
	{
		perceptron->optimize(*svm, dataset->getTrainingData(), dataset->getTrainingTarget());
	}
	else if (knn != NULL)
	{
		knn->SetPoints(dataset->getTrainingData(), dataset->getTrainingTarget());
	}
	else if (km != NULL)
	{
		km->SetPoints(dataset->getTrainingData(), dataset->getTrainingTarget());
	}
	else if (csvm != NULL)
	{
		SVM_Optimizer opt;
		opt.init(*csvm);
		opt.optimize(*svm, dataset->getTrainingData(), dataset->getTrainingTarget(), true);
	}
}

void Doc::GenerateDataset()
{
	OverlappingDist dist;
	delete dataset;
	dataset = new Dataset();
	dataset->CreateFromSource(dist, 50, 0);
}

void Doc::ClearDataset()
{
	delete dataset;
	Array<double> a;
	dataset = new Dataset();
	dataset->CreateFromArrays(a, a, a, a);
}

void Doc::AddDataPoint(Array<double> point, double label)
{
	Array<double> td = dataset->getTrainingData();
	Array<double> tt = dataset->getTrainingTarget();
	Array<double> a;
	int size = (td.ndim() == 0) ? 0 : td.dim(0);
	int d, dim = (td.ndim() == 0) ? 2 : td.dim(1);

	td.resize(size + 1, dim, true);
	tt.resize(size + 1, 1, true);
	for (d = 0; d < dim; d++) td(size, d) = point(d);
	tt(size, 0) = label;

	delete dataset;
	dataset = new Dataset();
	dataset->CreateFromArrays(td, tt, a, a);
}


////////////////////////////////////////////////////////////


ClassificationWidget::ClassificationWidget(Doc* doc, FrameWidget* parent)
		: QWidget(parent)
		, image(400, 400, QImage::Format_RGB32)
{
	this->doc = doc;
	this->frame = parent;
}


void ClassificationWidget::Draw(bool drawSoft, bool drawBound, bool drawCross, bool drawShade)
{
	Model* model = doc->predictiveModel;
	Dataset* dataset = doc->dataset;

	const Array<double>& input = dataset->getTrainingData();
	const Array<double>& target = dataset->getTrainingTarget();

	QPainter painter(&image);

	if (input.ndim() == 0)
	{
		image.fill(0xffc0c0c0);
		painter.drawText(100, 180, "Click left to insert a positive example");
		painter.drawText(100, 220, "Click right to insert a negative example");
	}
	else
	{
		int x, y;
		SVM* svm = dynamic_cast<SVM*>(model);

		if (model != NULL)
		{
			// compute the model prediction
			Array<double> point(2);
			Array<double> output;
			Array<double> value(400, 400);
			for (y = 0; y < 400; y++)
			{
				point(1) = 0.01 * y;
				for (x = 0; x < 400; x++)
				{
					point(0) = 0.01 * x;
					model->model(point, output);
					value(x, y) = output(0);
				}
			}

			// output the solution
			for (y = 0; y < 400; y++)
			{
				for (x = 0; x < 400; x++)
				{
					bool b0 = false;
					bool b1 = false;

					if (x > 0 && x < 399 && y > 0 && y < 399)
					{
						double v = value(x, y);
						if (v * value(x - 1, y) <= 0.0 || v * value(x + 1, y) <= 0.0 || v * value(x, y - 1) <= 0.0 || v * value(x, y + 1) <= 0.0) b0 = true;
						if (svm != NULL)
						{
							v = value(x, y) - 1.0;
							if (v *(value(x - 1, y) - 1.0) <= 0.0 || v *(value(x + 1, y) - 1.0) <= 0.0 || v *(value(x, y - 1) - 1.0) <= 0.0 || v *(value(x, y + 1) - 1.0) <= 0.0) b1 = true;
							v = value(x, y) + 1.0;
							if (v *(value(x - 1, y) + 1.0) <= 0.0 || v *(value(x + 1, y) + 1.0) <= 0.0 || v *(value(x, y - 1) + 1.0) <= 0.0 || v *(value(x, y + 1) + 1.0) <= 0.0) b1 = true;
						}
					}

					painter.setPen(Qt::white);
					if(drawSoft) {
						if(!drawBound) b0 = b1 = 0;
						if(b0) {
							if(drawShade) painter.setPen(Qt::white);
							else painter.setPen(Qt::black);
						} else if(b1) painter.setPen(Qt::yellow);
						else if(drawShade) painter.setPen(QColor(127 + tanh(value(x,y))*128, 127 - tanh(value(x,y))*128, 0));
					} else {
						if(!drawBound) b0 = b1 = 0;
						if(b0) {
							if(drawShade) painter.setPen(Qt::white);
							else painter.setPen(Qt::black);
						} else if(b1) painter.setPen(Qt::yellow);
						else if(drawShade) {
							if(value(x, y) > 0) painter.setPen(QColor(255, 0, 0));
							else painter.setPen(QColor(0, 255, 0));
						}
					}
					painter.drawPoint(x, y);
				}
			}
		}
		else
		{
			image.fill(0xffc0c0c0);
		}

		
		
		// output the dataset
		int HalfBaseSize = 15;
		int BaseSize = 2 * HalfBaseSize;
		
		unsigned i, ic = input.dim(0);
		QColor col;
		
		// circles around support vectors
		if (drawBound && (svm != NULL)) {
			Array<double> ex = svm->getPoints();
			for(i = 0; i < ex.dim(0); i++) {
				x = (int)(100.0 * ex(i, 0));
				y = (int)(100.0 * ex(i, 1));
				
				painter.setPen(Qt::yellow);
				painter.setBrush(Qt::yellow);
				painter.drawEllipse(x-HalfBaseSize-1, y-HalfBaseSize-1, BaseSize+2, BaseSize+2);
			}
		}
		
		
		for (i = 0; i < ic; i++)
		{
			x = (int)(100.0 * input(i, 0));
			y = (int)(100.0 * input(i, 1));
			
				
			if (target(i, 0) > 0.0) col = Qt::darkRed; 
			else col = Qt::darkGreen; 
			
			painter.setBrush(col);
			painter.setPen(col);
			painter.drawEllipse(x-HalfBaseSize, y-HalfBaseSize, BaseSize, BaseSize);
			painter.setBrush(Qt::white);
			painter.setPen(Qt::white);
			if(drawCross) {
				painter.drawRect(x-HalfBaseSize + 4, y-3, BaseSize - 8, 6);
				if(target(i, 0) <- 0.) painter.drawRect(x-3, y-HalfBaseSize+4, 6, BaseSize-8);
			}
		}

	
	}

	// inform QT that the widget needs to be redrawn
	update();
}

void ClassificationWidget::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);
	painter.drawImage(event->rect(), image);
}

void ClassificationWidget::mousePressEvent(QMouseEvent* event)
{
	Array<double> point(2);
	point(0) = 0.01 * event->x();
	point(1) = 0.01 * event->y();

	if (event->button() == Qt::LeftButton)
	{
		doc->AddDataPoint(point, 1.0);
		frame->OnCompute();
	}
	else if (event->button() == Qt::RightButton)
	{
		doc->AddDataPoint(point, -1.0);
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
{
}


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
{
}

PropertiesWidget::~PropertiesWidget()
{
}


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
{
}


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
		, wButtonCompute("compute solution", this)
		, wButtonSave("save", this)
		, wCheckBound("bound", this)
		, wCheckSoft("soft", this)
		, wCheckShade("shade", this)
		, wCheckCross("cross", this)
{
	setWindowTitle("ReClaM classification example");
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
	wButtonCompute.setGeometry(10, 370, 120, 20);
	wButtonSave.setGeometry(135, 370, 40, 20);
	wCheckBound.setGeometry(175, 365, 60, 20);
	wCheckSoft.setGeometry(175, 380, 60, 20);
	wCheckCross.setGeometry(236, 365, 60, 20);
	wCheckShade.setChecked(true);
	wCheckShade.setGeometry(236, 380, 60, 20);
	
	wKernel.addItem("linear kernel");
	wKernel.addItem("polynomial kernel");
	wKernel.addItem("Gaussian kernel");

	wMethod.addItem("Perceptron");
	wMethod.addItem("Nearest Neighbor");
	wMethod.addItem("Mean Classifier");
	wMethod.addItem("C-SVM");

	QObject::connect(&wButtonClearDataset, SIGNAL(clicked()), this, SLOT(OnClearDataset()));
	QObject::connect(&wButtonGenerateDataset, SIGNAL(clicked()), this, SLOT(OnGenerateDataset()));
	QObject::connect(&wKernel, SIGNAL(currentIndexChanged(int)), this, SLOT(OnChange(int)));
	QObject::connect(&wMethod, SIGNAL(currentIndexChanged(int)), this, SLOT(OnChange(int)));
	QObject::connect(&wButtonCompute, SIGNAL(clicked()), this, SLOT(OnCompute()));
	QObject::connect(&wButtonSave, SIGNAL(clicked()), this, SLOT(OnSave()));
	QObject::connect(&wCheckBound, SIGNAL(stateChanged(int)), this, SLOT(OnToggle()));
	QObject::connect(&wCheckSoft, SIGNAL(stateChanged(int)), this, SLOT(OnToggle()));
	QObject::connect(&wCheckCross, SIGNAL(stateChanged(int)), this, SLOT(OnToggle()));
	QObject::connect(&wCheckShade, SIGNAL(stateChanged(int)), this, SLOT(OnToggle()));
	
	wKernel.setCurrentIndex(0);
	wMethod.setCurrentIndex(1);
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
	{
	}
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
	}
	else if (method == 1)
	{
		wBarMethod.Append(new Property("#neighbors", doc.knn, 0, 1.0, 15.0));
	}
	else if (method == 2)
	{
	}
	else if (method == 3)
	{
		wBarMethod.Append(new Property("C", doc.csvm, 0, 0.001, 10000.0, true));
	}

	OnCompute();
}

void FrameWidget::OnCompute()
{
	QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
	doc.Train();
	wOutput.Draw(wCheckSoft.isChecked(), wCheckBound.isChecked(), wCheckCross.isChecked(), wCheckShade.isChecked());
	QApplication::restoreOverrideCursor();
}

void FrameWidget::OnSave()
{
	QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"),
													"untitled.png",
													tr("Images (*.png *.xpm *.jpg)"));
	wOutput.Save(fileName.toStdString().c_str());
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
