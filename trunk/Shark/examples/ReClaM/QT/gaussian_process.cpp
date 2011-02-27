//===========================================================================
/*!
 *  \file gaussian_process.cpp
 *
 *  \author  C. Igel
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
 *      $RCSfile: gaussian_process.cpp,v $<BR>
 *      $Revision: 1.5 $<BR>
 *      $Date: 2007/10/22 15:26:14 $
 *
 *  \par Changes:
 *      $Log: gaussian_process.cpp,v $
 *      Revision 1.5  2007/10/22 15:26:14  glasmtbl
 *      *** empty log message ***
 *
 *      Revision 1.4  2007/10/19 15:00:32  glasmtbl
 *      *** empty log message ***
 *
 *      Revision 1.3  2007/09/21 18:02:01  leninga
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


#include "gaussian_process.h"


////////////////////////////////////////////////////////////


CISlider::CISlider(char *title, double init, double l, double r, double step)
{
  	slider = new QwtSlider(this, Qt::Horizontal, QwtSlider::TopScale, QwtSlider::BgTrough);
// 	slider = new QwtSlider(this, Qt::Horizontal, QwtSlider::Top, QwtSlider::BgTrough);
	slider->setThumbWidth(10);
	legend = new QLabel();
	setIt(init);

	text = QString(title);
	legend = new QLabel(text + QString(": ") + QString::number(init));
	if (!step) step = (r - l) / 100.;
	slider->setRange(l, r, step);
	slider->setValue(init);
	slider->setFont(QFont(slider->font().family(), 10));
	legend->setFont(QFont(legend->font().family(), 12));
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(legend);
	layout->addWidget(slider);

	connect(slider, SIGNAL(valueChanged(double)), this, SLOT(setIt(double)));

	setLayout(layout);
}

QwtSlider* CISlider::slide()
{
	return slider;
}

void CISlider::setValue(double x)
{
	slider->setValue(x);
}

void CISlider::setIt(double x)
{
	legend->setText(text + ": " + QString::number(x));
	emit valuesChanged(x);
}


////////////////////////////////////////////////////////////


CILogSlider::CILogSlider(char *title, double init, double l, double r) :
		CISlider(title, init, l, r)
{
	legend->setText(text + QString(": ") + QString::number(init));
	double step = (log10(r) - log10(l)) / 100.;
	slider->setScaleEngine(new QwtLog10ScaleEngine);
	slider->setScaleMaxMinor(10);
	slider->setRange(log10(l), log10(r), step);
	slider->setScale(l, r);
	slider->setValue(log10(init));
	update();
}

void CILogSlider::setValue(double x)
{
	slider->setValue(log10(x));
}

void CILogSlider::setIt(double x)
{
	legend->setText(text + ": " + QString::number(pow(10., x)));
	emit valueChanged(pow(10., x));
}


////////////////////////////////////////////////////////////


PlotWidget::PlotWidget(const char *title)
{
	setTitle(title);

	insertLegend(new QwtLegend(), QwtPlot::RightLegend);
//	legend()->setDisplayPolicy(QwtLegend::AutoIdentifier, 7);
// 	legend()->setDisplayPolicy(QwtLegend::Auto, 7);

	sym = 1;
	ps  = 10;
	hideItem = false;

	// finally, refresh the plot
	replot();
	return;
}

QwtPlotCurve* PlotWidget::addLine(const char *title, const Array<double> &x, const Array<double> &y,
								  const QPen &pen)
{
	return addLine(title, x.begin(), y.begin(), x.dim(0), pen);
}

QwtPlotCurve* PlotWidget::addLine(const char *title, const double *x, const double *y, unsigned n,
								  const QPen &pen)
{
	QwtPlotCurve *c = new QwtPlotCurve(title);
	c->setStyle(QwtPlotCurve::Lines);
	return add(c, x, y, n, pen);
}

QwtPlotCurve* PlotWidget::addDash(const char *title, const Array<double> &x, const Array<double> &y,
								  const QPen &pen)
{
	return addDash(title, x.begin(), y.begin(), x.dim(0), pen);
}

QwtPlotCurve* PlotWidget::addDash(const char *title, const double *x, const double *y, unsigned n,
								  const QPen &pen)
{
	QwtPlotCurve *c = new QwtPlotCurve(title);
	QPen newPen;
	newPen = pen;
	newPen.setStyle(Qt::DashLine);
	c->setStyle(QwtPlotCurve::Lines);
	return add(c, x, y, n, newPen);
}

QwtPlotCurve* PlotWidget::addDots(const char *title, const Array<double> &x, const Array<double> &y,
								  const QPen &pen)
{
	return addDots(title, x.begin(), y.begin(), x.dim(0), pen);
}

QwtPlotCurve* PlotWidget::addDots(const char *title, const double *x, const double *y, unsigned n,
								  const QPen &pen)
{
	QwtPlotCurve *c = new QwtPlotCurve(title);
	QwtSymbol symbol (QwtSymbol::Style(sym), QBrush(pen.color(), Qt::SolidPattern), pen, QSize(ps, ps));
	c->setStyle(QwtPlotCurve::Dots);
	c->setSymbol(symbol);
	sym++;
	add(c, x, y, n, pen);
	return c;
}

void PlotWidget::hideItems()
{
	hideItem = true;
}

void PlotWidget::showItems()
{
	hideItem = false;
}

QwtPlotCurve* PlotWidget::add(QwtPlotCurve *c, const double *x, const double *y, unsigned n,
								  const QPen &pen)
{
	c->setRawData(x, y, n);
	c->setPen(pen);
	c->setRenderHint(QwtPlotItem::RenderAntialiased);
	c->attach(this);
	crvs.push_back(c);

	if (hideItem)
	{
		legend()->remove(c);
	}

	replot();
	return c;
}


////////////////////////////////////////////////////////////


GPWidget::GPWidget()
		: k(1.0)
		, model(&k, false)
{
	double sigma   = 1;
	double betaInv = 0.1;

	noise   = 0.01;
	N       = 10;

	// generate and train Gaussian Process
	gp = new GaussianProcess(&model, betaInv);

	generateTrainingData();
	train();

	// compute data points for plot
	computePlot();

	// create plot
	plot = new PlotWidget("Gaussian Process");
	plot->addLine("model", x, y1, 1001, QPen(Qt::blue, 2));
	plot->addDash("sd", x, l, 1001);
	plot->hideItems();
	plot->addDash("sd", x, u, 1001);
	plot->showItems();
	plot->addLine("f", x, y2, 1001, QPen(Qt::green));
	trainingCurve = plot->addDots("train", inTrain, targetTrain, QPen(Qt::yellow));

	QPushButton *stepOnEvidence = new QPushButton(tr("gradient step on evidence"), this);
        stepOnEvidence->setGeometry(62, 40, 75, 30);
	connect(stepOnEvidence, SIGNAL(clicked()), this, SLOT(gradientStep()));

	CISlider *ciSlideN = new CISlider(strdup("number of training samples"), N, 1, 51, 1.);
	QwtSlider *sliderN = ciSlideN->slide();
	sliderN->setScale(1, 51, 10);
	connect(sliderN, SIGNAL(valueChanged(double)), this, SLOT(setN(double)));
	connect(sliderN, SIGNAL(sliderPressed()), this, SLOT(setN()));

	ciSlideBI = new CILogSlider(strdup("negative precision"), betaInv, 1.e-14, 1.e2);
	connect(ciSlideBI, SIGNAL(valueChanged(double)), this, SLOT(setBI(double)));

	ciSlideS = new CILogSlider(strdup("kernel width"), sigma, 1.e-3, 1.e2);
	connect(ciSlideS, SIGNAL(valueChanged(double)), this, SLOT(setS(double)));

	CILogSlider *ciSlideE = new CILogSlider(strdup("noise"), noise, 1.e-8, 1e+2);
	connect(ciSlideE, SIGNAL(valueChanged(double)), this, SLOT(setE(double)));

	QVBoxLayout * sliderlayout = new QVBoxLayout;
	sliderlayout->addWidget(ciSlideN);
	sliderlayout->addWidget(ciSlideE);
	sliderlayout->addWidget(ciSlideS);
	sliderlayout->addWidget(ciSlideBI);
	sliderlayout->addWidget(stepOnEvidence);
	QVBoxLayout * plotlayout = new QVBoxLayout;
	plotlayout->addWidget(plot);
	plotlayout->addWidget(&modelFit, 0, Qt::AlignCenter);
	QGridLayout * toplayout = new QGridLayout;
	toplayout->addLayout(plotlayout,0,0);
	toplayout->addLayout(sliderlayout,0,1);

	setLayout(toplayout);
	plot->replot();
}

void GPWidget::computePlot()
{
	Array<double> p(1); // sample point
	unsigned n = 0;
	for (double d = -10; d <= 10; d += 0.02, n++)
	{
		p(0) = d;
		x[n] = d;
		y1[n] = (*gp)(p);
		double v = gp->Variance(p, p);
		l[n] = y1[n] - v;
		u[n] = y1[n] + v;
		y2[n] = sinc(d);
	}
}

// one dimensional benchmark function
double GPWidget::sinc(const double x)
{
	if (x == 0) return 1;
	return sin(x) / x;
}

void GPWidget::generateTrainingData()
{
	// generate training data
	inTrain.resize(N, 1u, false);
	targetTrain.resize(N, 1u, false);
	for (unsigned i = 0; i < N; i++)
	{
		inTrain(i, 0) = Rng::uni(-10, 10);
		targetTrain(i, 0)   = sinc(inTrain(i, 0)) + noise * Rng::gauss();
	}
	rprop.init(*gp);
}

void GPWidget::train()
{
	try
	{
		gp->train(inTrain, targetTrain);
		evidence = gpEv.error(*gp, inTrain, targetTrain);
		modelFit.setText("evidence: " + QString::number(-evidence));
	}
	catch (...)
	{
		std::cerr << "numerical problems" << std::endl;
	}
}

void GPWidget::setN(double l)
{
	N = unsigned(l);
	setN();
}

void GPWidget::setN()
{
	generateTrainingData();
	trainingCurve->setRawData(inTrain.begin(), targetTrain.begin(), N);
	train();
	computePlot();
	plot->replot();
}

void GPWidget::setE(double e)
{
	noise = e; // pow(10., e);
	generateTrainingData();

	trainingCurve->setRawData(inTrain.begin(), targetTrain.begin(), N);

	train();
	computePlot();
	plot->replot();
}

void GPWidget::setS(double s)
{
	gp->setSigma(s);
	train();
	computePlot();
	plot->replot();
}

void GPWidget::setBI(double bi)
{
	gp->setBetaInv(bi);
	train();
	computePlot();
	plot->replot();
}

void GPWidget::gradientStep()
{
	rprop.optimize(*gp, gpEv, inTrain, targetTrain);
	setTheta(gp->getParameter(0), gp->getParameter(1));
}

void GPWidget::setTheta(double bi, double s)
{
	gp->setBetaInv(bi);
	ciSlideBI->setValue(bi);
	gp->setSigma(s);
	ciSlideS->setValue(s);
	train();
	computePlot();
	plot->replot();
}


////////////////////////////////////////////////////////////


int main(int argc, char **argv)
{
	try
	{
		QApplication app(argc, argv);
		GPWidget widget;
		widget.resize(800, 400);
		widget.show();
		return app.exec();
	}
	catch (const SharkException& e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}
}
