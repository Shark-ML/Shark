//===========================================================================
/*!
 *  \file gaussian_process.h
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
 *      $RCSfile: gaussian_process.h,v $<BR>
 *      $Revision: 1.3 $<BR>
 *      $Date: 2007/10/22 15:26:14 $
 *
 *  \par Changes:
 *      $Log: gaussian_process.h,v $
 *      Revision 1.3  2007/10/22 15:26:14  glasmtbl
 *      *** empty log message ***
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


#ifndef _gaussian_process_example_H_
#define _gaussian_process_example_H_


#include <QApplication>
#include <QFont>
#include <QLabel>
#include <QLCDNumber>
#include <QPainter>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include <QWidget>

#include <qwt_legend.h>
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_plot_marker.h>
#include <qwt_scale_map.h>
#include <qwt_scale_engine.h>
#include <qwt_slider.h>
#include <qwt_symbol.h>

#include <Rng/GlobalRng.h>
#include <Array/Array.h>
#include <ReClaM/GaussianProcess.h>
#include <ReClaM/Rprop.h>


class CISlider : public QWidget
{
	Q_OBJECT

public:
	CISlider(char *title, double init, double l, double r, double step = 0);

	QwtSlider *slide();

	virtual void setValue(double x);

public slots:
	virtual void setIt(double x);

signals:
	void valuesChanged(double);

protected:
	QwtSlider *slider;
	QString   text;
	QLabel    *legend;
};


class CILogSlider : public CISlider
{
	Q_OBJECT

public:
	CILogSlider(char *title, double init, double l, double r);

	virtual void setValue(double x);

public slots:
	void setIt(double x);

signals:
	void valueChanged(double);
};


class PlotWidget  : public QwtPlot
{
public:
	PlotWidget(const char *title);

	QwtPlotCurve* addLine(const char *title, const Array<double> &x, const Array<double> &y,
						  const QPen &pen = QPen(Qt::black));
	QwtPlotCurve* addLine(const char *title, const double *x, const double *y, unsigned n,
						  const QPen &pen = QPen(Qt::black));
	QwtPlotCurve* addDash(const char *title, const Array<double> &x, const Array<double> &y,
						  const QPen &pen = QPen(Qt::black));
	QwtPlotCurve* addDash(const char *title, const double *x, const double *y, unsigned n,
						  const QPen &pen = QPen(Qt::black));
	QwtPlotCurve* addDots(const char *title, const Array<double> &x, const Array<double> &y,
						  const QPen &pen = QPen(Qt::black));
	QwtPlotCurve* addDots(const char *title, const double *x, const double *y, unsigned n,
						  const QPen &pen = QPen(Qt::black));
	void hideItems();
	void showItems();

protected:
	QwtPlotCurve *add(QwtPlotCurve *c, const double *x, const double *y, unsigned n,
					  const QPen &pen = QPen(Qt::black));

	std::vector<QwtPlotCurve *> crvs;
	int sym;
	int ps;
private:
	bool hideItem;
};


class GPWidget : public QWidget
{
	Q_OBJECT

public:
	GPWidget();

	void computePlot();
	double sinc(const double x);
	void generateTrainingData();
	void train();

public slots:
	void setN(double l);
	void setN();
	void setE(double e);
	void setS(double s);
	void setBI(double bi);
	void gradientStep();

protected:
	void setTheta(double bi, double s);

	PlotWidget* plot;
	CILogSlider* ciSlideS;
	CILogSlider* ciSlideBI;

	GaussianProcess *gp;

	NormalizedRBFKernel k;
	KernelExpansion model;

	double noise;
	Array<double> inTrain, targetTrain;

	unsigned N;

	double evidence;

private:
	double u[1001], l[1001], x[1001], y1[1001], y2[1001];
	QwtPlotCurve* trainingCurve;
	QLabel modelFit;

	GaussianProcessEvidence gpEv;
	IRpropPlus rprop;
};


#endif
