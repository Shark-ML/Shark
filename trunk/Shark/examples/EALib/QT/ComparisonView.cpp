//===========================================================================
/*!
 *  \file ComparisonView.cpp
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


#include <QPoint>
#include <QPainter>
#include <QFontMetrics>
#include <QMouseEvent>
#include <QMenu>
#include <QFile>
#include <QFileDialog>
#include <QMessageBox>
#include <vector>

#include "ComparisonView.h"
#include "Optimization.h"
#include "Experiment.h"


#define LOG10 2.302585092994045901
#define LEFT 60
#define RIGHT 20
#define BORDERX (LEFT+RIGHT)
#define TOP 10
#define BOTTOM 25
#define BORDERY (TOP+BOTTOM)
#define SAFE_Y 50
#define SAFE_X 50


////////////////////////////////////////////////////////////


// default colors
unsigned int ComparisonView::m_default[10] =
{
	0xffff0000,
	0xff00e000,
	0xff0000ff,
	0xffffc000,
	0xffc000c0,
	0xff00c0c0,
	0xffc04040,
	0xff808080,
	0xff008000,
	0xff000000,
};

// mode names
char ComparisonView::m_modename[5][50] =
{
	"",
	"Mean",
	"Median",
	"Minimum",
	"Maximum"
};


////////////////////////////////////////////////////////////


ComparisonView::ComparisonView(PropertyDesc* property, int mode, QWidget* parent)
: View(QString(m_modename[mode]) + " of " + QString(property->name()) + " over evaluations", NULL, parent)
{
	m_property = property;
	m_mode = mode;

	maxtime = 0;
	minval = 1e100;
	maxval = -1e100;
	m_bNeedsUpdate = false;

	setMinimumSize(BORDERX + SAFE_X, BORDERY + SAFE_Y);
	effectiveWidth = width();
	effectiveHeight = height();
	if (effectiveWidth < BORDERX + SAFE_X) effectiveWidth = BORDERX + SAFE_X;
	if (effectiveHeight < BORDERY + SAFE_Y) effectiveHeight = BORDERY + SAFE_Y;

	setAutoFillBackground(false);
	bitmap = NULL;
}

ComparisonView::~ComparisonView()
{
	if (bitmap != NULL) delete bitmap;
}


void ComparisonView::AddExperiment(Experiment* experiment)
{
	// check if the property is recorded
	int propindex = experiment->recordingPropertyIndex(m_property);
	if (propindex < 0) throw SHARKEXCEPTION("[ComparisonView::AddExperiment] property is not available");

	// fill in a new curve structure
	tCurve curve;
	curve.experiment = experiment;
	if (m_curve.size() < 10) curve.color = m_default[m_curve.size()];
	else curve.color = 0xff000000 + ((rand() % 128 + 64) << 16) + ((rand() % 128 + 64) << 8) + (rand() % 128 + 64);
	curve.mintime = experiment->firstEval();
	curve.propindex = propindex;
	m_curve.push_back(curve);

	// update the widget
	m_bNeedsUpdate = true;
	update();
}

void ComparisonView::OnReset()
{
	// ignore
}

void ComparisonView::OnChanged(int evals)
{
	// ignore
}

QSize ComparisonView::sizeHint() const
{
	return QSize(512, 384);
}

// void ComparisonView::SaveAsScriptFile(bool)
// {
// }

void ComparisonView::SaveAsDataFile(bool)
{
	QString filename = QFileDialog::getSaveFileName(this, "save data file", m_modename[m_mode] + QString("_") + m_property->name() + ".gnuplotdata");
	QFile file(filename);
	if (! file.open(QIODevice::WriteOnly))
	{
		QMessageBox::critical(NULL,
				"Save Gnuplot Data",
				"EXPORT FAILED"
			);
		return;
	}

	QString content;
	Array<double> data;
	CollectData(data);
	int c, cc = data.dim(0);
	int t, tc = data.dim(1);

	// "correct" unavailable data at start
	for (c=0; c<cc; c++)
	{
		int latest = -1;
		for (t=0; t<tc; t++) if (data(c, t) == 1e100) latest = t;
		for (t=0; t<=latest; t++) data(c, t) = data(c, latest + 1);
	}

	// output one column per curve
	for (t=0; t<tc; t++)
	{
		for (c=0; c<cc; c++)
		{
			if (c > 0) content += " ";
			content += QString::number(data(c, t));
		}
		content += "\n";
	}

	file.write(content.toAscii());
	file.close();
}

// compute dynamic ranges
void ComparisonView::ComputeRange()
{
	// x-axis
	{
		int space = effectiveWidth - BORDERX;
		double max_units = space / 40.0;
		double min_evals_per_unit = maxtime / max_units;
		int ex = (int)ceil(log(min_evals_per_unit) / LOG10);
		unitX = (int)Shark::round(pow(10.0, ex));
		if (unitX == 0) unitX = 1;
		numunitsx = (int)ceil((double)maxtime / (double)unitX);
		if (numunitsx == 0) numunitsx = 1;
		unitwidth = space / numunitsx;
	}

	// y-axis
	{
		int space = effectiveHeight - BORDERY;
		double max_units = space / 30.0;
		if (minval > maxval)
		{
			minrange = 0.0;
			maxrange = 1.0;
			unitY = 1.0;
			unitheight = space;
			numunitsy = 1;
		}
		else
		{
			double minv = minval;
			double maxv = maxval;
			if (m_property->isLogScale())
			{
				minv = log(minv) / LOG10;
				maxv = log(maxv) / LOG10;
			}

			if (minval == maxval)
			{
				minrange = floor(minv);
				maxrange = ceil(maxv);
				if (minrange == maxrange) maxrange += 1.0;
				unitY = 1.0;
				unitheight = space;
				numunitsy = 1;
			}
			else
			{
				if (m_property->isLogScale())
				{
					minrange = floor(minv);
					maxrange = ceil(maxv);
					int u = (int)Shark::round(maxrange - minrange);
					if (u <= max_units) unitY = 1.0;
					else unitY = ceil(u / max_units);
					numunitsy = (int)ceil((maxrange - minrange) / unitY);
					unitheight = space / numunitsy;
					minrange = maxrange - numunitsy * unitY;
				}
				else
				{
					double diff = maxv - minv;
					double min_diff_per_unit = diff / max_units;
					int ex = (int)ceil(log(min_diff_per_unit) / LOG10);
					unitY = pow(10.0, ex);
					numunitsy = (int)floor(diff / unitY);
					if (numunitsy == 0) numunitsy = 1;
					unitheight = space / numunitsy;
					minrange = floor(minv / unitY) * unitY;
					maxrange = ceil(maxv / unitY) * unitY;

					diff = maxrange - minrange;
					min_diff_per_unit = diff / max_units;
					ex = (int)ceil(log(min_diff_per_unit) / LOG10);
					unitY = pow(10.0, ex);
					numunitsy = (int)floor(diff / unitY);
					if (numunitsy == 0) numunitsy = 1;
					unitheight = space / numunitsy;
					minrange = floor(minv / unitY) * unitY;
					maxrange = ceil(maxv / unitY) * unitY;
				}
			}
		}
	}
}

void ComparisonView::CollectData(Array<double>& data)
{
	int x, xc = effectiveWidth - BORDERX;
	int e, ec = m_curve.size();

	// compute time range
	maxtime = 0;
	for (e=0; e<ec; e++)
	{
		if (m_curve[e].experiment->m_totalEvaluations > maxtime)
		{
			maxtime = m_curve[e].experiment->m_totalEvaluations;
		}
	}

	// get all values and compute value range
	minval = 1e100;
	maxval = -1e100;
	data.resize(ec, xc, false);
	for (x=0; x<xc; x++)
	{
		double time = (double)(x * maxtime) / (double)(xc - 1.0);
		for (e=0; e<ec; e++)
		{
			if (time < m_curve[e].mintime) data(e, x) = 1e100;
			else
			{
				double v = m_curve[e].experiment->recording(m_curve[e].propindex, time, m_mode);
				data(e, x) = v;
				if (v < minval) minval = v;
				if (v > maxval) maxval = v;
			}
		}
	}
}

void ComparisonView::ComputePlot()
{
	Array<double> value;
	CollectData(value);

	// compute coordinate system
	ComputeRange();

	// paint the bitmap
	QPainter painter(bitmap);
	painter.fillRect(0, 0, bitmap->width(), bitmap->height(), QBrush(Qt::white));
	paintCoordinates(painter);
	paintLegend(painter);

	int c, cc = m_curve.size();
	for (c=0; c<cc; c++) paintCurve(painter, &m_curve[c], value[c]);

	m_bNeedsUpdate = false;
}

// // input: iteration number
// // return: horizontal pixel position
// int ComparisonView::TransformX(int iter)
// {
// 	return (int)ceil(LEFT + (double)iter * (double)unitwidth / (double)unitX);
// }

// input: value
// return: vertical pixel position
int ComparisonView::TransformY(double value)
{
	if (m_property->isLogScale()) value = log(value) / LOG10;
	return effectiveHeight - BOTTOM - (int)Shark::round(unitheight * (value - minrange) / unitY);
}

void ComparisonView::paintLegend(QPainter& painter)
{
	painter.setPen(Qt::black);

	int e, ec = m_curve.size();
	int txtwidth = 50;
	for (e=0; e<ec; e++)
	{
		int w;
		if (! painter.isActive())
		{
			w = 12 * m_curve[e].experiment->name().length();
		}
		else
		{
			QFontMetrics metrics = painter.fontMetrics();
			QString str = m_curve[e].experiment->name();
			w = 0;
			int i, ic = str.length();
			for (i=0; i<ic; i++) w += metrics.charWidth(str, i);
		}
		if (w > txtwidth) txtwidth = w;
	}

	int totalwidth = 80 + txtwidth;
	int totalheight = 20 + 20 * ec;
	int x = effectiveWidth - RIGHT - 10 - totalwidth;
	painter.fillRect(x, TOP + 10, totalwidth, totalheight, QBrush(QColor(255, 240, 192)));
	painter.drawRect(x, TOP + 10, totalwidth, totalheight);

	for (e=0; e<ec; e++)
	{
		QString str = m_curve[e].experiment->name();
		painter.drawText(x + 10, TOP + 20 + 20 * e, txtwidth, 20, Qt::AlignRight | Qt::AlignVCenter, str);
	}

	for (e=0; e<ec; e++)
	{
		int color = m_curve[e].color;
		painter.setPen(QColor((color >> 16) & 255, (color >> 8) & 255, color & 255));
		painter.drawLine(x + txtwidth + 20, TOP + 30 + 20 * e, x + txtwidth + 70, TOP + 30 + 20 * e);
	}
}

void ComparisonView::paintCoordinates(QPainter& painter)
{
	int x;
	int y;
	double value;
	char txt[256];

	painter.setPen(Qt::gray);

	// draw vertical lines
	for (x=0; x<=numunitsx; x++)
	{
		int xx = LEFT + x * unitwidth;
		painter.drawLine(xx, TOP, xx, effectiveHeight - BOTTOM);
	}

	// draw horizontal lines
	for (y=0; y<=numunitsy; y++)
	{
		int yy = effectiveHeight - BOTTOM - (numunitsy - y) * unitheight;
		painter.drawLine(LEFT, yy, effectiveWidth - RIGHT, yy);
	}

	// draw box with coordinates
	painter.setPen(Qt::black);
	painter.drawRect(LEFT, TOP, effectiveWidth - BORDERX, effectiveHeight - BORDERY);
	for (x=0; x<=numunitsx; x++)
	{
		int xx = LEFT + x * unitwidth;
		painter.drawLine(xx, effectiveHeight - BOTTOM, xx, effectiveHeight - BOTTOM + 2);
		sprintf(txt, "%d", x * unitX);
		painter.drawText(QRect(xx - 40, effectiveHeight - BOTTOM + 4, 80, 16), Qt::AlignCenter | Qt::AlignTop, txt, NULL);
	}
	for (y=0; y<=numunitsy; y++)
	{
		int yy = effectiveHeight - BOTTOM - (numunitsy - y) * unitheight;
		painter.drawLine(LEFT - 2, yy, LEFT, yy);
		value = (numunitsy - y) * (maxrange - minrange) / numunitsy + minrange;
		if (m_property->isLogScale()) value = pow(10.0, value);
		sprintf(txt, "%.6g", value);
		painter.drawText(QRect(0, yy - 10, LEFT - 4, 20), Qt::AlignRight | Qt::AlignVCenter, txt, NULL);
	}
}

void ComparisonView::paintCurve(QPainter& painter, tCurve* curve, const Array<double>& values)
{
	int color = curve->color;
	painter.setPen(QColor((color >> 16) & 255, (color >> 8) & 255, color & 255));

	int i, ic = values.dim(0);
	if (ic == 0) return;
	int first = 0;
	std::vector<QPoint> pt(ic);
	for (i=0; i<ic; i++)
	{
		if (values(i) == 1e100) first++;
		else
		{
			pt[i].rx() = LEFT + i;
			pt[i].ry() = TransformY(values(i));
		}
	}
	painter.drawPolyline(&pt[first], ic - first);
}

void ComparisonView::paintEvent(QPaintEvent* event)
{
	if (bitmap == NULL) bitmap = new QPixmap(width(), height());
	if (m_bNeedsUpdate) ComputePlot();

	QPainter painter(this);
	painter.drawPixmap(0, 0, *bitmap);
}

void ComparisonView::resizeEvent(QResizeEvent* event)
{
	if (bitmap != NULL) delete bitmap;
	bitmap = new QPixmap(width(), height());

	effectiveWidth = width();
	effectiveHeight = height();
	if (effectiveWidth < BORDERX + SAFE_X) effectiveWidth = BORDERX + SAFE_X;
	if (effectiveHeight < BORDERY + SAFE_Y) effectiveHeight = BORDERY + SAFE_Y;

	m_bNeedsUpdate = true;
}

void ComparisonView::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::RightButton)
	{
		event->accept();
		QMenu* context = new QMenu();
		QAction* action;
// 		action = context->addAction("save as gnuplot script file");
// 		connect(action, SIGNAL(triggered(bool)), this, SLOT(SaveAsScriptFile(bool)));
		action = context->addAction("save as gnuplot data file");
		connect(action, SIGNAL(triggered(bool)), this, SLOT(SaveAsDataFile(bool)));
		context->popup(event->globalPos());
	}
	else event->ignore();
}
