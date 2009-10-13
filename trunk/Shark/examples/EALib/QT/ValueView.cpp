//===========================================================================
/*!
 *  \file ValueView.cpp
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
#include <vector>

#include "ValueView.h"
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


ValueView::ValueView(QString title, Experiment* experiment, PropertyDesc* property, QWidget* parent)
: View("[" + title + "] " + QString(property->name()), experiment, parent)
{
	m_property = property;
	m_totalEvaluations = m_experiment->m_totalEvaluations;

	setMinimumSize(BORDERX + SAFE_X, BORDERY + SAFE_Y);
	effectiveWidth = width();
	effectiveHeight = height();
	if (effectiveWidth < BORDERX + SAFE_X) effectiveWidth = BORDERX + SAFE_X;
	if (effectiveHeight < BORDERY + SAFE_Y) effectiveHeight = BORDERY + SAFE_Y;

	m_color = Qt::red;
	m_style = 17;

	setAutoFillBackground(false);
	performMinimalUpdate = false;
	numberOfDrawnPatterns = 0;
	bitmap = NULL;

	OnReset();
}

ValueView::~ValueView()
{
	if (bitmap != NULL) delete bitmap;
}


void ValueView::OnReset()
{
	m_evals.resize(false);
	m_values.resize(false);
	minval = 1e100;
	maxval = -1e100;
}

void ValueView::OnChanged(int evals)
{
	if (m_evals.ndim() == 0 || (numberOfDrawnPatterns == (int)m_evals.dim(0)))
		performMinimalUpdate = true;

	Array<double> data;
	m_experiment->m_algorithm.getObservation(m_property, data);

	m_evals.append_elem(evals);
	m_values.append_rows(data);

	// check for changes
	bool ch = false;
	int e, ec = data.nelem();
	for (e=0; e<ec; e++)
	{
		double v = data.elem(e);
		if (minval > maxval) { minval = maxval = v; ch = true; }
		else
		{
			if (v < minval) { minval = v; ch = true; }
			else if (v > maxval) { maxval = v; ch = true; }
		}
	}

	bool minimal = true;
	if (ch)
	{
		int old_numUnitsY = numunitsy;
		double old_unitY = unitY;
		ComputeRange();
		if (numunitsy != old_numUnitsY || unitY != old_unitY) minimal = false;
	}
	performMinimalUpdate = performMinimalUpdate && minimal;

	// update the widget
	update();
}

QSize ValueView::sizeHint() const
{
	return QSize(512, 384);
}

// compute dynamic ranges
void ValueView::ComputeRange()
{
	// x-axis
	{
		int space = effectiveWidth - BORDERX;
		double max_units = space / 40.0;
		double min_evals_per_unit = m_totalEvaluations / max_units;
		int ex = (int)ceil(log(min_evals_per_unit) / LOG10);
		unitX = (int)Shark::round(pow(10.0, ex));
		if (unitX == 0) unitX = 1;
		numunitsx = (int)ceil((double)m_totalEvaluations / (double)unitX);
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

// input: iteration number
// return: horizontal pixel position
int ValueView::TransformX(int iter)
{
	return LEFT + iter * unitwidth / unitX;
}

// input: value
// return: vertical pixel position
int ValueView::TransformY(double value)
{
	if (m_property->isLogScale()) value = log(value) / LOG10;
// 	return TOP + (int)Shark::round(unitheight * (maxrange - value) / unitY);
	return effectiveHeight - BOTTOM - (int)Shark::round(unitheight * (value - minrange) / unitY);
}

void ValueView::paintLegend(QPainter& painter)
{
}

void ValueView::paintCoordinates(QPainter& painter)
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

void ValueView::paintData(QPainter& painter)
{
	if (performMinimalUpdate)
	{
		if (m_values.ndim() == 1)
		{
			int i, ic = m_values.dim(0);
			int i_f = numberOfDrawnPatterns - 1;
			int i_n = ic - i_f;
			if (i_f < 0) { i_n += i_f; i_f = 0; }
			if (i_n <= 1) return;
			std::vector<QPoint> pt(i_n);
			for (i=0; i<i_n; i++)
			{
				pt[i].rx() = TransformX(m_evals(i_f + i));
				pt[i].ry() = TransformY(m_values(i_f + i));
			}
			if (m_style & 1)
			{
				for (i=1; i<i_n; i++)
				{
					painter.drawEllipse(pt[i].x() - 2, pt[i].y() - 2, 4, 4);
				}
			}
			if (m_style & 2)
			{
				for (i=1; i<i_n; i++)
				{
					painter.drawLine(pt[i].x() - 2, pt[i].y() - 2, pt[i].x() + 2, pt[i].y() + 2);
					painter.drawLine(pt[i].x() - 2, pt[i].y() + 2, pt[i].x() + 2, pt[i].y() - 2);
				}
			}
			if (m_style & 16) painter.drawPolyline(&pt[0], i_n);
		}
		else if (m_values.ndim() == 2)
		{
			int j, jc = m_values.dim(1);
			int i, ic = m_values.dim(0);
			int i_f = numberOfDrawnPatterns - 1;
			int i_n = ic - i_f;
			if (i_f < 0) { i_n += i_f; i_f = 0; }
			if (i_n <= 1 || jc == 0) return;
			std::vector<QPoint> pt(i_n);
			for (j=0; j<jc; j++)
			{
				for (i=0; i<i_n; i++)
				{
					pt[i].rx() = TransformX(m_evals(i_f + i));
					pt[i].ry() = TransformY(m_values(i_f + i, j));
				}
				if (m_style & 1)
				{
					for (i=1; i<i_n; i++)
					{
						painter.drawEllipse(pt[i].x() - 2, pt[i].y() - 2, 4, 4);
					}
				}
				if (m_style & 2)
				{
					for (i=1; i<i_n; i++)
					{
						painter.drawLine(pt[i].x() - 2, pt[i].y() - 2, pt[i].x() + 2, pt[i].y() + 2);
						painter.drawLine(pt[i].x() - 2, pt[i].y() + 2, pt[i].x() + 2, pt[i].y() - 2);
					}
				}
				if (m_style & 16) painter.drawPolyline(&pt[0], i_n);
			}
		}
	}
	else
	{
		if (m_values.ndim() == 1)
		{
			int i, ic = m_values.dim(0);
			if (ic == 0) return;
			std::vector<QPoint> pt(ic);
			for (i=0; i<ic; i++)
			{
				pt[i].rx() = TransformX(m_evals(i));
				pt[i].ry() = TransformY(m_values(i));
			}
			if (m_style & 1)
			{
				for (i=0; i<ic; i++)
				{
					painter.drawEllipse(pt[i].x() - 2, pt[i].y() - 2, 4, 4);
				}
			}
			if (m_style & 2)
			{
				for (i=0; i<ic; i++)
				{
					painter.drawLine(pt[i].x() - 2, pt[i].y() - 2, pt[i].x() + 2, pt[i].y() + 2);
					painter.drawLine(pt[i].x() - 2, pt[i].y() + 2, pt[i].x() + 2, pt[i].y() - 2);
				}
			}
			if (m_style & 16) painter.drawPolyline(&pt[0], ic);
		}
		else if (m_values.ndim() == 2)
		{
			int j, jc = m_values.dim(1);
			int i, ic = m_values.dim(0);
			if (ic == 0 || jc == 0) return;
			std::vector<QPoint> pt(ic);
			for (j=0; j<jc; j++)
			{
				for (i=0; i<ic; i++)
				{
					pt[i].rx() = TransformX(m_evals(i));
					pt[i].ry() = TransformY(m_values(i, j));
				}
				if (m_style & 1)
				{
					for (i=0; i<ic; i++)
					{
						painter.drawEllipse(pt[i].x() - 2, pt[i].y() - 2, 4, 4);
					}
				}
				if (m_style & 2)
				{
					for (i=0; i<ic; i++)
					{
						painter.drawLine(pt[i].x() - 2, pt[i].y() - 2, pt[i].x() + 2, pt[i].y() + 2);
						painter.drawLine(pt[i].x() - 2, pt[i].y() + 2, pt[i].x() + 2, pt[i].y() - 2);
					}
				}
				if (m_style & 16) painter.drawPolyline(&pt[0], ic);
			}
		}
	}
}

void ValueView::paintEvent(QPaintEvent* event)
{
	if (m_evals.ndim() == 0) return;

	if (bitmap == NULL) bitmap = new QPixmap(width(), height());
	QPainter painter(bitmap);

	if (! performMinimalUpdate)
	{
		painter.fillRect(rect(), painter.background());
		paintCoordinates(painter);
		paintLegend(painter);
	}
	painter.setPen(m_color);
	painter.setBrush(QBrush(m_color));
	paintData(painter);

	QPainter realpainter(this);
	realpainter.drawPixmap(0, 0, *bitmap);

	performMinimalUpdate = false;
	numberOfDrawnPatterns = m_evals.dim(0);
}

void ValueView::resizeEvent(QResizeEvent* event)
{
	if (bitmap != NULL) delete bitmap;
	bitmap = new QPixmap(width(), height());

	effectiveWidth = width();
	effectiveHeight = height();
	if (effectiveWidth < BORDERX + SAFE_X) effectiveWidth = BORDERX + SAFE_X;
	if (effectiveHeight < BORDERY + SAFE_Y) effectiveHeight = BORDERY + SAFE_Y;
	ComputeRange();

	performMinimalUpdate = false;
}
