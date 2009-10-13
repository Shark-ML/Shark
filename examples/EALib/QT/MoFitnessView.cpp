//===========================================================================
/*!
 *  \file MoFitnessView.cpp
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


#include "MoFitnessView.h"
#include "Optimization.h"
#include "Experiment.h"


#define LEFT 60
#define RIGHT 20
#define BORDERX (LEFT+RIGHT)
#define TOP 10
#define BOTTOM 25
#define BORDERY (TOP+BOTTOM)
#define LOG10 2.302585092994045901


MoFitnessView::MoFitnessView(QString title, Experiment* experiment, QWidget* parent)
: View("[" + title + "] objective space", experiment, parent)
{
	bitmap = NULL;

	setMinimumSize(50, 50);

	OnReset();
}

MoFitnessView::~MoFitnessView()
{
	if (bitmap != NULL) delete bitmap;
}


void MoFitnessView::OnReset()
{
	minval[0] = 1e100;
	minval[1] = 1e100;
	maxval[0] = -1e100;
	maxval[1] = -1e100;

	m_drawn = 0;
	m_point.resize(0, 2, false);
	performMinimalUpdate = false;
}

void MoFitnessView::OnChanged(int evals)
{
	if (m_point.ndim() == 0 || (m_point.ndim() == 2 && m_point.dim(0) == m_drawn)) performMinimalUpdate = true;

	Array<double> data;
	m_experiment->m_algorithm.getObservation(&PropertyDesc::mooFitness, data);
	if (data.nelem() == 0) { performMinimalUpdate = false; return; }

	m_point.append_rows(data);

	// check for changes
	bool ch = false;
	int e, ec = data.dim(0);
	for (e=0; e<ec; e++)
	{
		double x = data(e, 0);
		double y = data(e, 1);
		if (minval[0] > maxval[0]) { minval[0] = x - 1.0; maxval[0] = x + 1.0; ch = true; }
		else
		{
			if (x < minval[0]) { minval[0] = x - 1.0; ch = true; }
			else if (x > maxval[0]) { maxval[0] = x + 1.0; ch = true; }
		}
		if (minval[1] > maxval[1]) { minval[1] = y - 1.0; maxval[1] = y + 1.0; ch = true; }
		else
		{
			if (y < minval[1]) { minval[1] = y - 1.0; ch = true; }
			else if (y > maxval[1]) { maxval[1] = y + 1.0; ch = true; }
		}
	}

	performMinimalUpdate = performMinimalUpdate && (! ch);

	// update the widget
	update();
}

QSize MoFitnessView::sizeHint() const
{
	return QSize(512, 512);
}

void MoFitnessView::ComputeRange(double m, double M, int space, int mindist, double& start, double& step, int& N)
{
	if (m >= M)
	{
		start = m;
		step = 1e100;
		N = 0;
	}
	else
	{
		double max_steps = (double)space / (double)mindist;

		double diff = M - m;
		double min_diff_per_step = diff / max_steps;
		int ex = (int)ceil(log(min_diff_per_step) / LOG10);
		step = pow(10.0, ex);

		start = ceil(m / step) * step;
		double end = floor(M / step) * step;
		N = (int)Shark::round((end - start) / step) + 1;
	}
}

void MoFitnessView::paintEvent(QPaintEvent* event)
{
	ObjectiveFunctionVS<double>* fitness = m_experiment->m_problem.getObjectiveFunction();
	if (fitness->objectives() != 2)
	{
		QPainter realpainter(this);
		realpainter.drawText(0, 0, width(), height(), Qt::AlignCenter, "This view can only illustrate two-dimensional objective spaces.");
		return;
	}

	if (m_point.ndim() != 2 || m_point.dim(0) == 0) return;

	if (bitmap == NULL)
	{
		bitmap = new QPixmap(width(), height());
		performMinimalUpdate = false;
	}
	QPainter painter(bitmap);

	double fx = (width() - BORDERX) / (maxval[0] - minval[0]);
	double fy = (height() - BORDERY) / (maxval[1] - minval[1]);

	if (! performMinimalUpdate)
	{
		painter.fillRect(0, 0, width(), height(), QBrush(Qt::white));
		painter.setPen(Qt::black);
		painter.drawRect(LEFT, TOP, width() - BORDERX, height() - BORDERY);

		// draw coordinate lines and description
		char txt[256];

		painter.setPen(Qt::lightGray);
		double start, step;
		int n, N;
		ComputeRange(minval[0], maxval[0], width() - BORDERX, 40, start, step, N);
		for (n=0; n<N; n++)
		{
			double value = start + n * step;
			int pos = LEFT + (int)Shark::round(fx * (value - minval[0]));
			painter.setPen(Qt::gray);
			painter.drawLine(pos, TOP + 1, pos, height() - BOTTOM - 1);
			sprintf(txt, "%6.3g", value);
			painter.drawText(QRect(pos - 40, height() - BOTTOM + 4, 80, 16), Qt::AlignCenter, txt, NULL);
		}
		ComputeRange(minval[1], maxval[1], height() - BORDERY, 30, start, step, N);
		for (n=0; n<N; n++)
		{
			double value = start + n * step;
			int pos = height() - BOTTOM - (int)Shark::round(fy * (value - minval[1]));
			painter.drawLine(LEFT + 1, pos, width() - RIGHT - 1, pos);
			sprintf(txt, "%6.3g", value);
			painter.drawText(QRect(0, pos - 10, LEFT - 4, 20), Qt::AlignRight | Qt::AlignVCenter, txt, NULL);
		}

		painter.setPen(Qt::black);
		sprintf(txt, "%6.3g", minval[0]);
		painter.drawText(QRect(LEFT - 40, height() - BOTTOM + 4, 80, 16), Qt::AlignCenter | Qt::AlignTop, txt, NULL);
		sprintf(txt, "%6.3g", maxval[0]);
		painter.drawText(QRect(width() - RIGHT - 40, height() - BOTTOM + 4, 80, 16), Qt::AlignCenter, txt, NULL);
		sprintf(txt, "%6.3g", maxval[1]);
		painter.drawText(QRect(0, TOP - 10, LEFT - 4, 20), Qt::AlignRight | Qt::AlignVCenter, txt, NULL);
		sprintf(txt, "%6.3g", minval[1]);
		painter.drawText(QRect(0, height() - BOTTOM - 10, LEFT - 4, 20), Qt::AlignRight | Qt::AlignVCenter, txt, NULL);
	}

	painter.setBrush(QBrush(Qt::red));
	int i, ic = m_point.dim(0);
	if (performMinimalUpdate) i = m_drawn;
	else i = 0;
	for (; i<ic; i++)
	{
		int r = 255 * i / (m_experiment->m_totalEvaluations - 1);
		if (r > 255) r = 255;
		painter.setPen(QColor(r, 128, 255-r));

		int x = LEFT + (int)Shark::round(fx * (m_point(i, 0) - minval[0]));
		int y = height() - BOTTOM - (int)Shark::round(fy * (m_point(i, 1) - minval[1]));
		painter.drawEllipse(x-1, y-1, 2, 2);
// 		painter.drawPoint(x, y);
	}
	m_drawn = ic;

	QPainter realpainter(this);
	realpainter.drawPixmap(0, 0, *bitmap);

	performMinimalUpdate = false;
}

void MoFitnessView::resizeEvent(QResizeEvent* event)
{
	if (bitmap != NULL) delete bitmap;
	bitmap = new QPixmap(width(), height());

	performMinimalUpdate = false;
}
