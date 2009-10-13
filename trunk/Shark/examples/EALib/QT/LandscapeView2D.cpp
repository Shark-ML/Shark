//===========================================================================
/*!
 *  \file LandscapeView3D.cpp
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


#include "LandscapeView2D.h"
#include "Canyon.h"
#include "Experiment.h"


LandscapeView2D::LandscapeView2D(QString title, Experiment* experiment, QWidget* parent)
: View("[" + title + "] " + QString(experiment->m_problem.getObjectiveFunction()->name().c_str()), experiment, parent)
{
	m_monitorPopulation = true;
	m_monitorSearchDistribution = true;
	m_image = NULL;

	setMinimumSize(50, 50);

	// check the dimension
	ObjectiveFunctionVS<double>* fitness = m_experiment->m_problem.getObjectiveFunction();
	if (fitness->dimension() != 2) return;

	// create the color palette
	int i;
	for (i=0; i<3; i++)
	{
		m_palette[       i] = 0x000000fc + 0x00000001 * i;
	}
	for (i=0; i<255; i++)
	{
		m_palette[   3 + i] = 0x000000ff + 0x00000100 * i;
		m_palette[ 258 + i] = 0x0000ffff + 0xffffffff * i;
		m_palette[ 513 + i] = 0x0000ff00 + 0x00010000 * i;
		m_palette[ 768 + i] = 0x00ffff00 + 0xffffff00 * i;
	}
	m_palette[1023] = 0x00ff0000;
	for (i=0; i<1024; i++) m_palette[i] |= 0xff000000;

	OnReset();
}

LandscapeView2D::~LandscapeView2D()
{
	if (m_image != NULL) delete m_image;
}


void LandscapeView2D::OnReset()
{
	// sample a 512 x 512 value grid
	ObjectiveFunctionVS<double>* fitness = m_experiment->m_problem.getObjectiveFunction();	int x, y;
	if (fitness->dimension() != 2) return;

	double point[2];
	double m = 1e100;
	double M = -1e100;
	m_value.resize(512, 512, false);
	for (y=0; y<512; y++)
	{
		point[1] = (y - 256) / 128.0;
		for (x=0; x<512; x++)
		{
			point[0] = (x - 256) / 128.0;
			double v = (*fitness)(point);
			if (v < m) m = v;
			if (v > M) M = v;
			m_value(y, x) = v;
		}
	}
	for (y=0; y<512; y++)
	{
		for (x=0; x<512; x++)
		{
			m_value(y, x) = (m_value(y, x) - m) / (M - m);
		}
	}

	// compute the initial bitmap
	recomputeImage(width(), height());
}

void LandscapeView2D::OnChanged(int evals)
{
	// update the widget
	update();
}

QSize LandscapeView2D::sizeHint() const
{
	return QSize(512, 512);
}

void LandscapeView2D::recomputeImage(int sx, int sy)
{
	if (m_value.ndim() != 2) return;

	ObjectiveFunctionVS<double>* fitness = m_experiment->m_problem.getObjectiveFunction();
	if (fitness->dimension() != 2) return;

	Canyon* canyon = dynamic_cast<Canyon*>(fitness);

	if (m_image != NULL) delete m_image;
	m_image = new QImage(sx, sy, QImage::Format_RGB32);

	int x, y;
	int s = (sx > sy) ? sx : sy;

	m_scale = s / 3.9999;
	m_offsetX = sx / 2.0;
	m_offsetY = sy / 2.0;

	if (canyon != NULL)
	{
		for (y=0; y<sy; y++)
		{
			double yy = (y - m_offsetY) / m_scale;
			yy = 256.0 * (2.0 - yy);
			for (x=0; x<sx; x++)
			{
				double xx = (x - m_offsetX) / m_scale;
				xx = 256.0 * (xx + 2.0);

				double height;
				unsigned int color;
				canyon->get(xx, yy, height, color);
				m_image->setPixel(x, y, color);
			}
		}
	}
	else
	{
		for (y=0; y<sy; y++)
		{
			double yy = (y - m_offsetY) / m_scale;
			yy = (2.0 - yy) * 128.0;
			int yi = (int)yy;
			double yr = yy - yi;
			for (x=0; x<sx; x++)
			{
				double xx = (x - m_offsetX) / m_scale;
				xx = (xx + 2.0) * 128.0;
				int xi = (int)xx;

				double v;
				if (xx >= 511.0 || yy >= 511.0) v = m_value(yi, xi);
				else
				{
					double xr = xx - xi;
					double v00 = m_value(yi, xi);
					double v01 = m_value(yi, xi + 1);
					double v10 = m_value(yi + 1, xi);
					double v11 = m_value(yi + 1, xi + 1);
					v = (1.0 - xr) * ((1.0 - yr) * v00 + yr * v10) + xr * ((1.0 - yr) * v01 + yr * v11);
				}

				int index = (int)(1024 * v);
				if (index < 0) index = 0;
				if (index > 1023) index = 1023;
				m_image->setPixel(x, y, m_palette[index]);
			}
		}
	}
}

void LandscapeView2D::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);

	if (m_image == NULL)
	{
		painter.drawText(0, 0, width(), height(), Qt::AlignCenter, "This view can only illustrate two-dimensional search spaces.");
		return;
	}

	painter.drawImage(0, 0, *m_image);

	if (m_monitorSearchDistribution)
	{
		Array<double> position(2);
		Array<double> C(2, 2);
		position = 0.0;
		if (m_experiment->m_algorithm.getObservation(&PropertyDesc::position, position)
				&& m_experiment->m_algorithm.getObservation(&PropertyDesc::covariance, C))
		{
			Array<double> B(2, 2);
			Array<double> lambda(2);
			eigensymm(C, B, lambda);
			lambda(0) = sqrt(lambda(0));
			lambda(1) = sqrt(lambda(1));
			QPoint ellipse[33];
			int i;
			for (i=0; i<33; i++)
			{
				double alpha = 0.1963495408493620774 * i;
				double c = cos(alpha);
				double s = sin(alpha);
				double x = position(0) + lambda(0) * c * B(0, 0) + lambda(1) * s * B(0, 1);
				double y = position(1) + lambda(0) * c * B(1, 0) + lambda(1) * s * B(1, 1);
				ellipse[i].setX((int)(m_offsetX + m_scale * x));
				ellipse[i].setY((int)(m_offsetY - m_scale * y));
			}
			painter.setPen(Qt::lightGray);
			painter.drawPolyline(ellipse, 33);
		}
	}

	if (m_monitorPopulation)
	{
		Array<double> population;
		if (m_experiment->m_algorithm.getObservation(&PropertyDesc::population, population) && population.ndim() == 2)
		{
			painter.setBrush(QBrush(Qt::white));
			int i, ic = population.dim(0);
			for (i=0; i<ic; i++)
			{
				int x = (int)(m_offsetX + m_scale * population(i, 0));
				int y = (int)(m_offsetY - m_scale * population(i, 1));
				painter.drawEllipse(x - 2, y - 2, 4, 4);
			}
		}
	}
}

void LandscapeView2D::resizeEvent(QResizeEvent* event)
{
	recomputeImage(width(), height());
}
