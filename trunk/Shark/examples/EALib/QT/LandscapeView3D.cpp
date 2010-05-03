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


#include "LandscapeView3D.h"
#include "Canyon.h"
#include "Experiment.h"


LandscapeView3D::LandscapeView3D(QString title, Experiment* experiment, QWidget* parent)
: View("[" + title + "] " + QString(experiment->m_problem.getObjectiveFunction()->name().c_str()), experiment, parent)
, m_image(256, 256, QImage::Format_RGB32)
{
	m_monitorPopulation = true;
	m_monitorSearchDistribution = true;

	// check the dimension
	ObjectiveFunctionVS<double>* fitness = m_experiment->m_problem.getObjectiveFunction();
	if (fitness->dimension() != 2) return;

	// precompute tables
	int r, p;
	for (r=0; r<2048; r++)
	{
		dist[r] = r + 1.0;
	}
	for (p=0; p<256; p++) right[p] = (p - 127.5) / 256.0;

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

LandscapeView3D::~LandscapeView3D()
{
}


void LandscapeView3D::OnReset()
{
	viewposX = 0.0;
	viewposY = 0.0;
	viewposZ = 400.0;
	viewdirXY = 0.25 * M_PI;
	viewdirZ = 64.0;
	targetposX = viewposX;
	targetposY = viewposY;
	targetposZ = viewposZ;

	// sample a 512 x 512 value grid
	ObjectiveFunctionVS<double>* fitness = m_experiment->m_problem.getObjectiveFunction();
	int x, y;
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
}

void LandscapeView3D::OnChanged(int evals)
{
	// remember the position
	ObjectiveFunctionVS<double>* fitness = m_experiment->m_problem.getObjectiveFunction();
	Array<double> position;
	if (m_experiment->m_algorithm.getObservation(&PropertyDesc::position, position))
	{
		targetposX = (position(0) + 2.0) * 256.0;
		targetposY = (position(1) + 2.0) * 256.0;
		Canyon* canyon = dynamic_cast<Canyon*>(fitness);
		if (canyon != NULL)
		{
			unsigned int col;
			canyon->get(targetposX, targetposY, targetposZ, col);
		}
		else targetposZ = get(0.5 * targetposX, 0.5 * targetposY);
	}

	// compute the bitmap
	ComputeView();

	// update the widget
	update();
}

double LandscapeView3D::get(double x, double y)
{
	if (x < 0.0 || y < 0.0 || x >= 511.0 || y >= 511.0) return 255.0;

	int xi = (int)x;
	double xr = x - xi;
	int yi = (int)y;
	double yr = y - yi;

	double v00 = m_value(yi, xi);
	double v01 = m_value(yi, xi + 1);
	double v10 = m_value(yi + 1, xi);
	double v11 = m_value(yi + 1, xi + 1);
	double v = (1.0 - xr) * ((1.0 - yr) * v00 + yr * v10) + xr * ((1.0 - yr) * v01 + yr * v11);

	return 255.0 * v;
}

#define ADD_Z 30.0
#define CAM_DIST 50.0
#define MAX_SPEED 25.0
#define MAX_ROT 0.5
#define MAX_VIEW_Z_DEVIATION 128.0
#define TIME_CONST 0.1
void LandscapeView3D::ComputeView()
{
	ObjectiveFunctionVS<double>* fitness = m_experiment->m_problem.getObjectiveFunction();
	Canyon* canyon = dynamic_cast<Canyon*>(fitness);

	// compute position and direction
	{
		unsigned int col;
		double dx = targetposX - viewposX;
		double dy = targetposY - viewposY;
		double distXY = sqrt(dx * dx + dy * dy);
		if (distXY > 0.0)
		{
			// horizontal position dynamics
			double derivativeX;
			double derivativeY;

			double reldist = distXY / CAM_DIST - 1.0;
			double factor = TIME_CONST * reldist * sqrt(fabs(reldist));
			if (factor > 0.9) factor = 0.9;
			else if (factor < -0.5) factor = -0.5;
			derivativeX = factor * dx;
			derivativeY = factor * dy;

			int r, w;
			for (r=0; r<2; r++)
			{
				double rad = MAX_SPEED * (r + 1.0) / 2.0;
				for (w=0; w<8; w++)
				{
					double ang = w * M_PI / 4.0;
					double c = cos(ang);
					double s = sin(ang);
					double x = viewposX + c * rad;
					double y = viewposY + s * rad;
					double z;
					if (canyon != NULL) canyon->get(x, y, z, col);
					else z = get(0.5 * x, 0.5 * y);
					double deltaZ = MAX_SPEED / 16.0 * tanh((z + ADD_Z - viewposZ) / MAX_SPEED);
					derivativeX -= c * deltaZ;
					derivativeY -= s * deltaZ;
				}
			}

			double len = sqrt(derivativeX*derivativeX + derivativeY*derivativeY);
			if (len > MAX_SPEED)
			{
				derivativeX *= MAX_SPEED / len;
				derivativeY *= MAX_SPEED / len;
			}
			viewposX += derivativeX;
			viewposY += derivativeY;

			// vertical position dynamics
			viewposZ += TIME_CONST * (targetposZ - viewposZ);
			double minZ;
			if (canyon != NULL) canyon->get(viewposX, viewposY, minZ, col);
			else minZ = get(0.5 * viewposX, 0.5 * viewposY);
			minZ += ADD_Z;
			if (viewposZ < minZ) viewposZ = minZ;

			// heading direction dynamics
			double targetdirXY;
			if (dx == 0.0) targetdirXY = M_PI / 2.0;
			else targetdirXY = atan(dy / dx);
			if (dx < 0.0) targetdirXY += M_PI;
			if (targetdirXY < 0.0) targetdirXY += 2.0 * M_PI;
			double diff = targetdirXY - viewdirXY;
			if (diff > M_PI) diff -= 2.0 * M_PI;
			else if (diff < -M_PI) diff += 2.0 * M_PI;
			double rot = TIME_CONST * diff;
			if (fabs(rot) > MAX_ROT) rot *= MAX_ROT / fabs(rot);
			viewdirXY += rot;
			if (viewdirXY < 0.0) viewdirXY += 2.0 * M_PI;
			if (viewdirXY >= 2.0 * M_PI) viewdirXY -= 2.0 * M_PI;

			// vertical direction dynamics
			double dz = targetposZ - viewposZ;
			double targetdirZ = 128.0 + 128.0 * (dz / distXY);
			viewdirZ += TIME_CONST * (targetdirZ - viewdirZ);
			if (viewdirZ < 128.0 - MAX_VIEW_Z_DEVIATION) viewdirZ = 128.0 - MAX_VIEW_Z_DEVIATION;
			else if (viewdirZ > 128.0 + MAX_VIEW_Z_DEVIATION) viewdirZ = 128.0 + MAX_VIEW_Z_DEVIATION;
		}
	}

	// compute the image
	int p, r;
	double x, y;
	double h;
	unsigned int col;
	double ca = cos(viewdirXY);
	double sa = sin(viewdirXY);

	bool drawSearchDistribution = false;
	Matrix C_inv(2, 2);
	if (m_monitorSearchDistribution)
	{
		Matrix C(2, 2);
		if (m_experiment->m_algorithm.getObservation(&PropertyDesc::covariance, C))
		{
			C_inv = C.inverse();
			double det = C_inv(0, 0) * C_inv(1, 1) - C_inv(0, 1) * C_inv(1, 0);
			if ( det > 1e-20 )
				drawSearchDistribution = true;
		}
	}

	if (canyon != NULL)
	{
		for (p=0; p<256; p++)
		{
			int b = 256;
			double dr = right[p];
			for (r=0; r<2048; r++)
			{
				double d = dist[r];
				x = viewposX + (ca + dr * sa) * d;
				y = viewposY + (sa - dr * ca) * d;
				if (x >= 0.0 && x < 1023.0 && y >= 0.0 && y < 1023.0)
				{
					canyon->get(x, y, h, col);
					int nb = (int)floor(viewdirZ - 128.0 * (h - viewposZ) / d);
					if (nb < b)
					{
						if (drawSearchDistribution)
						{
							Vector delta(2);
							delta(0) = targetposX - x;
							delta(1) = targetposY - y;
							delta *= 0.00390625;		// 1/256
							double kernel = exp(-0.5 * (delta * (C_inv * delta)));
							if (kernel > 0.01)
							{
								int add = (int)(100.0 * kernel);
								int red = (col & 0x00ff0000) >> 16;
								if (add + red > 255) add = 255 - red;
								col += 0x00010000 * add;
							}
						}
						if (nb < 0) nb = 0;
						for (; b>nb; b--) m_image.setPixel(p, b-1, col);
						y = nb;
						if (nb == 0) break;
					}
				}
			}
			for (; b>0; b--) m_image.setPixel(p, b-1, 0xffc8d0ff);
		}
	}
	else
	{
		for (p=0; p<256; p++)
		{
			int b = 256;
			double dr = right[p];
			for (r=0; r<2048; r++)
			{
				double d = dist[r];
				x = viewposX + (ca + dr * sa) * d;
				y = viewposY + (sa - dr * ca) * d;
				if (x >= 0.0 && x < 1022.0 && y >= 0.0 && y < 1022.0)
				{
					h = get(0.5 * x, 0.5 * y);

					int nb = (int)floor(viewdirZ - 128.0 * (h - viewposZ) / d);
					if (nb < b)
					{
						col = m_palette[(int)(2.0 * h)];
						if (drawSearchDistribution)
						{
							Vector delta(2);
							delta(0) = targetposX - x;
							delta(1) = targetposY - y;
							delta *= 0.00390625;		// 1/256
							double kernel = exp(-0.5 * (delta * (C_inv * delta)));
							if (kernel > 0.01)
							{
								int add = (int)(100.0 * kernel);
								int red = (col & 0x00ff0000) >> 16;
								if (add + red > 255) add = 255 - red;
								col += 0x00010000 * add;
							}
						}
						if (nb < 0) nb = 0;
						for (; b>nb; b--) m_image.setPixel(p, b-1, col);
						y = nb;
					}
				}
			}
			for (; b>0; b--) m_image.setPixel(p, b-1, 0xffc8d0ff);
		}
	}

	if (m_monitorPopulation)
	{
		Array<double> population;
		if (m_experiment->m_algorithm.getObservation(&PropertyDesc::population, population) && population.ndim() == 2)
		{
			QPainter painter(&m_image);
			painter.setPen(Qt::white);
			painter.setBrush(QBrush(Qt::white));
			int i, ic = population.dim(0);
			for (i=0; i<ic; i++)
			{
				double x = (population(i, 0) + 2.0) * 256.0;
				double y = (population(i, 1) + 2.0) * 256.0;
				double z;
				if (canyon != NULL)
				{
					unsigned int col;
					canyon->get(x, y, z, col);
				}
				else z = get(0.5 * x, 0.5 * y);
				x -= viewposX;
				y -= viewposY;
				z -= viewposZ;
				double d = ca * x + sa * y;
				double l = -sa * x + ca * y;
				int diam = (int)Shark::round(200.0 / d);
				if (d <= 0.0) continue;
				int by = (int)Shark::round(viewdirZ - 128.0 * z / d);
				int bx = (int)Shark::round(127.5 * (1.0 - 2.0 * l / d));
				painter.drawEllipse(QRect(bx - diam/2, by - diam/2, diam, diam));
			}
		}
	}
}

QSize LandscapeView3D::sizeHint() const
{
	return QSize(512, 512);
}

void LandscapeView3D::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);

	int sx = width();
	int sy = height();
	int s = (sx > sy) ? sx : sy;
	int x = (sx - s) / 2;
	int y = (sy - s) / 2;
	painter.drawImage(QRect(x, y, s, s), m_image);
}

void LandscapeView3D::resizeEvent(QResizeEvent* event)
{
	ComputeView();
}
