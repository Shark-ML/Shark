//===========================================================================
/*!
 *  \file StatusWidget.cpp
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


#include <SharkDefs.h>

#include <Rng/GlobalRng.h>
#include "Canyon.h"
#include "StatusWidget.h"


int is_pos(double t)
{
	return (t > 0.0) ? 1 : 0;
}

int mkcolor(double t)
{
	if (t < 0.0) return 0;
	else if (t > 255.0) return 255;
	else return (int)t;
}


////////////////////////////////////////////////////////////


CanyonConstraint::CanyonConstraint()
{
}

CanyonConstraint::~CanyonConstraint()
{
}


bool CanyonConstraint::isFeasible(double* const& point) const
{
	return (point[0] >= -2.0 && point[0] < 2.0 && point[1] >= -2.0 && point[1] < 2.0);
}

bool CanyonConstraint::closestFeasible(double*& point) const
{
	if (point[0] < -2.0) point[0] = -2.0;
	else if (point[0] > 1.999) point[0] = 1.999;
	if (point[1] < -2.0) point[1] = -2.0;
	else if (point[1] > 1.999) point[1] = 1.999;
	return true;
}


////////////////////////////////////////////////////////////


Canyon::Canyon()
: ObjectiveFunctionVS<double>(2, &constraint)
{
	m_name = "canyon objective function";

	int x, y;
	for (y=0; y<11; y++)
	{
		for (x=0; x<11; x++)
		{
			double d = sqrt((double)((x-5)*(x-5) + (y-5)*(y-5)));
			if (d > 5.5) filter[y][x] = 0.0;
			else { double c = cos(0.285599332144526658 * d); filter[y][x] = c * c; }
		}
	}

	CreateLandscape();
}

Canyon::~Canyon()
{
}


void Canyon::Canyonize(int x, int y, double radius, double level)
{
	double depth = fractal(y, x) - level;
	height(y, x) = level;

	int xx, yy;
	for (yy=-50; yy<=50; yy++)
	{
		int yf = y+yy;
		if (yf<0 || yf>1023) continue;
		for (xx=-50; xx<=50; xx++)
		{
			int xf = x+xx;
			if (xf<0 || xf>1023) continue;

			double d = sqrt((double)(xx*xx + yy*yy));
			if (d > radius) continue;

			double p = pow(d / radius, 3.0);
			if (Rng::uni() > p)
			{
				double f = pow(d / radius, 1.2) * radius;
				double h = fractal(yf, xf) - depth + 0.3 * f;
				double min_h = level + 0.2 * f;
				if (h < min_h) h = min_h;
				if (h < height(yf, xf)) height(yf, xf) = h;
			}
		}
	}
}

void Canyon::River(iPos& pos, int to_x, int to_y, double& radius, double& level, std::vector<iPos>& river)
{
	int dx = to_x - pos.x;
	int dy = to_y - pos.y;
	int target = dx * to_x + dy * to_y;
	int sign = 1;
	if (dx * pos.x + dy * pos.y > target) sign = -1;

	while (sign * (dx * pos.x + dy * pos.y) < (sign * target))
	{
		river.push_back(pos);
		Canyonize(pos.x, pos.y, radius, level); level -= 0.05;

		if (abs(dx) > abs(dy))
		{
			int d = dx / abs(dx);
			pos.x += d;
			double delta = (double)d * (double)dy / (double)dx + Rng::uni(-2.0, 2.0);
			if (delta < -1.0) pos.y--;
			else if (delta >= 1.0) pos.y++;
			else if (delta < 0.0 && Rng::uni() < -delta) pos.y--;
			else if (Rng::uni() < delta) pos.y++;
		}
		else
		{
			int d = dy / abs(dy);
			pos.y += d;
			double delta = (double)d * (double)dx / (double)dy + Rng::uni(-2.0, 2.0);
			if (delta < -1.0) pos.x--;
			else if (delta >= 1.0) pos.x++;
			else if (delta < 0.0 && Rng::uni() < -delta) pos.x--;
			else if (Rng::uni() < delta) pos.x++;
		}

		radius += Rng::gauss(0.0, 50.0) - tanh(0.1 * (radius - 30.0));
		if (radius < 10.0) radius = 10.0;
		else if (radius > 50.0) radius = 50.0;
	}
}

#define THRESHOLD 5.0
void Canyon::Smoothen(int x, int y)
{
	double h = height(y, x);
	int xx, yy;
	bool b = false;

	if (x > 0) b = b | (fabs(height(y, x-1) - h) > THRESHOLD);
	if (x < 1023) b = b | (fabs(height(y, x+1) - h) > THRESHOLD);
	if (y > 0) b = b | (fabs(height(y-1, x) - h) > THRESHOLD);
	if (y < 1023) b = b | (fabs(height(y+1, x) - h) > THRESHOLD);
	if (! b) return;

	double m = 0.0;
	double w = 0.0;
	for (yy=0; yy<11; yy++)
	{
		int yf = y + yy - 5;
		if (yf < 0 || yf > 1023) continue;
		for (xx=0; xx<11; xx++)
		{
			int xf = x + xx - 5;
			if (xf < 0 || xf > 1023) continue;
			w += filter[yy][xx];
			m += filter[yy][xx] * height(yf, xf);
		}
	}
	m /= w;
	double d = h - m;

	height(y, x) -= d;
	for (yy=0; yy<11; yy++)
	{
		int yf = y + yy - 5;
		if (yf < 0 || yf > 1023) continue;
		for (xx=0; xx<11; xx++)
		{
			int xf = x + xx - 5;
			if (xf < 0 || xf > 1023) continue;

			height(yf, xf) += filter[yy][xx] * d / w;
		}
	}
}

#define RIVER_DIV 200.0
void Canyon::CreateLandscape()
{
	StatusWidget status("Creating Canyon ...");
	status.show();

	int i, x, y;

	fractal.resize(1024, 1024, false);
	height.resize(1024, 1024, false);
	color.resize(1024, 1024, false);

	// create a standard 2D fractal landscape
	status.setText("creating the fractal landscape");
	double m = 0.0;
	double M = 0.0;
	fractal(0, 0) = 0.0;
	for (i=512; i>0; i/=2)
	{
		for (y=0; y<1024; y+=2*i)
		{
			int yi = (y + 2*i) & 1023;
			for (x=0; x<1024; x+=2*i)
			{
				int xi = (x + 2*i) & 1023;

				double h;
				double h00 = fractal(y, x);
				double h01 = fractal(y, xi);
				double h10 = fractal(yi, x);
				double h11 = fractal(yi, xi);

				h = 0.5 * (h00 + h01) + i * Rng::gauss();
				fractal(y, x+i) = h;
				if (h < m) m = h;
				if (h > M) M = h;

				h = 0.5 * (h10 + h11) + i * Rng::gauss();
				fractal(yi, x+i) = h;
				if (h < m) m = h;
				if (h > M) M = h;

				h = 0.5 * (h00 + h10) + i * Rng::gauss();
				fractal(y+i, x) = h;
				if (h < m) m = h;
				if (h > M) M = h;

				h = 0.5 * (h01 + h11) + i * Rng::gauss();
				fractal(y+i, xi) = h;
				if (h < m) m = h;
				if (h > M) M = h;

				h = 0.25 * (h00 + h01 + h10 + h11) + i * Rng::gauss();
				fractal(y+i, x+i) = h;
				if (h < m) m = h;
				if (h > M) M = h;
			}
		}
	}
	for (y=0; y<1024; y++)
	{
		for (x=0; x<1024; x++)
		{
			height(y, x) = fractal(y, x) = 255.0 * (fractal(y, x) - m) / (M - m);
		}
	}

	// shape a river
	status.setText("shaping the river");
	std::vector<iPos> river;
	iPos pos;

	double z = 0.0;
	double rad = 30.0;
	pos.x = 0;
	pos.y = 200;

	// (0, 200) --> (800, 200)
	River(pos, 400, 300, rad, z, river);
	River(pos, 600, 200, rad, z, river);
	River(pos, 800, 200, rad, z, river);
	// (800, 200) --> (800, 600)
	River(pos, 800, 400, rad, z, river);
	River(pos, 700, 600, rad, z, river);
	// (800, 600) --> (200, 600)
	River(pos, 600, 700, rad, z, river);
	River(pos, 300, 600, rad, z, river);
	// (200, 600) --> (200, 800)
	River(pos, 200, 700, rad, z, river);
	River(pos, 200, 800, rad, z, river);

	// smoothen the edges
	status.setText("smoothening");
	std::vector<iPos> spos(1048576);
	i = 0;
	for (y=0; y<1024; y++)
	{
		for (x=0; x<1024; x++)
		{
			spos[i].x = x;
			spos[i].y = y;
			i++;
		}
	}
	for (i=0; i<1048576; i++)
	{
		x = Rng::discrete(0, 1048575);
		iPos tmp = spos[x];
		spos[x] = spos[i];
		spos[i] = tmp;
	}
	for (i=0; i<1048576; i++)
	{
		Smoothen(spos[i].x, spos[i].y);
	}

	// colorize
	status.setText("colorizing the landscape");
	double grass[3] = {96.0, 118.0, 45.0};
	double rock[3] = {105.0, 108.0, 117.0};
	double mud[3] = {111.0, 90.0, 52.0};

	for (y=0; y<1024; y++)
	{
		for (x=0; x<1024; x++)
		{
			int n = 0;
			double s = 0.0;
			int d = 0;
			double l = 0.0;
			int m = 0;
			double h = height(y, x);
			if (y > 0) { s += fabs(height(y-1, x) - h); d += is_pos(height(y-1, x) - h); n++; l += height(y-1, x) - h; m++; }
			if (y < 1023) { s += fabs(height(y+1, x) - h); d += is_pos(height(y+1, x) - h); n++; l += h - height(y+1, x); m++; }
			if (x > 0) { s += fabs(height(y, x-1) - h); d += is_pos(height(y, x-1) - h); n++; }
			if (x < 1023) { s += fabs(height(y, x+1) - h); d += is_pos(height(y, x+1) - h); n++; }
			s /= (double)n;
			l /= (double)m;

			double p_m = 0.0;
// 			double p_w = 0.0;
			double p_r = 0.0;

			if (s > 2.0) p_r = 2.0;
			else if (s > 1.0) p_r = (s - 1.0) / 1.0;
			if (h < 0.0) p_m = 0.5;
			else if (h < 128.0) p_m = (128.0 - h) / 256.0;
			else
			{
				double ppr = (h - 128.0) / 256.0;
				if (ppr > p_r) p_r = ppr;
			}

			// compose the color
			double red = (1.0 - p_m) * grass[0] + p_m * mud[0];
			double green = (1.0 - p_m) * grass[1] + p_m * mud[1];
			double blue = (1.0 - p_m) * grass[2] + p_m * mud[2];
// 			if (Rng::uni() < p_m)
// 			{
// 				red = mud[0];
// 				green = mud[1];
// 				blue = mud[2];
// 			}
			if (Rng::uni() < p_r)
			{
				red = rock[0] + 3.0 * Rng::gauss();
				green = rock[1] + 3.0 * Rng::gauss();
				blue = rock[2] + 3.0 * Rng::gauss();
			}

			double dl = 32.0 * tanh(l);
			red += dl;
			green += dl;
			blue += dl;

			unsigned int rr = mkcolor(red);
			unsigned int gg = mkcolor(green);
			unsigned int bb = mkcolor(blue);
			color(y, x) = 0xff000000 | (bb << 16) | (gg << 8) | rr;
		}
	}

	status.setText("colorizing the river");
	for (i=0; i<(int)river.size(); i++)
	{
		int xx, yy;
		for (yy=-1; yy<=1; yy++)
		{
			y = river[i].y + yy;
			if (y < 0 || y > 1023) continue;
			for (xx=-1; xx<=1; xx++)
			{
				x = river[i].x + xx;
				if (x < 0 || x > 1023) continue;
				color(y, x) = 0xffe07070;
			}
		}
	}

	status.setText("adding color noise");
	for (y=0; y<1024; y++)
	{
		for (x=0; x<1024; x++)
		{
			color(y, x) += Rng::discrete(0, 9) * 0x00010000
						 + Rng::discrete(0, 9) * 0x00000100
						 + Rng::discrete(0, 9) * 0x00000001;
		}
	}
}

unsigned int Canyon::objectives() const
{
	return 1;
}

void Canyon::get(double x, double y, double& height, unsigned int& color) const
{
	double h[4];
	int r[4];
	int g[4];
	int b[4];

	int x0 = (int)x;
	int y0 = (int)y;
	if (x0 < 0 || y0 < 0 || x0 > 1023 || y0 > 1023)
	{
		height = 256.0;
		color = 0x00000000;
		return;
	}

	int x1 = (x0 + 1) & 1023;
	int y1 = (y0 + 1) & 1023;
	double xr = x - x0;
	double yr = y - y0;

	h[0] = this->height(y0, x0);
	h[1] = this->height(y0, x1);
	h[2] = this->height(y1, x0);
	h[3] = this->height(y1, x1);
	r[0] = this->color(y0, x0) & 0x000000ff;
	r[1] = this->color(y0, x1) & 0x000000ff;
	r[2] = this->color(y1, x0) & 0x000000ff;
	r[3] = this->color(y1, x1) & 0x000000ff;
	g[0] = (this->color(y0, x0) & 0x0000ff00) >> 8;
	g[1] = (this->color(y0, x1) & 0x0000ff00) >> 8;
	g[2] = (this->color(y1, x0) & 0x0000ff00) >> 8;
	g[3] = (this->color(y1, x1) & 0x0000ff00) >> 8;
	b[0] = (this->color(y0, x0) & 0x00ff0000) >> 16;
	b[1] = (this->color(y0, x1) & 0x00ff0000) >> 16;
	b[2] = (this->color(y1, x0) & 0x00ff0000) >> 16;
	b[3] = (this->color(y1, x1) & 0x00ff0000) >> 16;

// 	if (h[0] < h[1] && h[0] < h[2] && h[3] < h[1] && h[3] < h[2])
// 	{
// 		if (xr > yr)
// 			height = 200.0 + (1.0 - xr) * h[0] + yr * h[3] + (xr - yr) * h[1];
// 		else
// 			height = 200.0 + (1.0 - yr) * h[0] + xr * h[3] + (yr - xr) * h[2];
// 	}
// 	else if (h[0] > h[1] && h[0] > h[2] && h[3] > h[1] && h[3] > h[2])
// 	{
// 		if (xr + yr > 1.0)
// 			height = 200.0 + (1.0 - xr) * h[2] + (1.0 - yr) * h[1] + (xr + yr - 1.0) * h[3];
// 		else
// 			height = 200.0 + xr * h[1] + yr * h[2] + (1.0 - xr - yr) * h[0];
// 	}
// 	else
		height = 200.0 + (1.0 - xr) * ((1.0 - yr) * h[0] + yr * h[2]) + xr * ((1.0 - yr) * h[1] + yr * h[3]);

	double rr = (1.0 - xr) * ((1.0 - yr) * r[0] + yr * r[2]) + xr * ((1.0 - yr) * r[1] + yr * r[3]);
	double gg = (1.0 - xr) * ((1.0 - yr) * g[0] + yr * g[2]) + xr * ((1.0 - yr) * g[1] + yr * g[3]);
	double bb = (1.0 - xr) * ((1.0 - yr) * b[0] + yr * b[2]) + xr * ((1.0 - yr) * b[1] + yr * b[3]);
	int ri = (int)rr;
	int gi = (int)gg;
	int bi = (int)bb;
// 	color = 0xff000000 | (bi << 16) | (gi << 8) | ri;
	color = 0xff000000 | (ri << 16) | (gi << 8) | bi;
}

void Canyon::result(double* const& point, std::vector<double>& value)
{
	value.resize(1);

	double x = 256.0 * (point[0] + 2.0);
	double y = 256.0 * (point[1] + 2.0);
	if (x < 0.0) x = 0.0;
	else if (x > 1023.99) x = 1023.99;
	if (y < 0.0) y = 0.0;
	else if (y > 1023.99) y = 1023.99;

	int x0 = (int)x;
	int y0 = (int)y;
	int x1 = (x0 + 1) & 1023;
	int y1 = (y0 + 1) & 1023;
	double xr = x - x0;
	double yr = y - y0;

	double height;
	double h[4];
	h[0] = this->height(y0, x0);
	h[1] = this->height(y0, x1);
	h[2] = this->height(y1, x0);
	h[3] = this->height(y1, x1);

	if (h[0] < h[1] && h[0] < h[2] && h[3] < h[1] && h[3] < h[2])
	{
		if (fabs(xr - yr) < 0.1) height = 200.0 + (1.0 - 0.5 * xr - 0.5 * yr) * h[0] + (0.5 * xr + 0.5 * yr) * h[3];
		else
		{
			if (xr > yr)
				height = 200.0 + (1.0 - xr) * h[0] + yr * h[3] + (xr - yr) * h[1];
			else
				height = 200.0 + (1.0 - yr) * h[0] + xr * h[3] + (yr - xr) * h[2];
		}
	}
	else if (h[0] > h[1] && h[0] > h[2] && h[3] > h[1] && h[3] > h[2])
	{
		if (fabs(xr + yr - 1.0) < 0.1) height = 200.0 + (0.5 + 0.5 * xr - 0.5 * yr) * h[1] + (0.5 - 0.5 * xr + 0.5 * yr) * h[2];
		else
		{
			if (xr + yr > 1.0)
				height = 200.0 + (1.0 - xr) * h[2] + (1.0 - yr) * h[1] + (xr + yr - 1.0) * h[3];
			else
				height = 200.0 + xr * h[1] + yr * h[2] + (1.0 - xr - yr) * h[0];
		}
	}
	else
		height = 200.0 + (1.0 - xr) * ((1.0 - yr) * h[0] + yr * h[2]) + xr * ((1.0 - yr) * h[1] + yr * h[3]);

	value[0] = height;
	m_timesCalled++;
}

bool Canyon::ProposeStartingPoint(double*& point) const
{
	point[0] = Rng::uni(-2.0, -0.0);
	point[1] = Rng::uni(-1.9, -1.3);
	return true;
}
