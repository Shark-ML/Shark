//===========================================================================
/*!
 *  \file Canyon.h
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


#ifndef _Canyon_H_
#define _Canyon_H_


#include <vector>
#include <EALib/ObjectiveFunction.h>


class CanyonConstraint : public ConstraintHandler<double*>
{
public:
	CanyonConstraint();
	~CanyonConstraint();

	bool isFeasible(double* const& point) const;
	bool closestFeasible(double*& point) const;
};


class Canyon : public ObjectiveFunctionVS<double>
{
public:
	Canyon();
	~Canyon();

	unsigned int objectives() const;

	void result(double* const& point, std::vector<double>& value);
	void get(double x, double y, double& height, unsigned int& color) const;

	bool ProposeStartingPoint(double*& point) const;

protected:
	struct iPos
	{
		int x;
		int y;
	};

	void River(iPos& pos, int to_x, int to_y, double& radius, double& level, std::vector<iPos>& river);
	void Canyonize(int x, int y, double radius, double level);
	void Smoothen(int x, int y);
	void CreateLandscape();

	// landscape description
	Array<double> fractal;
	Array<double> height;
	Array<unsigned int> color;

	// population
	struct tPos
	{
		double x;
		double y;
	};
	std::vector<tPos> individual;

	// Gaussian search distribution
	double meanX;
	double meanY;
	double covarXX;
	double covarXY;
	double covarYY;

	double filter[11][11];

	CanyonConstraint constraint;
};


// class LandscapeView
// {
// public:
// 	LandscapeView();
// 	~LandscapeView();
// 
// 	// compute a 256 x 256 pixel image
// 	void ComputeView(const Scene& scene, unsigned int* image);
// 
// 	double posX;
// 	double posY;
// 	double posZ;
// 	double direction;
// 
// protected:
// 	// precomputed tables
// 	double dist[2048];
// 	double right[256];
// };


// CMA killer function!!!
class Edge : public ObjectiveFunctionVS<double>
{
public:
	Edge()
	: ObjectiveFunctionVS<double>(2)
	{ }

	~Edge()
	{ }

	unsigned int objectives() const { return 1; }

	void result(double* const& point, std::vector<double>& value)
	{
		value.resize(1);
		value[0] = fabs(point[1] + fabs(point[0])) - 0.01 * point[0] + 1.0;
		m_timesCalled++;
	}

	bool ProposeStartingPoint(double*& point) const
	{
		point[0] = Rng::uni(-2.0, -1.0);
		point[1] = Rng::uni(-2.0, -1.0);
		return true;
	}
};


#endif
