//===========================================================================
/*!
 *  \file ArtificialDistributions.cpp
 *
 *  \brief Artificial benchmark data
 *
 *  \author  T. Glasmachers
 *  \date    2006
 *
 *  \par Copyright (c) 1999-2009:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
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

#include <Rng/GlobalRng.h>
#include <ReClaM/ArtificialDistributions.h>
#include <LinAlg/LinAlg.h>


Chessboard::Chessboard(int dim, int size)
{
	dataDim = dim;
	targetDim = 1;
	this->size = size;
}

Chessboard::~Chessboard()
{
}


bool Chessboard::GetData(Array<double>& data, Array<double>& target, int count)
{
	data.resize(count, dataDim, false);
	target.resize(count, 1, false);

	int i, j;
	double v;
	int t;
	for (i = 0; i < count; i++)
	{
		t = 0;
		for (j = 0; j < dataDim; j++)
		{
			v = Rng::uni(0.0, size);
			t += (int)floor(v);
			data(i, j) = v;
		}
		target(i, 0) = (t & 1) ? -1.0 : 1.0;
	}

	return true;
}


////////////////////////////////////////////////////////////


NoisyChessboard::NoisyChessboard(int dim, int size, double noiselevel)
: Chessboard(dim, size)
{
	this->noiselevel = noiselevel;
}

NoisyChessboard::~NoisyChessboard()
{
}


bool NoisyChessboard::GetData(Array<double>& data, Array<double>& target, int count)
{
	Chessboard::GetData(data, target, count);
	int i, ic = target.dim(0);
	for (i=0; i<ic; i++)
	{
		if (Rng::uni(0.0, 1.0) < noiselevel) target(i, 0) = 2 * Rng::discrete(0, 1) - 1;
	}

	return true;
}


////////////////////////////////////////////////////////////


NoisyInterval::NoisyInterval(double bayesRate, int dimensions)
{
	if (bayesRate < 0.0) bayesRate = 0.0;
	if (bayesRate > 0.5) bayesRate = 0.5;
	this->bayesRate = bayesRate;
	this->dimensions = dimensions;

	dataDim = dimensions;
	targetDim = 1;
}

NoisyInterval::~NoisyInterval()
{
}


bool NoisyInterval::GetData(Array<double>& data, Array<double>& target, int count)
{
	data.resize(count, dimensions, false);
	target.resize(count, 1, false);

	int i, d;
	double x, z;
	double p, q;

	if (bayesRate == 0.0) { p = 1.0; q = 0.0; }
	else if (bayesRate < 0.25) { p = 0.25; q = bayesRate; }
	else if (bayesRate < 0.5) { p = 2.0 - 4.0 * bayesRate; q = 1.0; }
	else { p = 0.0; q = 1.0; }

	for (i = 0; i < count; i++)
	{
		x = Rng::uni(-1.0, 1.0);
		z = Rng::uni(-1.0, 1.0);
		data(i, 0) = x;
		target(i, 0) = (p * x <= q * z) ? -1.0 : + 1.0;
		for (d=1; d<dimensions; d++) data(i, d) = Rng::uni(-1.0, 1.0);
	}

	return true;
}


////////////////////////////////////////////////////////////


SphereDistribution1::SphereDistribution1(int dim)
{
	dimension = dim;

	dataDim = dim;
	targetDim = 1;
}

SphereDistribution1::~SphereDistribution1()
{
}


bool SphereDistribution1::GetData(Array<double>& data, Array<double>& target, int count)
{
	data.resize(count, dimension, false);
	target.resize(count, 1, false);
	int i, d;
	for (i = 0; i < count; i++)
	{
		double norm = Rng::uni(0.0, 2.0);
		if (norm < 1.0)
		{
			target(i, 0) = 1.0;
		}
		else
		{
			target(i, 0) = -1.0;
			norm += 1.0;
		}
		double len2 = 0.0;
		for (d = 0; d < dimension; d++)
		{
			double z = Rng::gauss();
			data(i, d) = z;
			len2 += z * z;
		}
		double len = sqrt(len2);
		for (d = 0; d < dimension; d++)
		{
			data(i, d) *= norm / len;
		}
	}
	return true;
}


////////////////////////////////////////////////////////////


SparseDistribution::SparseDistribution(int n, int m, int k)
{
	dim1 = n;
	dim2 = m;
	num = k;

	dataDim = 2*n+m;
	targetDim = 1;
}

SparseDistribution::~SparseDistribution()
{
}


bool SparseDistribution::GetData(Array<double>& data, Array<double>& target, int count)
{
	int i, j, jc = 2 * dim1 + dim2;
	int f;
	bool b;
	std::vector<int> c;

	data.resize(count, jc, false);
	target.resize(count, 1, false);
	data = 0.0;

	for (i=0; i<count; i++)
	{
		c.resize(jc);
		for (j=0; j<jc; j++) c[j] = j;
		b = Rng::coinToss();
		f = Rng::discrete(0, dim1 - 1);
		if (b) target(i, 0) = 1.0;
		else
		{
			f += dim1;
			target(i, 0) = -1.0;
		}
		data(i, f) = 1.0;
		c.erase(c.begin() + f);
		for (j=1; j<num; j++)
		{
			f = Rng::discrete(0, c.size() - 1);
			data(i, c[f]) = 1.0;
			c.erase(c.begin() + f);
		}
	}

	return true;
}


////////////////////////////////////////////////////////////


TransformedProblem::TransformedProblem(DataSource& source, Array2D<double>& transformation)
: base(source)
{
	this->transformation = transformation;

	SIZE_CHECK(transformation.ndim() == 2);
	SIZE_CHECK((int)transformation.dim(1) == source.getDataDimension());

	dataDim = transformation.dim(0);
	targetDim = source.getTargetDimension();
}

TransformedProblem::~TransformedProblem()
{
}


bool TransformedProblem::GetData(Array<double>& data, Array<double>& target, int count)
{
	Array<double> x;
	if (! base.GetData(x, target, count)) return false;

	data.resize(count, dataDim, false);

	int i;
	for (i=0; i<count; i++)
	{
		matColVec(data[i], transformation, x[i]);
	}

	return true;
}


////////////////////////////////////////////////////////////


MultiClassTestProblem::MultiClassTestProblem(double epsilon)
{
	dataDim = 1;
	targetDim = 1;
	m_epsilon = epsilon;
}

MultiClassTestProblem::~MultiClassTestProblem()
{
}


bool MultiClassTestProblem::GetData(Array<double>& data, Array<double>& target, int count)
{
	data.resize(count, 1, false);
	target.resize(count, 1, false);
	int i;
	for (i=0; i<count; i++)
	{
		double r = Rng::uni(0.0, 1.0);
		if (r < 0.5 * m_epsilon)
		{
			data(i, 0) = 1.549 + Rng::gauss(0.0, 1e-8);
			target(i, 0) = 1.0;
		}
		else if (r < m_epsilon)
		{
			data(i, 0) = 1.551 + Rng::gauss(0.0, 1e-8);
			target(i, 0) = 2.0;
		}
		else
		{
			data(i, 0) = (asin(Rng::uni(-1.0, 1.0)) / M_PI) - 1.5;
			target(i, 0) = 0.0;
		}
	}

	return true;
}
