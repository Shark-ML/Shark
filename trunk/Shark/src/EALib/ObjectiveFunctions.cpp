//===========================================================================
/*!
 *  \file ObjectiveFunctions.cpp
 *
 *  \brief standard benchmark functions
 *
 *  \author  Christian Igel, Tobias Glasmachers
 *  \date    2008
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
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
 *
 *
 */
//===========================================================================


#include <EALib/ObjectiveFunctions.h>
#include <LinAlg/LinAlg.h>


Sphere::Sphere(unsigned d) : ObjectiveFunctionVS<double>(d, NULL)
{
	m_name = "Sphere";
}

Sphere::~Sphere()
{}


unsigned int Sphere::objectives() const
{
	return 1;
}

void Sphere::result(double* const& point, std::vector<double>& value)
{
	unsigned i;
	double sum = 0.;
	for (i = 0; i < m_dimension; i++) sum += point[i] * point[i];
	value.resize(1);
	value[0] = sum;
	m_timesCalled++;
}

bool Sphere::ProposeStartingPoint(double*& point) const
{
	double r2 = 0.0;
	unsigned int i;
	for (i = 0; i < m_dimension; i++)
	{
		double a = Rng::gauss();
		point[i] = a;
		r2 += a * a;
	}
	double r = sqrt(r2);
	for (i = 0; i < m_dimension; i++) point[i] /= r;
	return true;
}

bool Sphere::utopianFitness(std::vector<double>& fitness) const
{
	fitness.resize(1, false);
	fitness[0] = 0.0;
	return true;
}


////////////////////////////////////////////////////////////


NoisySphere::NoisySphere(unsigned d) : ObjectiveFunctionVS<double>(d, NULL)
{
	m_name = "NoisySphere";
}

NoisySphere::~NoisySphere()
{}


unsigned int NoisySphere::objectives() const
{
	return 1;
}

void NoisySphere::result(double* const& point, std::vector<double>& value)
{
	unsigned i;
	double sum = 0.;
	for (i = 0; i < m_dimension; i++) sum += point[i] * point[i];
	value.resize(1);
	value[0] = sum + sum / (2. * m_dimension) * Rng::cauchy();
	m_timesCalled++;
}

bool NoisySphere::ProposeStartingPoint(double*& point) const
{
	double r2 = 0.0;
	unsigned int i;
	for (i = 0; i < m_dimension; i++)
	{
		double a = Rng::gauss();
		point[i] = a;
		r2 += a * a;
	}
	double r = sqrt(r2);
	for (i = 0; i < m_dimension; i++) point[i] /= r;
	return true;
}


////////////////////////////////////////////////////////////


Paraboloid::Paraboloid(unsigned d, double c)
		: TransformedObjectiveFunction(base, d)
		, base(d)
{
	m_name = "Paraboloid";

	Array<double> tmp(m_dimension, m_dimension);
	Array<double> diag(m_dimension, m_dimension);
	diag = 0.0;
	unsigned i;
	for (i = 0; i < m_dimension; i++) diag(i, i) = pow(c, (double(i) / double(m_dimension - 1)));
	matMat(tmp, m_Transformation, diag);
	m_Transformation = tmp;
}

Paraboloid::~Paraboloid()
{}


////////////////////////////////////////////////////////////


Tablet::Tablet(unsigned d, double c) : TransformedObjectiveFunction(base, d)
		, base(d)
{
	m_name = "Tablet";

	Array<double> tmp(m_dimension, m_dimension);
	Array<double> diag(m_dimension, m_dimension);
	diag = 0.0;
	diag(0, 0) = c;

	for (unsigned i = 1; i < m_dimension; i++)
		diag(i, i) = 1.0;

	matMat(tmp, m_Transformation, diag);
	m_Transformation = tmp;
}

Tablet::~Tablet()
{}


////////////////////////////////////////////////////////////


Cigar::Cigar(unsigned d, double c) : TransformedObjectiveFunction(base, d)
		, base(d)
{
	m_name = "Cigar";

	Array<double> tmp(m_dimension, m_dimension);
	Array<double> diag(m_dimension, m_dimension);
	diag = 0.0;
	diag(0, 0) = 1;

	for (unsigned i = 1; i < m_dimension; i++)
		diag(i, i) = c;

	matMat(tmp, m_Transformation, diag);
	m_Transformation = tmp;
}

Cigar::~Cigar()
{}


////////////////////////////////////////////////////////////


Twoaxis::Twoaxis(unsigned d, double c) : TransformedObjectiveFunction(base, d)
		, base(d)
{
	m_name = "Twoaxis";

	Array<double> tmp(m_dimension, m_dimension);
	Array<double> diag(m_dimension, m_dimension);
	diag = 0.0;

	unsigned i;
	for (i = 0; i < m_dimension / 2; i++)
		diag(i, i) = c;
	for (i++; i < m_dimension; i++)
		diag(i, i) = 1;

	matMat(tmp, m_Transformation, diag);
	m_Transformation = tmp;
}

Twoaxis::~Twoaxis()
{}


////////////////////////////////////////////////////////////


Ackley::Ackley(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, -32.0, 32.0))
{
	m_name = "Ackley's function";
}

Ackley::~Ackley()
{
	delete constrainthandler;
}


unsigned int Ackley::objectives() const
{
	return 1;
}

void Ackley::result(double* const& point, std::vector<double>& value)
{

	const double A = 20.;
	const double B = 0.2;
	const double C = M_2PI;

	unsigned i;
	double   a, b;

	for (a = b = 0.0, i = 0; i < m_dimension; ++i)
	{
		a += point[i] * point[i];
		b += cos(C * point[i]);
	}

	value.resize(1);
	value[0] = -A * exp(-B * sqrt(a / m_dimension)) - exp(b / m_dimension) + A + M_E;

	m_timesCalled++;
}

bool Ackley::ProposeStartingPoint(double*& point) const
{

	const double bound = 32.0;

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = 2 * bound * Rng::uni() - bound;

	return true;
}

bool Ackley::utopianFitness(std::vector<double>& fitness) const
{
	fitness.resize(1, false);
	fitness[0] = 0.0;
	return true;
}


////////////////////////////////////////////////////////////


Rastrigin::Rastrigin(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, -5.12, 5.12))
{
	m_name = "Rastrigin's function";
}

Rastrigin::~Rastrigin()
{
	delete constrainthandler;
}


unsigned int Rastrigin::objectives() const
{
	return 1;
}

void Rastrigin::result(double* const& point, std::vector<double>& value)
{

	double result = 10.0 * m_dimension;

	for (unsigned int i = 0; i < m_dimension; i++)
		result += point[i] * point[i] - 10.0 * cos(M_2PI * point[i]);

	value.resize(1);
	value[0] = result;

	m_timesCalled++;
}

bool Rastrigin::ProposeStartingPoint(double*& point) const
{

	const double bound = 5.12;

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = 2 * bound * Rng::uni() - bound;

	return true;
}

bool Rastrigin::utopianFitness(std::vector<double>& fitness) const
{
	fitness.resize(1, false);
	fitness[0] = 0.0;
	return true;
}


////////////////////////////////////////////////////////////


Griewangk::Griewangk(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, -600.0, 600.0))
{
	m_name = "Griewangk's function";
}

Griewangk::~Griewangk()
{
	delete constrainthandler;
}


unsigned int Griewangk::objectives() const
{
	return 1;
}

void Griewangk::result(double* const& point, std::vector<double>& value)
{

	double sum  = 0.0;
	double prod = 1.0;

	for (unsigned int i = 0; i < m_dimension; i++)
	{
		sum  += point[i] * point[i];
		prod *= cos(point[i] / sqrt((double)(i + 1)));
	}

	value.resize(1);
	value[0] = 1.0 + sum / 4000.0 - prod;

	m_timesCalled++;
}

bool Griewangk::ProposeStartingPoint(double*& point) const
{

	const double bound = 600.0;

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = 2 * bound * Rng::uni() - bound;

	return true;
}

bool Griewangk::utopianFitness(std::vector<double>& fitness) const
{
	fitness.resize(1, false);
	fitness[0] = 0.0;
	return true;
}


////////////////////////////////////////////////////////////


Rosenbrock::Rosenbrock(unsigned d) : ObjectiveFunctionVS<double>(d, NULL)
{
	m_name = "Rosenbrock's function";
}

Rosenbrock::~Rosenbrock()
{}


unsigned int Rosenbrock::objectives() const
{
	return 1;
}

void Rosenbrock::result(double* const& point, std::vector<double>& value)
{

	double result = 0.0;

	for (unsigned i = 0; i < m_dimension - 1; i++)
		result += 100.0 * (point[i+1] - point[i] * point[i]) *
				  (point[i+1] - point[i] * point[i]) +
				  (point[i] - 1.0) * (point[i] - 1.0);

	value.resize(1);
	value[0] = result;

	m_timesCalled++;
}

bool Rosenbrock::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::gauss();

	return true;
}

bool Rosenbrock::utopianFitness(std::vector<double>& fitness) const
{
	fitness.resize(1, false);
	fitness[0] = 0.0;
	return true;
}


////////////////////////////////////////////////////////////


RosenbrockRotated:: RosenbrockRotated(unsigned d): TransformedObjectiveFunction(base, d)
		, base(d)
{
	m_name = "DifferentPowersRotated";
}

RosenbrockRotated::~RosenbrockRotated()
{}


////////////////////////////////////////////////////////////


DiffPow::DiffPow(unsigned d) : ObjectiveFunctionVS<double>(d, NULL)
{
	m_name = "DifferentPowers";
}

DiffPow::~DiffPow()
{}


unsigned int DiffPow::objectives() const
{
	return 1;
}

void DiffPow::result(double* const& point, std::vector<double>& value)
{

	double result = 0.0;

	for (unsigned i = 0; i < m_dimension; i++)
		result += pow(fabs(point[i]), 2.0 + 10.0 * i / (m_dimension - 1.0));

	value.resize(1);
	value[0] = result;

	m_timesCalled++;
}

bool DiffPow::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::gauss();

	return true;
}

bool DiffPow::utopianFitness(std::vector<double>& fitness) const
{
	fitness.resize(1, false);
	fitness[0] = 0.0;
	return true;
}


////////////////////////////////////////////////////////////


DiffPowRotated::DiffPowRotated(unsigned d): TransformedObjectiveFunction(base, d)
		, base(d)
{
	m_name = "DifferentPowersRotated";
}

DiffPowRotated::~DiffPowRotated()
{}


////////////////////////////////////////////////////////////


Schwefel::Schwefel(unsigned d) : ObjectiveFunctionVS<double>(d, NULL)
{
	m_name = "Rosenbrock's function";
}

Schwefel::~Schwefel()
{}


unsigned int Schwefel::objectives() const
{
	return 1;
}

void Schwefel::result(double* const& point, std::vector<double>& value)
{

	double result = 0.0;

	for (unsigned i = 0; i < m_dimension; i++)
		result += point[i] * sin(sqrt(fabs(point[i])));

	value.resize(1);
	value[0] = -result;

	m_timesCalled++;
}

bool Schwefel::ProposeStartingPoint(double*& point) const
{

	const double bound = 500.0;

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = 2 * bound * Rng::uni() - bound;

	return true;
}


////////////////////////////////////////////////////////////


SchwefelEllipsoid::SchwefelEllipsoid(unsigned d) : ObjectiveFunctionVS<double>(d, NULL)
{
	m_name = "SchwefelEllipsoid";
}

SchwefelEllipsoid::~SchwefelEllipsoid()
{}


unsigned int SchwefelEllipsoid::objectives() const
{
	return 1;
}

void SchwefelEllipsoid::result(double* const& point, std::vector<double>& value)
{
	unsigned i, j;
	double sum = 0.;
	double sumHelp;
	for (i = 0; i < m_dimension; i++)
	{
		sumHelp = 0.;
		for (j = 0; j <= i; j++)
		{
			sumHelp += point[j];
		}
		sum += sumHelp * sumHelp;
	}
	value.resize(1);
	value[0] = sum;
	m_timesCalled++;
}

bool SchwefelEllipsoid::ProposeStartingPoint(double*& point) const
{
	unsigned int i;
	for (i = 0; i < m_dimension; i++) point[i] = 1.;
	return true;
}


bool SchwefelEllipsoid::utopianFitness(std::vector<double>& fitness) const
{
	fitness.resize(1, false);
	fitness[0] = 0.0;
	return true;
}


////////////////////////////////////////////////////////////


SchwefelEllipsoidRotated::SchwefelEllipsoidRotated(unsigned d)
		: TransformedObjectiveFunction(base, d)
		, base(d)
{
	m_name = "SchwefelEllipsoidRotated";
}

SchwefelEllipsoidRotated::~SchwefelEllipsoidRotated()
{}


////////////////////////////////////////////////////////////


RandomFitness::RandomFitness(unsigned d) : ObjectiveFunctionVS<double>(d, NULL)
{
	m_name = "Random Fitness";
}

RandomFitness::~RandomFitness()
{}


unsigned int RandomFitness::objectives() const
{
	return 1;
}

void RandomFitness::result(double* const& point, std::vector<double>& value)
{
	value.resize(1);
	value[0] = Rng::gauss();
	m_timesCalled++;
}

bool RandomFitness::ProposeStartingPoint(double*& point) const
{
	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::gauss();
	return true;
}
