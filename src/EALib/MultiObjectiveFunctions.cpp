//===========================================================================
/*!
 *  \file MultiObjectiveFunctions.cpp
 *
 *  \brief standard multi-objective benchmark functions
 *
 *  \author  Bjoern Weghenkel, Asja Fischer
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


#include <EALib/MultiObjectiveFunctions.h>
#include <LinAlg/LinAlg.h>


bool BhinKornConstraintHandler::isFeasible(double* const& point) const
{
	unsigned int i;
	for (i = 0; i < m_dimension; i++) if (point[i] > 20.0 || point[i] < -20.0) return false;
	if ((pow(point[0], 2) + pow(point[1], 2) - 255.0) > 0) return false;
	else if (point[0] + 3*point[1] + 10.0 > 0) return false;

	return true;
}


////////////////////////////////////////////////////////////


BhinKorn::BhinKorn() : ObjectiveFunctionVS<double>(2, new BhinKornConstraintHandler())
{
	m_name = "BhinKorn";
}

BhinKorn::~BhinKorn()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int BhinKorn::objectives() const
{
	return 2;
}

void BhinKorn::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = pow(point[0] - 2.0, 2) + pow(point[1] - 1.0, 2) + 2.0;
	value[1] = 9 * point[0] - pow(point[1] - 1.0, 2);

	m_timesCalled++;
}

bool BhinKorn::ProposeStartingPoint(double*& point) const
{
	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(-6.0, 6.0);

	return true;
}

bool BhinKorn::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool BhinKorn::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 0.0;
	value[1] = 144.0;
	return true;
}


////////////////////////////////////////////////////////////


Schaffer::Schaffer() : ObjectiveFunctionVS<double>(1, new BoxConstraintHandler(1, -6.0, 6.0))
{
	m_name = "Schaffer`s f2";
}

Schaffer::~Schaffer()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int Schaffer::objectives() const
{
	return 2;
}

void Schaffer::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = pow(point[0], 2);
	value[1] = pow((point[0] - 2), 2);

	m_timesCalled++;
}

bool Schaffer::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(-6.0, 6.0);

	return true;
}

bool Schaffer::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool Schaffer::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 36.0;
	value[1] = 64.0;
	return true;
}


////////////////////////////////////////////////////////////


ZDT1::ZDT1(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "ZDT1";
}

ZDT1::~ZDT1()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZDT1::objectives() const
{
	return 2;
}

void ZDT1::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = point[0];

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += point[i];

	sum /= (m_dimension - 1.0);

	g = 1.0 + (9.0 * sum);
	h = 1.0 - sqrt(point[0] / g);

	value[1] = g * h;

	m_timesCalled++;
}

bool ZDT1::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool ZDT1::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZDT1::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}


////////////////////////////////////////////////////////////


ZDT2::ZDT2(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "ZDT2";
}

ZDT2::~ZDT2()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZDT2::objectives() const
{
	return 2;
}

void ZDT2::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = point[0];

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += point[i];

	sum /= (m_dimension - 1.0);

	g = 1.0 + (9.0 * sum);
	h = 1.0 - pow((point[0] / g), 2);

	value[1] = g * h;

	m_timesCalled++;
}

bool ZDT2::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool ZDT2::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZDT2::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}


////////////////////////////////////////////////////////////


ZDT3::ZDT3(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "ZDT3";
}

ZDT3::~ZDT3()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZDT3::objectives() const
{
	return 2;
}

void ZDT3::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = point[0];

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += point[i];

	sum /= (m_dimension - 1.0);

	g = 1.0 + (9.0 * sum);
	h = 1.0 - sqrt(point[0] / g) - (point[0] / g) * sin(10 * M_PI * point[0]);

	value[1] = g * h;

	m_timesCalled++;
}

bool ZDT3::ProposeStartingPoint(double*& point) const
{
	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool ZDT3::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZDT3::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}


////////////////////////////////////////////////////////////


ZDT4::ZDT4(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, -5.0, 5.0, 0, 0.0, 1.0))
{
	m_name = "ZDT4";
}

ZDT4::~ZDT4()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}

unsigned int ZDT4::objectives() const
{
	return 2;
}

void ZDT4::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = point[0];

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += pow(point[i], 2) - (10.0 * cos(4 * M_PI * point[i]));

	g = 1.0 + (10.0 * (m_dimension - 1.0)) + sum;
	h = 1.0 - sqrt(point[0] / g);

	value[1] = g * h;

	m_timesCalled++;
}

bool ZDT4::ProposeStartingPoint(double*& point) const
{

	point[0] = Rng::uni(0.0, 1.0);

	for (unsigned int i = 1; i < m_dimension; i++)
		point[i] = Rng::uni(-5.0, 5.0);

	return true;
}

bool ZDT4::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZDT4::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 1.0 + 10.0 * m_dimension;
	return true;
}


////////////////////////////////////////////////////////////


// /**
//  * \todo grenzen richtigrum?
//  */
// bool CuboidConstraintHandler::isFeasible(const double*& point) const
// {
// 	if (point[0] < m_lower_1 || point[0] > m_upper_1)
// 		return false;
// 
// 	for (unsigned i = 1; i < m_dimension; i++)
// 		if (point[i] < m_lower_2 || point[i] > m_upper_2)
// 			return false;
// 
// 	return true;
// }

// /**
//  * \todo grenzen richtigrum?
//  */
// bool CuboidConstraintHandler::closestFeasible(double*& point) const
// {
// 	if (point[0] < m_lower_1)
// 		point[0] = m_lower_1;
// 	else if (point[0] > m_upper_1)
// 		point[0] = m_upper_1;
// 
// 	for (unsigned i = 1; i < m_dimension; i++)
// 	{
// 		if (point[i] < m_lower_2)
// 			point[i] = m_lower_2;
// 		else if (point[i] > m_upper_2)
// 			point[i] = m_upper_2;
// 	}
// 
// 	return true;
// }


////////////////////////////////////////////////////////////


ZDT5::ZDT5(unsigned d) : ObjectiveFunctionVS<long>(d, new ZDT5ConstraintHandler(d, 0, long(0x3fffffff), 0, long(0x0000001f)))
{
	m_name = "ZDT5";
}

ZDT5::~ZDT5()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZDT5::objectives() const
{
	return 2;
}

void ZDT5::result(long* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = 1.0 + u(point[0]);

	double sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
	{
		if (u(point[i]) < 5)
			sum += 2 + u(point[i]);
		else
			sum += 1;
	}

	value[1] = sum * 1 / value[0];

	m_timesCalled++;
}

bool ZDT5::ProposeStartingPoint(long*& point) const
{

	point[0] = (int) Rng::uni(0, 0x3fffffff);

	for (unsigned i = 1; i < m_dimension; i++)
		point[i] = (int) Rng::uni(0, 0x0000001f);

	return true;
}

unsigned ZDT5::u(long x) const
{

	unsigned bits = 0;

	for (unsigned i = 0; i < 8 * sizeof(x); i++, x >>= 1)
		if (x % 2)
			bits++;

	return bits;
}


////////////////////////////////////////////////////////////


bool ZDT5ConstraintHandler::isFeasible(long* const& point) const
{
	if (point[0] < m_lower_1 || point[0] > m_upper_1)
		return false;

	for (unsigned i = 1; i < m_dimension; i++)
		if (point[i] < m_lower_2 || point[i] > m_upper_2)
			return false;

	return true;
}

bool ZDT5ConstraintHandler::closestFeasible(long*& point) const
{
	if (point[0] < m_lower_1)
		point[0] = m_lower_1;
	else if (point[0] > m_upper_1)
		point[0] = m_upper_1;

	for (unsigned i = 1; i < m_dimension; i++)
	{
		if (point[i] < m_lower_2)
			point[i] = m_lower_2;
		else if (point[i] > m_upper_2)
			point[i] = m_upper_2;
	}

	return true;
}


////////////////////////////////////////////////////////////


ZDT6::ZDT6(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "ZDT6";
}

ZDT6::~ZDT6()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZDT6::objectives() const
{
	return 2;
}

void ZDT6::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	value[0] = 1.0 - exp(-4.0 * point[0]) * pow(sin(6 * M_PI * point[0]), 6);

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += point[i];

	sum /= (m_dimension - 1.0);

	g = 1.0 + 9.0 * pow(sum, 0.25);
	h = 1.0 - pow(value[0] / g, 2);

	value[1] = g * h;

	m_timesCalled++;
}

bool ZDT6::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool ZDT6::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZDT6::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}


////////////////////////////////////////////////////////////


IHR1::IHR1(unsigned d): ObjectiveFunctionVS<double>(d , new BoxConstraintHandler(d, -1, 1)), m_Transformation(0, 0)
{
	m_name = "IHR1";
	initRandomRotation();
}

IHR1::~IHR1()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int IHR1::objectives() const
{
	return 2;
}

void IHR1::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	std::vector<double> y(m_dimension);
	transform(point, y);

	value[0] = fabs(y[0]);

	double g = 0;
	double ymax = fabs(m_Transformation(0, 0));

	for (unsigned i = 1; i < m_dimension; i++)
		ymax = std::max(fabs(m_Transformation(0, i)), ymax);
	ymax = 1 / ymax;

	for (unsigned i = 1; i < m_dimension; i++)
		g += hg(y[i]);
	g = 9 * g / (m_dimension - 1.) + 1.;

	value[1] = g * hf(1. - sqrt(h(y[0], m_dimension) / g), y[0], ymax);

	m_timesCalled++;
}

bool IHR1::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(-1.0, 1.0);

	return true;
}

bool IHR1::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool IHR1::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}

void IHR1::initRandomRotation()
{

	unsigned i, j, c;
	Matrix H(m_dimension, m_dimension);
	m_Transformation.resize(m_dimension, m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		for (c = 0; c < m_dimension; c++)
		{
			H(i, c) = Rng::gauss(0, 1);
		}
	}
	m_Transformation = H;
	for (i = 0; i < m_dimension; i++)
	{
		for (j = 0; j < i; j++)
			for (c = 0; c < m_dimension; c++)
// 				m_Transformation(i, c) -= scalarProduct(H[i], H[j]) * H(j, c) / scalarProduct(H[j], H[j]);
				m_Transformation(i, c) -= (H[i] * H[j]) * H(j, c) / (H[j].norm2());
		H = m_Transformation;
	}
	for (i = 0; i < m_dimension; i++)
	{
		double normB = m_Transformation[i].norm();
		for (j = 0; j < m_dimension; j++)
			m_Transformation(i, j) = m_Transformation(i, j) / normB;
	}
}

void IHR1::transform(const double* in, std::vector<double>& out) const
{
	unsigned i, j;
	out.resize(m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		out[i] = 0.0;
		for (j = 0; j < m_dimension; j++)
			out[i] += m_Transformation(j, i) * in[j];
	}
}

double IHR1::h(double x, double n)
{
	return 1 / (1 + exp(-x / sqrt(n)));
}

double IHR1::hf(double x, double y0, double ymax)
{
	if (fabs(y0) <=  ymax)
		return x;
	return fabs(y0) + 1;
}

double IHR1::hg(double x)
{
	return (x*x) / (fabs(x) + 0.1);
}


////////////////////////////////////////////////////////////


IHR2::IHR2(unsigned d): ObjectiveFunctionVS<double>(d , new BoxConstraintHandler(d, -1.0, 1.0)), m_Transformation(0, 0)
{
	m_name = "IHR2";
	initRandomRotation();
}

IHR2::~IHR2()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int IHR2::objectives() const
{
	return 2;
}

void IHR2::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	std::vector<double> y(m_dimension);
	transform(point, y);

	value[0] = fabs(y[0]);

	double g = 0;
	double ymax = fabs(m_Transformation(0, 0));

	for (unsigned i = 1; i < m_dimension; i++)
		ymax = std::max(fabs(m_Transformation(0, i)), ymax);
	ymax = 1 / ymax;

	for (unsigned i = 1; i < m_dimension; i++)
		g += hg(y[i]);
	g = 9 * g / (m_dimension - 1.) + 1.;


	value[1] = g * hf(1. - Shark::sqr(y[0] / g), y[0], ymax);

	m_timesCalled++;
}

bool IHR2::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(-1.0, 1.0);

	return true;
}

bool IHR2::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool IHR2::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}

void IHR2::initRandomRotation()
{

	unsigned i, j, c;
	Matrix H(m_dimension, m_dimension);
	m_Transformation.resize(m_dimension, m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		for (c = 0; c < m_dimension; c++)
		{
			H(i, c) = Rng::gauss(0, 1);
		}
	}
	m_Transformation = H;
	for (i = 0; i < m_dimension; i++)
	{
		for (j = 0; j < i; j++)
			for (c = 0; c < m_dimension; c++)
// 				m_Transformation(i, c) -= scalarProduct(H[i], H[j]) * H(j, c) / scalarProduct(H[j], H[j]);
				m_Transformation(i, c) -= (H[i] * H[j]) * H(j, c) / (H[j].norm2());
		H = m_Transformation;
	}
	for (i = 0; i < m_dimension; i++)
	{
		double normB = m_Transformation[i].norm();
		for (j = 0; j < m_dimension; j++)
			m_Transformation(i, j) = m_Transformation(i, j) / normB;
	}
}

void IHR2::transform(const double* in, std::vector<double>& out) const
{
	unsigned i, j;
	out.resize(m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		out[i] = 0.0;
		for (j = 0; j < m_dimension; j++)
			out[i] += m_Transformation(j, i) * in[j];
	}
}

double IHR2::hf(double x, double y0, double ymax)
{
	if (fabs(y0) <=  ymax)
		return x;
	return fabs(y0) + 1;
}

double IHR2::hg(double x)
{
	return (x*x) / (fabs(x) + 0.1);
}


////////////////////////////////////////////////////////////


IHR3::IHR3(unsigned d): ObjectiveFunctionVS<double>(d , new BoxConstraintHandler(d, -1.0, 1.0)), m_Transformation(0, 0)
{
	m_name = "IHR3";
	initRandomRotation();
}

IHR3::~IHR3()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int IHR3::objectives() const
{
	return 2;
}

void IHR3::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	std::vector<double> y(m_dimension);
	transform(point, y);

	value[0] = fabs(y[0]);

	double g = 0;
	double ymax = fabs(m_Transformation(0, 0));

	for (unsigned i = 1; i < m_dimension; i++)
		ymax = std::max(fabs(m_Transformation(0, i)), ymax);
	ymax = 1 / ymax;

	for (unsigned i = 1; i < m_dimension; i++)
		g += hg(y[i]);
	g = 9 * g / (m_dimension - 1.) + 1.;


	value[1] = g * hf(1. - sqrt(h(y[0], m_dimension) / g) - h(y[0], m_dimension) / g * sin(10 * M_PI * y[0]), y[0], ymax);

	m_timesCalled++;
}

bool IHR3::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(-1.0, 1.0);

	return true;
}

bool IHR3::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool IHR3::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}

void IHR3::initRandomRotation()
{

	unsigned i, j, c;
	Matrix H(m_dimension, m_dimension);
	m_Transformation.resize(m_dimension, m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		for (c = 0; c < m_dimension; c++)
		{
			H(i, c) = Rng::gauss(0, 1);
		}
	}
	m_Transformation = H;
	for (i = 0; i < m_dimension; i++)
	{
		for (j = 0; j < i; j++)
			for (c = 0; c < m_dimension; c++)
// 				m_Transformation(i, c) -= scalarProduct(H[i], H[j]) * H(j, c) / scalarProduct(H[j], H[j]);
				m_Transformation(i, c) -= (H[i] * H[j]) * H(j, c) / (H[j].norm2());
		H = m_Transformation;
	}
	for (i = 0; i < m_dimension; i++)
	{
		double normB = m_Transformation[i].norm();
		for (j = 0; j < m_dimension; j++)
			m_Transformation(i, j) = m_Transformation(i, j) / normB;
	}
}

void IHR3::transform(const double* in, std::vector<double>& out) const
{
	unsigned i, j;
	out.resize(m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		out[i] = 0.0;
		for (j = 0; j < m_dimension; j++)
			out[i] += m_Transformation(j, i) * in[j];
	}
}

double IHR3::h(double x, double n)
{
	return 1 / (1 + exp(-x / sqrt(n)));
}

double IHR3::hf(double x, double y0, double ymax)
{
	if (fabs(y0) <=  ymax)
		return x;
	return fabs(y0) + 1;
}

double IHR3::hg(double x)
{
	return (x*x) / (fabs(x) + 0.1);
}


////////////////////////////////////////////////////////////


IHR6::IHR6(unsigned d): ObjectiveFunctionVS<double>(d , new BoxConstraintHandler(d, -1.0, 1.0)), m_Transformation(0, 0)
{
	m_name = "IHR6";
	initRandomRotation();
}

IHR6::~IHR6()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int IHR6::objectives() const
{
	return 2;
}

void IHR6::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	std::vector<double> y(m_dimension);
	transform(point, y);

	value[0] = 1 - exp(-4 * fabs(y[0])) * pow(sin(6 * M_PI * y[0]), 6);

	double g = 0;
	double ymax = fabs(m_Transformation(0, 0));

	for (unsigned i = 1; i < m_dimension; i++)
		ymax = std::max(fabs(m_Transformation(0, i)), ymax);
	ymax = 1 / ymax;

	for (unsigned i = 1; i < m_dimension; i++)
		g += hg(y[i]);

	g = 1 + 9 * pow(g / (m_dimension - 1.0), 0.25);

	value[1] = g * hf(1. - Shark::sqr(value[0] / g), y[0], ymax);

	m_timesCalled++;
}

bool IHR6::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(-1.0, 1.0);

	return true;
}

bool IHR6::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool IHR6::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}

void IHR6::initRandomRotation()
{

	unsigned i, j, c;
	Matrix H(m_dimension, m_dimension);
	m_Transformation.resize(m_dimension, m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		for (c = 0; c < m_dimension; c++)
		{
			H(i, c) = Rng::gauss(0, 1);
		}
	}
	m_Transformation = H;
	for (i = 0; i < m_dimension; i++)
	{
		for (j = 0; j < i; j++)
			for (c = 0; c < m_dimension; c++)
// 				m_Transformation(i, c) -= scalarProduct(H[i], H[j]) * H(j, c) / scalarProduct(H[j], H[j]);
				m_Transformation(i, c) -= (H[i] * H[j]) * H(j, c) / (H[j].norm2());
		H = m_Transformation;
	}
	for (i = 0; i < m_dimension; i++)
	{
		double normB = m_Transformation[i].norm();
		for (j = 0; j < m_dimension; j++)
			m_Transformation(i, j) = m_Transformation(i, j) / normB;
	}
}

void IHR6::transform(const double* in, std::vector<double>& out) const
{
	unsigned i, j;
	out.resize(m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		out[i] = 0.0;
		for (j = 0; j < m_dimension; j++)
			out[i] += m_Transformation(j, i) * in[j];
	}
}

double IHR6::hf(double x, double y0, double ymax)
{
	if (fabs(y0) <=  ymax)
		return x;
	return fabs(y0) + 1;
}

double IHR6::hg(double x)
{
	return (x*x) / (fabs(x) + 0.1);
}


////////////////////////////////////////////////////////////


ZZJ07_F1 :: ZZJ07_F1(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "ZZJ07_F1";
}

ZZJ07_F1::~ZZJ07_F1()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZZJ07_F1::objectives() const
{
	return 2;
}

void ZZJ07_F1::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = point[0];

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += (point[i] - point[0]) * (point[i] - point[0]);

	sum /= (m_dimension - 1.0);

	g = 1.0 + (9.0 * sum);
	h = 1.0 - sqrt(point[0] / g);

	value[1] = g * h;

	m_timesCalled++;
}

bool ZZJ07_F1::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool ZZJ07_F1::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZZJ07_F1::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}


////////////////////////////////////////////////////////////


ZZJ07_F2 :: ZZJ07_F2(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "ZZJ07_F2";
}

ZZJ07_F2::~ZZJ07_F2()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZZJ07_F2::objectives() const
{
	return 2;
}

void ZZJ07_F2::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = point[0];

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += (point[i] - point[0]) * (point[i] - point[0]);

	sum /= (m_dimension - 1.0);

	g = 1.0 + (9.0 * sum);
	h = 1.0 -  pow((point[0] / g), 2);

	value[1] = g * h;

	m_timesCalled++;
}

bool ZZJ07_F2::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool ZZJ07_F2::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZZJ07_F2::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}


////////////////////////////////////////////////////////////


ZZJ07_F3 :: ZZJ07_F3(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "ZZJ07_F3";
}

ZZJ07_F3::~ZZJ07_F3()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZZJ07_F3::objectives() const
{
	return 2;
}

void ZZJ07_F3::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = 1.0 - exp(-4.0 * point[0]) * pow(sin(6 * M_PI * point[0]), 6);

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += (point[i] - point[0]) * (point[i] - point[0]);

	sum /= 9;

	g = 1.0 + 9.0 * pow(sum, 0.25);
	h = 1.0 -  pow((value[0] / g), 2);

	value[1] = g * h;

	m_timesCalled++;
}

bool ZZJ07_F3::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool ZZJ07_F3::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZZJ07_F3::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}


////////////////////////////////////////////////////////////


ZZJ07_F4 :: ZZJ07_F4(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "ZZJ07_F4";
}

ZZJ07_F4::~ZZJ07_F4()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZZJ07_F4::objectives() const
{
	return 3;
}

void ZZJ07_F4::result(double* const& point, std::vector<double>& value)
{

	value.resize(3);

	double g = 0.0;

	for (unsigned i = 2; i < m_dimension; i++)
		g += (point[i] - point[0]) * (point[i] - point[0]);

	value[0] = cos(0.5 * M_PI * point[0]) * cos(0.5 * M_PI * point[1]) * (1 + g) ;
	value[1] = cos(0.5 * M_PI * point[0]) * sin(0.5 * M_PI * point[1]) * (1 + g) ;
	value[2] = sin(0.5 * M_PI * point[0]) * (1 + g) ;

	m_timesCalled++;
}

bool ZZJ07_F4::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool ZZJ07_F4::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZZJ07_F4::nadirFitness(std::vector<double>& value) const
{
	value.resize(3);
	value[0] = m_dimension - 1.0;
	value[1] = m_dimension - 1.0;
	value[2] = m_dimension - 1.0;
	return true;
}


////////////////////////////////////////////////////////////


ZZJ07_F5 :: ZZJ07_F5(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "ZZJ07_F5";
}

ZZJ07_F5::~ZZJ07_F5()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZZJ07_F5::objectives() const
{
	return 2;
}

void ZZJ07_F5::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = point[0];

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += (point[i] * point[i] - point[0]) * (point[i] * point[i] - point[0]);

	sum /= (m_dimension - 1.0);

	g = 1.0 + (9.0 * sum);
	h = 1.0 - sqrt(point[0] / g);

	value[1] = g * h;

	m_timesCalled++;
}

bool ZZJ07_F5::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool ZZJ07_F5::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZZJ07_F5::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}


////////////////////////////////////////////////////////////


ZZJ07_F6 :: ZZJ07_F6(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "ZZJ07_F6";
}

ZZJ07_F6::~ZZJ07_F6()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZZJ07_F6::objectives() const
{
	return 2;
}

void ZZJ07_F6::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = sqrt(point[0]);

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += (point[i] * point[i] - point[0]) * (point[i] * point[i] - point[0]);

	sum /= (m_dimension - 1.0);

	g = 1.0 + (9.0 * sum);
	h = 1.0 -  pow((value[0] / g), 2);

	value[1] = g * h;

	m_timesCalled++;
}

bool ZZJ07_F6::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool ZZJ07_F6::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZZJ07_F6::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}


////////////////////////////////////////////////////////////


ZZJ07_F7 :: ZZJ07_F7(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "ZZJ07_F7";
}

ZZJ07_F7::~ZZJ07_F7()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZZJ07_F7::objectives() const
{
	return 2;
}

void ZZJ07_F7:: result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = 1.0 - exp(-4.0 * point[0]) * pow(sin(6 * M_PI * point[0]), 6);

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += (point[i] * point[i] - point[0]) * (point[i] * point[i] - point[0]);

	sum /= 9;

	g = 1.0 + 9.0 * pow(sum, 0.25);
	h = 1.0 -  pow((value[0] / g), 2);

	value[1] = g * h;

	m_timesCalled++;
}

bool ZZJ07_F7::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool ZZJ07_F7::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZZJ07_F7::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 10.0;
	return true;
}


////////////////////////////////////////////////////////////


ZZJ07_F8 :: ZZJ07_F8(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "ZZJ07_F8";
}

ZZJ07_F8::~ZZJ07_F8()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZZJ07_F8::objectives() const
{
	return 3;
}

void ZZJ07_F8::result(double* const& point, std::vector<double>& value)
{

	value.resize(3);

	double g = 0.0;

	for (unsigned i = 2; i < m_dimension; i++)
		g += (point[i] * point[i] - point[0]) * (point[i] * point[i] - point[0]);

	value[0] = cos(0.5 * M_PI * point[0]) * cos(0.5 * M_PI * point[1]) * (1 + g) ;
	value[1] = cos(0.5 * M_PI * point[0]) * sin(0.5 * M_PI * point[1]) * (1 + g) ;
	value[2] = sin(0.5 * M_PI * point[0]) * (1 + g) ;

	m_timesCalled++;
}

bool ZZJ07_F8::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool ZZJ07_F8::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZZJ07_F8::nadirFitness(std::vector<double>& value) const
{
	value.resize(3);
	value[0] = m_dimension - 1.0;
	value[1] = m_dimension - 1.0;
	value[2] = m_dimension - 1.0;
	return true;
}


////////////////////////////////////////////////////////////


ZZJ07_F9 :: ZZJ07_F9(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 10.0, 0, 0.0, 1.0))
{
	m_name = "ZZJ07_F9";
}

ZZJ07_F9::~ZZJ07_F9()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZZJ07_F9::objectives() const
{
	return 2;
}

void ZZJ07_F9::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = point[0];

	double g, h, sum = 0.0, prod = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += (point[i] * point[i] - point[0]) * (point[i] * point[i] - point[0]);

	for (unsigned i = 1; i < m_dimension; i++)
		prod *= cos((point[i] * point[i] - point[0]) / sqrt((double)i));

	g = sum / 4000.0 - prod + 2.0;

	h = 1.0 - sqrt(point[0] / g);

	value[1] = g * h;

	m_timesCalled++;
}

bool ZZJ07_F9::ProposeStartingPoint(double*& point) const
{

	point[0] = Rng::uni(0.0, 1.0);
	for (unsigned int i = 1; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 10.0);

	return true;
}

bool ZZJ07_F9::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZZJ07_F9::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = m_dimension / 40.0 + 2.0;
	return true;
}


////////////////////////////////////////////////////////////


ZZJ07_F10 :: ZZJ07_F10(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 10.0, 0, 0.0, 1.0))
{
	m_name = "ZZJ07_F10";
}

ZZJ07_F10::~ZZJ07_F10()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int ZZJ07_F10::objectives() const
{
	return 2;
}

void ZZJ07_F10::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = point[0];

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += (point[i] * point[i] - point[0]) * (point[i] * point[i] - point[0]) - 10 * cos(2.0 * M_PI * (point[i] * point[i] - point[0]));

	g = 1.0 + 10.0 * (m_dimension - 1.0) + sum;
	h = 1.0 - sqrt(point[0] / g);

	value[1] = g * h;

	m_timesCalled++;
}

bool ZZJ07_F10::ProposeStartingPoint(double*& point) const
{

	point[0] = Rng::uni(0.0, 1.0);
	for (unsigned int i = 1; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 10.0);

	return true;
}

bool ZZJ07_F10::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool ZZJ07_F10::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 1.0 + 121.0 * (m_dimension - 1.0);
	return true;
}


////////////////////////////////////////////////////////////


LZ06_F1::LZ06_F1(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "LZ06_F1";
}

LZ06_F1::~LZ06_F1()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int LZ06_F1::objectives() const
{
	return 2;
}

void LZ06_F1::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = point[0];

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += fabs(point[0] - sin(0.5 * point[i] * M_PI));

	g = 1.0 + sum / (m_dimension - 1.0);
	h = 1.0 - sqrt(point[0] / g);

	value[1] = h;

	m_timesCalled++;
}

bool LZ06_F1::ProposeStartingPoint(double*& point) const
{

	for (unsigned i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool LZ06_F1::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool LZ06_F1::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 1.0;
	return true;
}


////////////////////////////////////////////////////////////


LZ06_F2::LZ06_F2(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "LZ06_F2";
}

LZ06_F2::~LZ06_F2()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int LZ06_F2::objectives() const
{
	return 2;
}

void LZ06_F2::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);
	value[0] = point[0];

	double g, h, sum = 0.0;

	for (unsigned i = 1; i < m_dimension; i++)
		sum += fabs(point[0] - sin(0.5 * point[i] * M_PI));

	g = 1.0 + sum / (m_dimension - 1.0);
	h = 1.0 - pow(point[0] / g, 2.0);

	value[1] = h;

	m_timesCalled++;
}

bool LZ06_F2::ProposeStartingPoint(double*& point) const
{

	for (unsigned i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool LZ06_F2::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool LZ06_F2::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 1.0;
	value[1] = 1.0;
	return true;
}


///////////////////////////////////////////////////////////


LZ07_F1::LZ07_F1(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "LZ07_F1";
}

LZ07_F1::~LZ07_F1()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int LZ07_F1::objectives() const
{
	return 2;
}

void LZ07_F1::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	double cardNum_J1, cardNum_J2 ;
	cardNum_J1 = floor((m_dimension - 1.0) / 2.0);
	cardNum_J2 = ceil((m_dimension - 1.0) / 2.0);

	double  sum1 = 0.0, sum2 = 0.0;

	for (unsigned i = 2; i < m_dimension; i += 2)
		sum1 += pow(point[i] - pow(point[0], 0.5 * (1.0 + 3 * (i - 1) / (m_dimension - 2))), 2);

	for (unsigned i = 1; i < m_dimension; i += 2)
		sum2 += pow(point[i] - pow(point[0], 0.5 * (1.0 + 3 * (i - 1) / (m_dimension - 2))), 2);

	value[0] = point[0] + 2 / cardNum_J1 * sum1 ;

	value[1] = 1 - sqrt(point[0]) + 2 / cardNum_J2 * sum2 ;

	m_timesCalled++;
}

bool LZ07_F1::ProposeStartingPoint(double*& point) const
{

	for (unsigned i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool LZ07_F1::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool LZ07_F1::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 3.0;
	value[1] = 3.0;
	return true;
}


///////////////////////////////////////////////////////////


LZ07_F2::LZ07_F2(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, -1.0, 1.0, 0, 0.0, 1.0))
{
	m_name = "LZ07_F2";
}

LZ07_F2::~LZ07_F2()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int LZ07_F2::objectives() const
{
	return 2;
}

void LZ07_F2::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	double cardNum_J1, cardNum_J2 ;
	cardNum_J1 = floor((m_dimension - 1.0) / 2.0);
	cardNum_J2 = ceil((m_dimension - 1.0) / 2.0);

	double  sum1 = 0.0, sum2 = 0.0;

	for (unsigned i = 2; i < m_dimension; i += 2)
		sum1 += pow(point[i] - sin(6 * M_PI * point[0] + (i + 1) * M_PI / m_dimension), 2);

	for (unsigned i = 1; i < m_dimension; i += 2)
		sum2 += pow(point[i] - sin(6 * M_PI * point[0] + (i + 1) * M_PI / m_dimension), 2);

	value[0] = point[0] + 2 / cardNum_J1 * sum1 ;

	value[1] = 1 - sqrt(point[0]) + 2 / cardNum_J2 * sum2 ;

	m_timesCalled++;
}

bool LZ07_F2::ProposeStartingPoint(double*& point) const
{

	point[0] = Rng::uni(0.0, 1.0);
	for (unsigned i = 1; i < m_dimension; i++)
		point[i] = Rng::uni(-1.0, 1.0);

	return true;
}

bool LZ07_F2::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool LZ07_F2::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 9.0;
	value[1] = 9.0;
	return true;
}


///////////////////////////////////////////////////////////


LZ07_F3::LZ07_F3(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, -1.0, 1.0, 0, 0.0, 1.0))
{
	m_name = "LZ07_F3";
}

LZ07_F3::~LZ07_F3()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int LZ07_F3::objectives() const
{
	return 2;
}

void LZ07_F3::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	double cardNum_J1, cardNum_J2 ;
	cardNum_J1 = floor((m_dimension - 1.0) / 2.0);
	cardNum_J2 = ceil((m_dimension - 1.0) / 2.0);

	double  sum1 = 0.0, sum2 = 0.0;

	for (unsigned i = 2; i < m_dimension; i += 2)
		sum1 += pow(point[i] - 0.8 * point[0] * cos(6 * M_PI * point[0] + (i + 1) * M_PI / m_dimension), 2);

	for (unsigned i = 1; i < m_dimension; i += 2)
		sum2 += pow(point[i] - 0.8 * point[0] * sin(6 * M_PI * point[0] + (i + 1) * M_PI / m_dimension), 2);

	value[0] = point[0] + 2 / cardNum_J1 * sum1 ;

	value[1] = 1 - sqrt(point[0]) + 2 / cardNum_J2 * sum2 ;

	m_timesCalled++;
}

bool LZ07_F3::ProposeStartingPoint(double*& point) const
{

	point[0] = Rng::uni(0.0, 1.0);
	for (unsigned i = 1; i < m_dimension; i++)
		point[i] = Rng::uni(-1.0, 1.0);

	return true;
}

bool LZ07_F3::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool LZ07_F3::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 7.48;
	value[1] = 7.48;
	return true;
}


///////////////////////////////////////////////////////////


LZ07_F4::LZ07_F4(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, -1.0, 1.0, 0, 0.0, 1.0))
{
	m_name = "LZ07_F4";
}

LZ07_F4::~LZ07_F4()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int LZ07_F4::objectives() const
{
	return 2;
}

void LZ07_F4::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	double cardNum_J1, cardNum_J2 ;
	cardNum_J1 = floor((m_dimension - 1.0) / 2.0);
	cardNum_J2 = ceil((m_dimension - 1.0) / 2.0);

	double  sum1 = 0.0, sum2 = 0.0;

	for (unsigned i = 2; i < m_dimension; i += 2)
		sum1 += pow(point[i] - 0.8 * point[0] * cos((6 * M_PI * point[0] + (i + 1) * M_PI / m_dimension)) / 3, 2);

	for (unsigned i = 1; i < m_dimension; i += 2)
		sum2 += pow(point[i] - 0.8 * point[0] * sin(6 * M_PI * point[0] + (i + 1) * M_PI / m_dimension), 2);

	value[0] = point[0] + 2 / cardNum_J1 * sum1 ;

	value[1] = 1 - sqrt(point[0]) + 2 / cardNum_J2 * sum2 ;

	m_timesCalled++;
}

bool LZ07_F4::ProposeStartingPoint(double*& point) const
{

	point[0] = Rng::uni(0.0, 1.0);
	for (unsigned i = 1; i < m_dimension; i++)
		point[i] = Rng::uni(-1.0, 1.0);

	return true;
}

bool LZ07_F4::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool LZ07_F4::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 7.48;
	value[1] = 7.48;
	return true;
}


///////////////////////////////////////////////////////////


LZ07_F5::LZ07_F5(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, -1.0, 1.0, 0, 0.0, 1.0))
{
	m_name = "LZ07_F5";
}

LZ07_F5::~LZ07_F5()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int LZ07_F5::objectives() const
{
	return 2;
}

void LZ07_F5::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	double cardNum_J1, cardNum_J2 ;
	cardNum_J1 = floor((m_dimension - 1.0) / 2.0);
	cardNum_J2 = ceil((m_dimension - 1.0) / 2.0);

	double  sum1 = 0.0, sum2 = 0.0;

	for (unsigned i = 2; i < m_dimension; i += 2)
		sum1 += pow(point[i] - 0.3 * point[0] * (point[0] * cos(4 * (6 * M_PI * point[0] + (i + 1) *
					M_PI / m_dimension)) + 2) * sin(6 * M_PI * point[0] + (i + 1) * M_PI / m_dimension), 2);

	for (unsigned i = 1; i < m_dimension; i += 2)
		sum2 += pow(point[i] - 0.3 * point[0] * (point[0] * cos(4 * (6 * M_PI * point[0] + (i + 1) *
					M_PI / m_dimension)) + 2) * cos(6 * M_PI * point[0] + (i + 1) * M_PI / m_dimension), 2);

	value[0] = point[0] + 2 / cardNum_J1 * sum1 ;

	value[1] = 1 - sqrt(point[0]) + 2 / cardNum_J2 * sum2 ;

	m_timesCalled++;
}

bool LZ07_F5::ProposeStartingPoint(double*& point) const
{

	point[0] = Rng::uni(0.0, 1.0);
	for (unsigned i = 1; i < m_dimension; i++)
		point[i] = Rng::uni(-1.0, 1.0);

	return true;
}

bool LZ07_F5::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool LZ07_F5::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 8.22;
	value[1] = 8.22;
	return true;
}


///////////////////////////////////////////////////////////


LZ07_F6::LZ07_F6(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, -2.0, 2.0, 0, 0.0, 1.0, 1, 0.0, 1.0))
{
	m_name = "LZ07_F6";
}

LZ07_F6::~LZ07_F6()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int LZ07_F6::objectives() const
{
	return 3;
}

void LZ07_F6::result(double* const& point, std::vector<double>& value)
{

	value.resize(3);

	double cardNum_J1, cardNum_J2 , cardNum_J3;
	cardNum_J1 = floor((m_dimension - 1.0) / 3.0);
	cardNum_J2 = floor((m_dimension - 2.0) / 3.0);
	cardNum_J3 = floor(m_dimension / 3.0);

	double  sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

	for (unsigned i = 3; i < m_dimension; i += 3)
		sum1 += pow(point[i] - 2.0 * point[1] * sin(2.0 * M_PI * point[0] + (i + 1) * M_PI / m_dimension), 2);

	for (unsigned i = 4; i < m_dimension; i += 3)
		sum2 += pow(point[i] - 2.0 * point[1] * sin(2.0 * M_PI * point[0] + (i + 1) * M_PI / m_dimension), 2);

	for (unsigned i = 2; i < m_dimension; i += 3)
		sum3 += pow(point[i] - 2.0 * point[1] * sin(2.0 * M_PI * point[0] + (i + 1) * M_PI / m_dimension), 2);

	value[0] = cos(0.5 * point[0] * M_PI) * cos(0.5 * point[1] * M_PI) + 2 / cardNum_J1 * sum1;

	value[1] = cos(0.5 * point[0] * M_PI) * sin(0.5 * point[1] * M_PI) + 2 / cardNum_J2 * sum2;

	value[2] = sin(0.5 * point[0] * M_PI) + 2 / cardNum_J3 * sum3;

	m_timesCalled++;
}

bool LZ07_F6::ProposeStartingPoint(double*& point) const
{

	point[0] = Rng::uni(0.0, 1.0);
	point[1] = Rng::uni(0.0, 1.0);
	for (unsigned i = 2; i < m_dimension; i++)
		point[i] = Rng::uni(-2.0, 2.0);

	return true;
}

bool LZ07_F6::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool LZ07_F6::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 33.0;
	value[1] = 33.0;
	return true;
}


////////////////////////////////////////////////////////////


// bool CuboidConstraintHandler_2::isFeasible(const double*& point) const
// {
// 	if (point[0] < m_lower_1 || point[0] > m_upper_1)
// 		return false;
// 
// 	if (point[1] < m_lower_1 || point[1] > m_upper_1)
// 		return false;
// 
// 	for (unsigned i = 2; i < m_dimension; i++)
// 		if (point[i] < m_lower_2 || point[i] > m_upper_2)
// 			return false;
// 
// 	return true;
// }
// 
// bool CuboidConstraintHandler_2::closestFeasible(double*& point) const
// {
// 	if (point[0] < m_lower_1)
// 		point[0] = m_lower_1;
// 	else if (point[0] > m_upper_1)
// 		point[0] = m_upper_1;
// 
// 	if (point[1] < m_lower_1)
// 		point[1] = m_lower_1;
// 	else if (point[1] > m_upper_1)
// 		point[1] = m_upper_1;
// 
// 	for (unsigned i = 2; i < m_dimension; i++)
// 	{
// 		if (point[i] < m_lower_2)
// 			point[i] = m_lower_2;
// 		else if (point[i] > m_upper_2)
// 			point[i] = m_upper_2;
// 	}
// 
// 	return true;
// }


///////////////////////////////////////////////////////////


LZ07_F7::LZ07_F7(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "LZ07_F7";
}

LZ07_F7::~LZ07_F7()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int LZ07_F7::objectives() const
{
	return 2;
}

void LZ07_F7::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	double cardNum_J1, cardNum_J2 ;
	cardNum_J1 = floor((m_dimension - 1.0) / 2.0);
	cardNum_J2 = ceil((m_dimension - 1.0) / 2.0);

	double  sum1 = 0.0, sum2 = 0.0;

	double y_i;
	for (unsigned i = 2; i < m_dimension; i += 2)
	{
		y_i = point[i] - pow(point[0], 0.5 * (1.0 + 3 * (i - 1) / (m_dimension - 2)));
		sum1 += 4 * pow(y_i, 2) - cos(8 * y_i * M_PI) + 1.0;
	}
	for (unsigned i = 1; i < m_dimension; i += 2)
	{
		y_i = point[i] - pow(point[0], 0.5 * (1.0 + 3 * (i - 1) / (m_dimension - 2)));
		sum2 += 4 * pow(y_i, 2) - cos(8 * y_i * M_PI) + 1.0;
	}
	value[0] = point[0] + 2 / cardNum_J1 * sum1 ;

	value[1] = 1 - sqrt(point[0]) + 2 / cardNum_J2 * sum2 ;

	m_timesCalled++;
}

bool LZ07_F7::ProposeStartingPoint(double*& point) const
{

	for (unsigned i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool LZ07_F7::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool LZ07_F7::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 13.0;
	value[1] = 13.0;
	return true;
}


///////////////////////////////////////////////////////////


LZ07_F8::LZ07_F8(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "LZ07_F8";
}

LZ07_F8::~LZ07_F8()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int LZ07_F8::objectives() const
{
	return 2;
}

void LZ07_F8::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	double cardNum_J1, cardNum_J2 ;
	cardNum_J1 = floor((m_dimension - 1.0) / 2.0);
	cardNum_J2 = ceil((m_dimension - 1.0) / 2.0);

	double  sum1 = 0.0, sum2 = 0.0, prod1 = 0.0, prod2 = 0.0;

	double y_i;
	for (unsigned i = 2; i < m_dimension; i += 2)
	{
		y_i = point[i] - pow(point[0], 0.5 * (1.0 + 3 * (i - 1) / (m_dimension - 2)));
		sum1 += pow(y_i, 2);
		prod1 *= cos(20 * y_i * M_PI / sqrt(i + 1.0));
	}
	for (unsigned i = 1; i < m_dimension; i += 2)
	{
		y_i = point[i] - pow(point[0], 0.5 * (1.0 + 3 * (i - 1) / (m_dimension - 2)));
		sum2 += pow(y_i, 2);
		prod2 += cos(20 * y_i * M_PI / sqrt(i + 1.0));
	}
	value[0] = point[0] + 4 / cardNum_J1 * (2 * sum1 + prod1 + 1);

	value[1] = 1 - sqrt(point[0]) + 4 / cardNum_J2 * (2 * sum2 + prod2 + 1) ;

	m_timesCalled++;
}

bool LZ07_F8::ProposeStartingPoint(double*& point) const
{

	for (unsigned i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool LZ07_F8::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool LZ07_F8::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 4.0 * m_dimension + 9.0;
	value[1] = 4.0 * m_dimension + 9.0;
	return true;
}


///////////////////////////////////////////////////////////


LZ07_F9::LZ07_F9(unsigned d) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, -1.0, 1.0, 0, 0.0, 1.0))
{
	m_name = "LZ07_F9";
}

LZ07_F9::~LZ07_F9()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int LZ07_F9::objectives() const
{
	return 2;
}

void LZ07_F9::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	double cardNum_J1, cardNum_J2 ;
	cardNum_J1 = floor((m_dimension - 1.0) / 2.0);
	cardNum_J2 = ceil((m_dimension - 1.0) / 2.0);

	double  sum1 = 0.0, sum2 = 0.0;

	for (unsigned i = 2; i < m_dimension; i += 2)
		sum1 += pow(point[i] - sin(6 * M_PI * point[0] + (i + 1) * M_PI / m_dimension), 2);

	for (unsigned i = 1; i < m_dimension; i += 2)
		sum2 += pow(point[i] - sin(6 * M_PI * point[0] + (i + 1) * M_PI / m_dimension), 2);

	value[0] = point[0] + 2 / cardNum_J1 * sum1 ;

	value[1] = 1 - pow(point[0], 2) + 2 / cardNum_J2 * sum2 ;

	m_timesCalled++;
}

bool LZ07_F9::ProposeStartingPoint(double*& point) const
{

	point[0] = Rng::uni(0.0, 1.0);
	for (unsigned i = 1; i < m_dimension; i++)
		point[i] = Rng::uni(-1.0, 1.0);

	return true;
}

bool LZ07_F9::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool LZ07_F9::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 9.0;
	value[1] = 9.0;
	return true;
}


///////////////////////////////////////////////////////////


ELLIBase::ELLIBase(unsigned d, double a) : ObjectiveFunctionVS<double>(d, NULL)
{
	m_name = "ELLIBase";
	m_a = a;
}

ELLIBase::~ELLIBase()
{}


unsigned int ELLIBase::objectives() const
{
	return 2;
}

void ELLIBase::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	double sum1 = 0.0, sum2 = 0.0;

	for (unsigned i = 0; i < m_dimension; i++)
	{
		sum1 += pow(m_a, 2.0 * (i / (m_dimension - 1.0))) * point[i] * point[i];
	}

	value[0] = sum1 / (m_a * m_a * m_dimension);


	for (unsigned i = 0; i < m_dimension; i++)
	{
		sum2 += pow(m_a, 2 * (i / (m_dimension - 1.0))) * (point[i] - 2.0) * (point[i] - 2.0);
	}

	value[1] =  sum2 / (m_a * m_a * m_dimension);

	m_timesCalled++;
}

bool ELLIBase::ProposeStartingPoint(double*& point) const
{
	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::gauss();
	return true;
}

bool ELLIBase::utopianFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 0.0;
	value[1] = 0.0;
	return true;
}


////////////////////////////////////////////////////////////


ELLI1:: ELLI1(unsigned d, double a): TransformedObjectiveFunction(base, d)
		, base(d, a)
{
	m_name = "ELLI1";
}

ELLI1::~ELLI1()
{}


unsigned ELLI1::objectives() const
{
	return base.objectives();
}

bool ELLI1::ProposeStartingPoint(double*& point) const
{
	return base.ProposeStartingPoint(point);
}


////////////////////////////////////////////////////////////


ELLI2::ELLI2(unsigned d, double a): ObjectiveFunctionVS<double>(d , NULL), m_Transformation_1(0, 0), m_Transformation_2(0, 0)
{
	m_name = "ELLI2";
	m_a = a;
	initRandomRotation();
}

ELLI2::~ELLI2()
{}


unsigned int ELLI2::objectives() const
{
	return 2;
}

void ELLI2::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	double sum_1 = 0, sum_2 = 0;

	std::vector<double> y(m_dimension);
	std::vector<double> z(m_dimension);

	transform_1(point, y);
	transform_2(point, z);

	for (unsigned i = 0; i < m_dimension; i++)
	{
		sum_1 += pow(m_a, 2.0 * (i / (m_dimension - 1.0))) * y[i] * y[i];
		sum_2 += pow(m_a, 2.0 * (i / (m_dimension - 1.0))) * (z[i] - 2.0) * (z[i] - 2.0);
	}

	value[0] = sum_1 / (m_a * m_a * m_dimension);

	value[1] = sum_2 / (m_a * m_a * m_dimension);


	m_timesCalled++;
}

bool ELLI2::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::gauss();

	return true;
}

void ELLI2::initRandomRotation()
{

	unsigned i, j, c;
	Matrix H_1(m_dimension, m_dimension);
	Matrix H_2(m_dimension, m_dimension);
	m_Transformation_1.resize(m_dimension, m_dimension);
	m_Transformation_2.resize(m_dimension, m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		for (c = 0; c < m_dimension; c++)
		{
			H_1(i, c) = Rng::gauss(0, 1);
			H_2(i, c) = Rng::gauss(0, 1);
		}
	}
	m_Transformation_1 = H_1;
	m_Transformation_2 = H_2;
	for (i = 0; i < m_dimension; i++)
	{
		for (j = 0; j < i; j++)
		{
			for (c = 0; c < m_dimension; c++)
			{
				m_Transformation_1(i, c) -= (H_1[i] * H_1[j]) * H_1(j, c) / (H_1[j].norm2());
				m_Transformation_2(i, c) -= (H_2[i] * H_2[j]) * H_2(j, c) / (H_2[j].norm2());
			}
		}
		H_1 = m_Transformation_1;
		H_2 = m_Transformation_2;
	}
	for (i = 0; i < m_dimension; i++)
	{
		double normB_1 = m_Transformation_1[i].norm();
		double normB_2 = m_Transformation_2[i].norm();
		for (j = 0; j < m_dimension; j++)
		{
			m_Transformation_1(i, j) = m_Transformation_1(i, j) / normB_1;
			m_Transformation_2(i, j) = m_Transformation_2(i, j) / normB_2;
		}
	}
}

void ELLI2::transform_1(const double* in, std::vector<double>& out) const
{
	unsigned i, j;
	out.resize(m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		out[i] = 0.0;
		for (j = 0; j < m_dimension; j++)
			out[i] += m_Transformation_1(j, i) * in[j];
	}
}

void ELLI2::transform_2(const double* in, std::vector<double>& out) const
{
	unsigned i, j;
	out.resize(m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		out[i] = 0.0;
		for (j = 0; j < m_dimension; j++)
			out[i] += m_Transformation_2(j, i) * in[j];
	}
}


///////////////////////////////////////////////////////////


CIGTABBase::CIGTABBase(unsigned d, double a) : ObjectiveFunctionVS<double>(d, NULL)
{
	m_name = "CIGTABBase";
	m_a = a;
}

CIGTABBase::~CIGTABBase()
{}


unsigned int CIGTABBase::objectives() const
{
	return 2;
}

void CIGTABBase::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	double result = point[0] * point[0] + m_a * m_a * point[m_dimension-1] * point[m_dimension-1];

	for (unsigned i = 1; i < m_dimension - 1; i++)
	{
		result += m_a * point[i] * point[i];
	}

	value[0] = result / (m_a * m_a * m_dimension);

	result = (point[0] - 2) * (point[0] - 2) + m_a * m_a * (point[m_dimension-1] - 2) * (point[m_dimension-1] - 2);

	for (unsigned i = 1; i < m_dimension - 1; i++)
	{
		result += m_a * (point[i] - 2) * (point[i] - 2);
	}

	value[1] = result / (m_a * m_a * m_dimension);
	m_timesCalled++;
}

bool CIGTABBase::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::gauss();
	return true;
}

bool CIGTABBase::utopianFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 0.0;
	value[1] = 0.0;
	return true;
}


////////////////////////////////////////////////////////////


CIGTAB1:: CIGTAB1(unsigned d, double a): TransformedObjectiveFunction(base, d)
		, base(d, a)
{
	m_name = "CIGTAB1";
}

CIGTAB1::~CIGTAB1()
{}


unsigned CIGTAB1::objectives() const
{
	return base.objectives();
}

bool CIGTAB1::ProposeStartingPoint(double*& point) const
{
	return base.ProposeStartingPoint(point);
}


////////////////////////////////////////////////////////////


CIGTAB2::CIGTAB2(unsigned d, double a): ObjectiveFunctionVS<double>(d , NULL), m_Transformation_1(0, 0), m_Transformation_2(0, 0)
{
	m_name = "CIGTAB2";
	m_a = a;
	initRandomRotation();
}

CIGTAB2::~CIGTAB2()
{}


unsigned int CIGTAB2::objectives() const
{
	return 2;
}

void CIGTAB2::result(double* const& point, std::vector<double>& value)
{

	value.resize(2);

	std::vector<double> y(m_dimension);
	std::vector<double> z(m_dimension);

	transform_1(point, y);
	transform_2(point, z);

	double result_1 = y[0] * y[0] + m_a * m_a * y[m_dimension-1] * y[m_dimension-1];
	double result_2 = z[0] * z[0] + m_a * m_a * z[m_dimension-1] * z[m_dimension-1];


	for (unsigned i = 1; i < m_dimension - 1; i++)
	{
		result_1 += m_a * y[i] * y[i];
		result_2 += m_a * (z[i] - 2) * (z[i] - 2);
	}

	value[0] = result_1 / (m_a * m_a * m_dimension);


	value[1] = result_2 / (m_a * m_a * m_dimension);


	m_timesCalled++;
}

bool CIGTAB2::ProposeStartingPoint(double*& point) const
{

	for (unsigned int i = 0; i < m_dimension; i++)
		point[i] = Rng::gauss();

	return true;
}

void CIGTAB2::initRandomRotation()
{

	unsigned i, j, c;
	Matrix H_1(m_dimension, m_dimension);
	Matrix H_2(m_dimension, m_dimension);
	m_Transformation_1.resize(m_dimension, m_dimension);
	m_Transformation_2.resize(m_dimension, m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		for (c = 0; c < m_dimension; c++)
		{
			H_1(i, c) = Rng::gauss(0, 1);
			H_2(i, c) = Rng::gauss(0, 1);
		}
	}
	m_Transformation_1 = H_1;
	m_Transformation_2 = H_2;
	for (i = 0; i < m_dimension; i++)
	{
		for (j = 0; j < i; j++)
		{
			for (c = 0; c < m_dimension; c++)
			{
				m_Transformation_1(i, c) -= (H_1[i] * H_1[j]) * H_1(j, c) / (H_1[j].norm2());
				m_Transformation_2(i, c) -= (H_2[i] * H_2[j]) * H_2(j, c) / (H_2[j].norm2());
			}
		}
		H_1 = m_Transformation_1;
		H_2 = m_Transformation_2;
	}
	for (i = 0; i < m_dimension; i++)
	{
		double normB_1 = m_Transformation_1[i].norm();
		double normB_2 = m_Transformation_2[i].norm();
		for (j = 0; j < m_dimension; j++)
		{
			m_Transformation_1(i, j) = m_Transformation_1(i, j) / normB_1;
			m_Transformation_2(i, j) = m_Transformation_2(i, j) / normB_2;
		}
	}
}

void CIGTAB2::transform_1(const double* in, std::vector<double>& out) const
{
	unsigned i, j;
	out.resize(m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		out[i] = 0.0;
		for (j = 0; j < m_dimension; j++)
			out[i] += m_Transformation_1(j, i) * in[j];
	}
}

void CIGTAB2::transform_2(const double* in, std::vector<double>& out) const
{
	unsigned i, j;
	out.resize(m_dimension);
	for (i = 0; i < m_dimension; i++)
	{
		out[i] = 0.0;
		for (j = 0; j < m_dimension; j++)
			out[i] += m_Transformation_2(j, i) * in[j];
	}
}


///////////////////////////////////////////////////////////


DTLZ1::DTLZ1(unsigned d, unsigned m) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "DTLZ1";
	m_objectives = m;
}

DTLZ1::~DTLZ1()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int DTLZ1::objectives() const
{
	return m_objectives;
}

void DTLZ1::result(double* const& point, std::vector<double>& value)
{

	if (m_objectives > m_dimension)
		throw SHARKEXCEPTION("[DTLZ1::result] number of objectives exceeds the search space dimension.");

	value.resize(m_objectives);

	int k = m_dimension - m_objectives + 1 ;
	double g = 0.0;

	for (unsigned int i = m_dimension - k + 1; i <= m_dimension; i++)
		g += pow(point[i-1] - 0.5, 2) - cos(20 * M_PI * (point[i-1] - 0.5));

	g = 100 * (k + g);

	for (unsigned int i = 1; i <= m_objectives; i++)
	{
		double f = 0.5 * (1 + g);
		for (unsigned int j = m_objectives - i; j >= 1; j--)
			f *= point[j-1];

		if (i > 1)
			f *= 1 - point[(m_objectives - i + 1) - 1];

		value[i-1] = f;
	}

	m_timesCalled++;
}

bool DTLZ1::ProposeStartingPoint(double*& point) const
{

	for (unsigned i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool DTLZ1::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool DTLZ1::nadirFitness(std::vector<double>& value) const
{
	value.resize(m_objectives);
	int k = m_dimension - m_objectives + 1;
	unsigned int i;
	for (i=0; i<m_objectives; i++) value[i] = 112.5 * k + 0.5;
	return true;
}


////////////////////////////////////////////////////////////


DTLZ2::DTLZ2(unsigned d, unsigned m) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "DTLZ2";
	m_objectives = m;
}

DTLZ2::~DTLZ2()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int DTLZ2::objectives() const
{
	return m_objectives;
}

void DTLZ2::result(double* const& point, std::vector<double>& value)
{

	if (m_objectives > m_dimension)
		throw SHARKEXCEPTION("[DTLZ2::result] number of objectives exceeds the search space dimension.");

	value.resize(m_objectives);

	int    k ;
	double g ;

	k = m_dimension - m_objectives + 1 ;
	g = 0.0 ;

	for (unsigned int i = m_dimension - k + 1; i <= m_dimension; i++)
		g += pow(point[i-1] - 0.5, 2);

	for (unsigned int i = 1; i <= m_objectives; i++)
	{
		double f = (1 + g);
		for (int j = m_objectives - i; j >= 1; j--)
			f *= cos(point[j-1] * M_PI / 2);

		if (i > 1)
			f *= sin(point[(m_objectives - i + 1) - 1] * M_PI / 2);

		value[i-1] = f ;
	}

	m_timesCalled++;
}

bool DTLZ2::ProposeStartingPoint(double*& point) const
{
	for (unsigned i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool DTLZ2::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool DTLZ2::nadirFitness(std::vector<double>& value) const
{
	value.resize(m_objectives);
	int k = m_dimension - m_objectives + 1;
	unsigned int i;
	for (i=0; i<m_objectives; i++) value[i] = 0.25 * k + 1.0;
	return true;
}


////////////////////////////////////////////////////////////


DTLZ3::DTLZ3(unsigned d, unsigned m) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "DTLZ3";
	m_objectives = m;
}

DTLZ3::~DTLZ3()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int DTLZ3::objectives() const
{
	return m_objectives;
}

void DTLZ3::result(double* const& point, std::vector<double>& value)
{

	if (m_objectives > m_dimension)
		throw SHARKEXCEPTION("[DTLZ3::result] number of objectives exceeds the search space dimension.");

	value.resize(m_objectives);

	int    k ;
	double g ;

	k = m_dimension - m_objectives + 1 ;
	g = 0.0 ;

	for (unsigned int i = m_dimension - k + 1; i <= m_dimension; i++)
		g += pow(point[i-1] - 0.5, 2) - cos(20 * M_PI * (point[i-1] - 0.5));

	g = 100 * (k + g);

	for (unsigned int i = 1; i <= m_objectives; i++)
	{
		double f = (1 + g);
		for (unsigned int j = m_objectives - i; j >= 1; j--)
			f *= cos(point[j-1] * M_PI / 2);

		if (i > 1)
			f *= sin(point[(m_objectives - i + 1) - 1] * M_PI / 2);

		value[i-1] = f ;
	}

	m_timesCalled++;
}

bool DTLZ3::ProposeStartingPoint(double*& point) const
{
	for (unsigned i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool DTLZ3::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool DTLZ3::nadirFitness(std::vector<double>& value) const
{
	value.resize(m_objectives);
	int k = m_dimension - m_objectives + 1;
	unsigned int i;
	for (i=0; i<m_objectives; i++) value[i] = 225.0 * k + 1.0;
	return true;
}


////////////////////////////////////////////////////////////


DTLZ4::DTLZ4(unsigned d, unsigned m) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "DTLZ4";
	m_objectives = m;
}

DTLZ4::~DTLZ4()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}

unsigned int DTLZ4::objectives() const
{
	return m_objectives;
}

void DTLZ4::result(double* const& point, std::vector<double>& value)
{

	if (m_objectives > m_dimension)
		throw SHARKEXCEPTION("[DTLZ4::result] number of objectives exceeds the search space dimension.");

	value.resize(m_objectives);

	int    k ;
	double g ;
	double alpha;

	k = m_dimension - m_dimension + 1 ;
	alpha = 100;
	g = 0.0 ;

	for (unsigned int i = m_dimension - k + 1; i <= m_dimension; i++)
		g += pow(point[i-1] - 0.5, 2);

	for (unsigned int i = 1; i <= m_objectives; i++)
	{
		double f = (1 + g);
		for (unsigned int j = m_objectives - i; j >= 1; j--)
			f *= cos(pow(point[j-1], alpha) * M_PI / 2);

		if (i > 1)
			f *= sin(pow(point[(m_objectives - i + 1) - 1], alpha) * M_PI / 2);

		value[i-1] = f ;
	}


	m_timesCalled++;
}

bool DTLZ4::ProposeStartingPoint(double*& point) const
{
	for (unsigned i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool DTLZ4::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool DTLZ4::nadirFitness(std::vector<double>& value) const
{
	value.resize(m_objectives);
	int k = m_dimension - m_objectives + 1;
	unsigned int i;
	for (i=0; i<m_objectives; i++) value[i] = 0.25 * k + 1.0;
	return true;
}


////////////////////////////////////////////////////////////


DTLZ5::DTLZ5(unsigned d, unsigned m) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "DTLZ5";
	m_objectives = m;
}

DTLZ5::~DTLZ5()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}

unsigned int DTLZ5::objectives() const
{
	return m_objectives;
}

void DTLZ5::result(double* const& point, std::vector<double>& value)
{

	if (m_objectives > m_dimension)
		throw SHARKEXCEPTION("[DTLZ5::result] number of objectives exceeds the search space dimension.");

	value.resize(m_objectives);

	int    k ;
	double g ;

	std::vector<double> phi(m_objectives);

	k = m_dimension - m_objectives + 1 ;
	g = 0.0 ;

	for (unsigned int i = m_dimension - k + 1; i <= m_dimension; i++)
		g += pow(point[i-1] - 0.5, 2);

	double t = M_PI  / (4 * (1 + g));

	phi[0] = point[0] * M_PI / 2;
	for (unsigned int i = 2; i <= (m_objectives - 1); i++)
		phi[i-1] = t * (1 + 2 * g * point[i-1]);

	for (unsigned int i = 1; i <= m_objectives; i++)
	{
		double f = (1 + g);

		for (unsigned int j = m_objectives - i; j >= 1; j--)
			f *= cos(phi[j-1]);

		if (i > 1)
			f *= sin(phi[(m_objectives - i + 1) - 1]);

		value[i-1] = f ;
	}

	m_timesCalled++;
}

bool DTLZ5::ProposeStartingPoint(double*& point) const
{

	for (unsigned i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool DTLZ5::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool DTLZ5::nadirFitness(std::vector<double>& value) const
{
	value.resize(m_objectives);
	int k = m_dimension - m_objectives + 1;
	unsigned int i;
	for (i=0; i<m_objectives; i++) value[i] = 0.25 * k + 1.0;
	return true;
}


////////////////////////////////////////////////////////////


DTLZ6::DTLZ6(unsigned d, unsigned m) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "DTLZ6";
	m_objectives = m;
}

DTLZ6::~DTLZ6()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}

unsigned int DTLZ6::objectives() const
{
	return m_objectives;
}

void DTLZ6::result(double* const& point, std::vector<double>& value)
{

	if (m_objectives > m_dimension)
		throw SHARKEXCEPTION("[DTLZ6::result] number of objectives exceeds the search space dimension.");

	value.resize(m_objectives);

	int    k ;
	double g ;

	std::vector<double> phi(m_objectives);

	k = m_dimension - m_objectives + 1 ;
	g = 0.0 ;

	for (unsigned int i = m_dimension - k + 1; i <= m_dimension; i++)
		g += pow(point[i-1], 0.1);

	double t = M_PI  / (4 * (1 + g));

	phi[0] = point[0] * M_PI / 2;
	for (unsigned int i = 2; i <= m_objectives - 1; i++)
		phi[i-1] = t * (1 + 2 * g * point[i-1]);

	for (unsigned int i = 1; i <= m_objectives; i++)
	{
		double f = (1 + g);

		for (int j = m_objectives - i; j >= 1; j--)
			f *= cos(phi[j-1]);

		if (i > 1)
			f *= sin(phi[(m_objectives - i + 1) - 1]);

		value[i-1] = f ;
	}
	m_timesCalled++;
}

bool DTLZ6::ProposeStartingPoint(double*& point) const
{

	for (unsigned i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool DTLZ6::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool DTLZ6::nadirFitness(std::vector<double>& value) const
{
	value.resize(m_objectives);
	int k = m_dimension - m_objectives + 1;
	unsigned int i;
	for (i=0; i<m_objectives; i++) value[i] = k + 1.0;
	return true;
}


////////////////////////////////////////////////////////////


DTLZ7::DTLZ7(unsigned d, unsigned m) : ObjectiveFunctionVS<double>(d, new BoxConstraintHandler(d, 0.0, 1.0))
{
	m_name = "DTLZ7";
	m_objectives = m;
}

DTLZ7::~DTLZ7()
{
	if (constrainthandler != NULL)
		delete constrainthandler;
}


unsigned int DTLZ7::objectives() const
{
	return m_objectives;
}

void DTLZ7::result(double* const& point, std::vector<double>& value)
{

	if (m_objectives > m_dimension)
		throw SHARKEXCEPTION("[DTLZ7::result] number of objectives exceeds the search space dimension.");

	value.resize(m_objectives);

	int    k;
	double g;
	double h;

	k = m_dimension - m_objectives + 1 ;
	g = 0.0 ;
	for (unsigned int i = m_dimension - k + 1; i <= m_dimension; i++)
		g += point[i-1];

	g = 1 + 9 * g / k;

	for (unsigned int i = 1; i <= m_objectives - 1; i++)
		value[i-1] = point[i-1];

	h = 0.0 ;
	for (unsigned int j = 1; j <= m_objectives - 1; j++)
		h += point[j-1] / (1 + g) * (1 + sin(3 * M_PI * point[j-1]));

	h = m_objectives - h ;

	value[m_objectives-1] = (1 + g) * h;
	m_timesCalled++;
}

bool DTLZ7::ProposeStartingPoint(double*& point) const
{
	for (unsigned i = 0; i < m_dimension; i++)
		point[i] = Rng::uni(0.0, 1.0);

	return true;
}

bool DTLZ7::utopianFitness(std::vector<double>& value) const
{
	return false;		// TODO
}

bool DTLZ7::nadirFitness(std::vector<double>& value) const
{
	value.resize(m_objectives);
	unsigned int i;
	for (i=0; i<m_objectives; i++) value[i] = 1.0;
	value[m_objectives - 1] = 11.0 * m_objectives;
	return true;
}


////////////////////////////////////////////////////////////


Superspheres::Superspheres(unsigned int dim)
: ObjectiveFunctionVS<double>(dim, new BoxConstraintHandler(dim, 1.0, 5.0, 0, 0.0, 0.5 * M_PI))
{
	m_name = "Superspheres";
}

Superspheres::~Superspheres()
{
}


unsigned int Superspheres::objectives() const
{
	return 2;
}

void Superspheres::result(double* const& point, std::vector<double>& value)
{
	unsigned int i;
	double d = 0.0;
	if (m_dimension > 1)
	{
		for (i=1; i<m_dimension; i++) d += Shark::sqr(point[i]);
		d /= (m_dimension - 1.0);
	}
	double r = sin(M_PI * d);
	r *= r;
	r += 1.0;
	value.resize(2);
	value[0] = (r * cos(point[0]));
	value[1] = (r * sin(point[0]));
	m_timesCalled++;
}

bool Superspheres::ProposeStartingPoint(double*& point) const
{
	unsigned int i;
	point[0] = Rng::uni(0.0, 0.5 * M_PI);
	for (i=1; i<m_dimension; i++) point[i] = Rng::uni(1.0, 5.0);
	return true;
}

bool Superspheres::utopianFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 0.0;
	value[1] = 0.0;
	return true;
}

bool Superspheres::nadirFitness(std::vector<double>& value) const
{
	value.resize(2);
	value[0] = 2.0;
	value[1] = 2.0;
	return true;
}
