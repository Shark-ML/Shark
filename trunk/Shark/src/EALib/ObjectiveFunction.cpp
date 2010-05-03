//===========================================================================
/*!
 *  \file ObjectiveFunction.cpp
 *
 *  \brief general objective function class
 *
 *  \author  Christian Igel, ...
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
 *  \par Project:
 *      EALib
 *
 *  This file is part of EALib. This library is free software;
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


#include <EALib/ObjectiveFunction.h>


////////////////////////////////////////////////////////////


BoxConstraintHandler::BoxConstraintHandler(unsigned int dim, double lower, double upper)
{
	m_dimension = dim;
	m_lower.resize(m_dimension);
	m_upper.resize(m_dimension);
	unsigned int i;
	for (i=0; i<m_dimension; i++)
	{
		m_lower[i] = lower;
		m_upper[i] = upper;
	}
}

BoxConstraintHandler::BoxConstraintHandler(unsigned int dim, double lower, double upper, unsigned int exception, double exceptionLower, double exceptionUpper)
{
	m_dimension = dim;
	m_lower.resize(m_dimension);
	m_upper.resize(m_dimension);
	unsigned int i;
	for (i=0; i<m_dimension; i++)
	{
		m_lower[i] = lower;
		m_upper[i] = upper;
	}
	m_lower[exception] = exceptionLower;
	m_upper[exception] = exceptionUpper;
}

BoxConstraintHandler::BoxConstraintHandler(unsigned int dim, double lower, double upper, unsigned int exception1, double exception1Lower, double exception1Upper, unsigned int exception2, double exception2Lower, double exception2Upper)
{
	m_dimension = dim;
	m_lower.resize(m_dimension);
	m_upper.resize(m_dimension);
	unsigned int i;
	for (i=0; i<m_dimension; i++)
	{
		m_lower[i] = lower;
		m_upper[i] = upper;
	}
	m_lower[exception1] = exception1Lower;
	m_upper[exception1] = exception1Upper;
	m_lower[exception2] = exception2Lower;
	m_upper[exception2] = exception2Upper;
}

BoxConstraintHandler::BoxConstraintHandler(const std::vector<double>& lower, const std::vector<double>& upper)
{
	SIZE_CHECK(lower.size() == upper.size());
	m_dimension = lower.size();
	m_lower = lower;
	m_upper = upper;
}

BoxConstraintHandler::~BoxConstraintHandler()
{
}


bool BoxConstraintHandler::isFeasible(double* const& point) const
{
	unsigned int i;
	for (i=0; i<m_dimension; i++) if (point[i] < m_lower[i] || point[i] > m_upper[i]) return false;
	return true;
}

bool BoxConstraintHandler::closestFeasible(double*& point) const
{
	unsigned int i;
	for (i=0; i<m_dimension; i++)
	{
		if (point[i] < m_lower[i]) point[i] = m_lower[i];
		else if (point[i] > m_upper[i]) point[i] = m_upper[i];
	}
	return true;
}


////////////////////////////////////////////////////////////


ObjectiveFunction::ObjectiveFunction()
: m_timesCalled(0)
{
}

ObjectiveFunction::~ObjectiveFunction()
{
}


////////////////////////////////////////////////////////////


TransformedObjectiveFunction::TransformedObjectiveFunction(ObjectiveFunctionVS<double>& base, unsigned d)
: baseObjective(base)
, m_Transformation(0, 0)
{
	initRandomRotation(d);
}

TransformedObjectiveFunction::TransformedObjectiveFunction(ObjectiveFunctionVS<double>& base, const Array<double>& transformation)
: baseObjective(base)
, m_Transformation(transformation)
{
	m_dimension = baseObjective.dimension();
	init(transformation);
}

TransformedObjectiveFunction::~TransformedObjectiveFunction()
{
}


void TransformedObjectiveFunction::init(const Array<double>& transformation) {
	SIZE_CHECK(transformation.ndim() == 2);
	m_dimension = transformation.dim(0);
	SIZE_CHECK(transformation.dim(1) == m_dimension);

	baseObjective.init(m_dimension);
	m_Transformation = transformation;
}

void TransformedObjectiveFunction::initRandomRotation(unsigned d) {
	if (d == 0) d = baseObjective.dimension();
	else baseObjective.init(d);
	m_dimension = d;

	unsigned i, j, c;
	Matrix H(m_dimension, m_dimension);
	m_Transformation.resize(m_dimension, m_dimension);
	for(i = 0; i < m_dimension; i++) {
		for(c = 0; c < m_dimension; c++) {
			H(i, c) = Rng::gauss(0, 1);
		}
	}
	m_Transformation = H;
	for(i = 0; i < m_dimension; i++) {
		for(j = 0; j < i; j++)
			for(c = 0; c < m_dimension; c++)
// 				m_Transformation(i, c) -= scalarProduct(H[i], H[j]) * H(j, c) / scalarProduct(H[j], H[j]);
				m_Transformation(i, c) -= (H[i] * H[j]) * H(j, c) / (H[j].norm2());
		H = m_Transformation;
	}
	for(i = 0; i < m_dimension; i++) {
		double normB = m_Transformation[i].norm();
		for(j = 0; j < m_dimension; j++)
			m_Transformation(i, j) = m_Transformation(i, j) / normB;
	}
}

void TransformedObjectiveFunction::result(double* const& point, std::vector<double>& value) {
	std::vector<double> tmp(m_dimension);
	transform(point, tmp);
	((ObjectiveFunctionT<const double*>&)baseObjective).result((const double*)&tmp[0], value);
	m_timesCalled++;
}

bool TransformedObjectiveFunction::ProposeStartingPoint(double*& point) const
{
	std::vector<double> tmp(m_dimension);
	double* p = &tmp[0];
	if (((ObjectiveFunctionT<double*>&)baseObjective).ProposeStartingPoint(p))
	{
		transformInverse(tmp, point);
		return true;
	}
	else return false;
}

bool TransformedObjectiveFunction::utopianFitness(std::vector<double>& value) const
{
	return baseObjective.utopianFitness(value);
}

bool TransformedObjectiveFunction::nadirFitness(std::vector<double>& value) const
{
	return baseObjective.nadirFitness(value);
}

unsigned int TransformedObjectiveFunction::objectives() const
{
	return baseObjective.objectives();
}

void TransformedObjectiveFunction::transform(const double* in, std::vector<double>& out) const {
	unsigned i, j;
	out.resize(m_dimension);
	for (i = 0; i < m_dimension; i++) {
		out[i] = 0.0;
		for(j = 0; j < m_dimension; j++)
			out[i] += m_Transformation(j, i) * in[j];
	}
}

void TransformedObjectiveFunction::transformInverse(const std::vector<double>& in, double* out) const
{
	Matrix inv = m_Transformation.inverse();
	unsigned i, j;
	for (i = 0; i < m_dimension; i++) {
		out[i] = 0.0;
		for(j = 0; j < m_dimension; j++)
			out[i] += inv(j, i) * in[j];
	}
}
