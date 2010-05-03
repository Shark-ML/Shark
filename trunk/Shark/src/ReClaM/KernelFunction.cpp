//===========================================================================
/*!
*  \file KernelFunction.cpp
*
*  \brief Kernel function base class and simple function implementations
*
*  \author  T. Glasmachers
*  \date    2006
*
*  \par Copyright (c) 1999-2006:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
*  \par Project:
*      ReClaM
*
*
*  <BR>
*
*
*  <BR><HR>
*  This file is part of ReClaM. This library is free software;
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
#include <ReClaM/KernelFunction.h>
#include <math.h>


////////////////////////////////////////////////////////////////////////////////


KernelFunction::KernelFunction()
{
}

KernelFunction::~KernelFunction()
{
}


double KernelFunction::evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const
{
	SIZE_CHECK(x1.ndim() == 1);
	SIZE_CHECK(x2.ndim() == 1);
	SIZE_CHECK(x1.nelem() == x2.nelem());
	KernelFunction* pT = const_cast<KernelFunction*>(this);

	int i, ic = getParameterDimension();
	double temp;
	double ret = eval(x1, x2);
	derivative.resize(ic, false);
	for (i = 0; i < ic; i++)
	{
		temp = getParameter(i);
		pT->setParameter(i, temp + epsilon);
		derivative(i) = (eval(x1, x2) - ret) / epsilon;
		pT->setParameter(i, temp);
	}
	return ret;
}

void KernelFunction::model(const Array<double>& input, Array<double> &output)
{
	output.resize(1, false);
	output(0) = eval(input[0], input[1]);
}

void KernelFunction::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	evalDerivative(input[0], input[1], derivative);
}

void KernelFunction::modelDerivative(const Array<double>& input, Array<double> &output, Array<double>& derivative)
{
	output.resize(1, false);
	output(0) = evalDerivative(input[0], input[1], derivative);
}


////////////////////////////////////////////////////////////////////////////////


LinearKernel::LinearKernel()
{
	parameter.resize(0, false);
}

LinearKernel::~LinearKernel()
{
}


double LinearKernel::eval(const Array<double>& x1, const Array<double>& x2) const
{
	unsigned int i, ic = x1.nelem();
	SIZE_CHECK(x1.ndim() == 1);
	SIZE_CHECK(x2.ndim() == 1);
	SIZE_CHECK(x2.nelem() == ic);

	double ret = 0.0;
	for (i = 0; i < ic; i++) ret += x1(i) * x2(i);
	return ret;
}

double LinearKernel::evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const
{
	derivative.resize(0, false);
	return eval(x1, x2);
}


////////////////////////////////////////////////////////////////////////////////


PolynomialKernel::PolynomialKernel(int degree, double offset)
{
	parameter.resize(2, false);
	parameter(0) = degree;
	parameter(1) = offset;
}

PolynomialKernel::~PolynomialKernel()
{
}


double PolynomialKernel::eval(const Array<double>& x1, const Array<double>& x2) const
{
	unsigned int i, ic = x1.nelem();
	SIZE_CHECK(x1.ndim() == 1);
	SIZE_CHECK(x2.ndim() == 1);
	SIZE_CHECK(x2.nelem() == ic);

	double dot = 0.0;
	for (i = 0; i < ic; i++) dot += x1(i) * x2(i);
	return pow(dot + parameter(1), parameter(0));
}

void PolynomialKernel::setParameter(unsigned int index, double value)
{
	if (index == 0)
	{
		parameter(0) = (int)value;
		if (parameter(0) < 1.0) throw SHARKEXCEPTION("[PolynomialKernel::setParameter] invalid degree");
	}
	else if (index == 1)
	{
		parameter(1) = value;
	}
	else throw SHARKEXCEPTION("[PolynomialKernel::setParameter] invalid parameter index");
}

bool PolynomialKernel::isFeasible()
{
	return (parameter(0) > 0.0 && (int)parameter(0) == parameter(0));
}


////////////////////////////////////////////////////////////////////////////////


RBFKernel::RBFKernel(double gamma)
{
	parameter.resize(1, false);
	parameter(0) = gamma;
}

RBFKernel::~RBFKernel()
{
}


double RBFKernel::eval(const Array<double>& x1, const Array<double>& x2) const
{
	unsigned int i, ic = x1.nelem();
	SIZE_CHECK(x1.ndim() == 1);
	SIZE_CHECK(x2.ndim() == 1);
	SIZE_CHECK(x2.nelem() == ic);

	double tmp, dist2 = 0.0;
	for (i = 0; i < ic; i++)
	{
		tmp = x1(i) - x2(i);
		dist2 += tmp * tmp;
	}
	return exp(-parameter(0) * dist2);
}

double RBFKernel::evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const
{
	unsigned int i, ic = x1.nelem();
	SIZE_CHECK(x1.ndim() == 1);
	SIZE_CHECK(x2.ndim() == 1);
	SIZE_CHECK(x2.nelem() == ic);

	derivative.resize(1, false);
	double tmp, dist2 = 0.0;
	for (i = 0; i < ic; i++)
	{
		tmp = x1(i) - x2(i);
		dist2 += tmp * tmp;
	}
	double ret = exp(-parameter(0) * dist2);
	derivative(0) = -dist2 * ret;
	return ret;
}

bool RBFKernel::isFeasible()
{
	return (parameter(0) > 0.0);
}

double RBFKernel::getSigma()
{
	return sqrt(.5 / getParameter(0));
};

void RBFKernel::setSigma(double sigma)
{
	setParameter(0, .5 / (sigma * sigma));
}


////////////////////////////////////////////////////////////////////////////////


NormalizedRBFKernel::NormalizedRBFKernel()
{
	parameter.resize(1, false);
	parameter = 1.;
}

NormalizedRBFKernel::NormalizedRBFKernel(double s)
{
	parameter.resize(1, false);
	parameter = s;
}


void NormalizedRBFKernel::setSigma(double s)
{
	setParameter(0, s);
}

double NormalizedRBFKernel::eval(const Array<double> &x,  const Array<double> &z) const
{
	double sum = 0;
	double sigma = parameter(0);

	for (unsigned i = 0; i < x.dim(0); i++) sum += (x(i) - z(i)) * (x(i) - z(i));
	return exp(-sum / (2*sigma*sigma)) / (sigma * Sqrt2PI);
}

double NormalizedRBFKernel::evalDerivative(const Array<double> &x,  const Array<double> &z, Array<double>& derivative) const
{
	double sum = 0;
	double sigma = parameter(0);

	derivative.resize(1, false);

	for (unsigned i = 0; i < x.dim(0); i++) sum += (x(i) - z(i)) * (x(i) - z(i));
	derivative(0) = (-exp(-sum / (2 * sigma * sigma)) + exp(-sum / (2 * sigma * sigma)) * sum / (sigma * sigma)) / (sigma * sigma * sqrt(2. * M_PI));
	return exp(-sum / (2*sigma*sigma)) / (sigma * Sqrt2PI);
}


////////////////////////////////////////////////////////////////////////////////


NormalizedKernel::NormalizedKernel(KernelFunction* base)
{
	baseKernel = base;
	int i, ic = baseKernel->getParameterDimension();
	parameter.resize(ic, false);
	for (i = 0; i < ic; i++) setParameter(i, baseKernel->getParameter(i));
}

NormalizedKernel::~NormalizedKernel()
{}


void NormalizedKernel::setParameter(unsigned int index, double value)
{
	parameter(index) = value;
	baseKernel->setParameter(index, value);
}

double NormalizedKernel::eval(const Array<double>& x1, const Array<double>& x2) const
{
	return baseKernel->eval(x1, x2) / sqrt(baseKernel->eval(x1, x1) * baseKernel->eval(x2, x2));
}

double NormalizedKernel::evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const
{
	int i, ic = getParameterDimension();
	Array<double> der11(ic);
	Array<double> der12(ic);
	Array<double> der22(ic);

	// compute the single kernel values with derivatives
	double k11 = baseKernel->evalDerivative(x1, x1, der11);
	double k12 = baseKernel->evalDerivative(x1, x2, der12);
	double k22 = baseKernel->evalDerivative(x2, x2, der22);
	double s = sqrt(k11 * k22);

	// compute the derivative using the quotient rule
	derivative.resize(ic, false);
	for (i = 0; i < ic; i++)
	{
		derivative(i) = (der12(i) * k11 * k22 - 0.5 * k12 * (k11 * der22(i) + der11(i) * k22)) / (s * k11 * k22);
	}

	// return the normalized kernel value
	return k12 / s;
}

bool NormalizedKernel::isFeasible()
{
	return baseKernel->isFeasible();
}


////////////////////////////////////////////////////////////////////////////////


WeightedSumKernel::WeightedSumKernel(const std::vector<KernelFunction*>& base)
{
	int i, ic = base.size();
	if (ic == 0) throw SHARKEXCEPTION("[WeightedSumKernel] There must be at least one sub-kernel.");
	int j, jc;
	int p, pc = ic - 1;
	baseKernel.resize(ic);
	weight.resize(ic);
	for (i = 0; i < ic; i++)
	{
		pc += base[i]->getParameterDimension();
		baseKernel[i] = base[i];
		weight[i] = 1.0;
	}
	weightsum = ic;
	parameter.resize(pc, false);

	p = 0;
	for (i = 0; i < ic - 1; i++)
	{
		parameter(p) = 0.0;
		p++;
	}
	for (i = 0; i < ic; i++)
	{
		jc = base[i]->getParameterDimension();
		for (j = 0; j < jc; j++)
		{
			parameter(p) = base[i]->getParameter(j);
			p++;
		}
	}
}

WeightedSumKernel::~WeightedSumKernel()
{
}


void WeightedSumKernel::setParameter(unsigned int index, double value)
{
	parameter(index) = value;

	unsigned int ic = baseKernel.size();
	if (index < ic - 1)
	{
		// compute the weights
		unsigned int i;
		double a;
		weight[0] = 1.0;
		weightsum = 1.0;
		for (i = 0; i < ic - 1; i++)
		{
			a = exp(parameter(i));
			weight[i+1] = a;
			weightsum += a;
		}
	}
	else
	{
		// redirect the parameter to the corresponding sub-kernel
		index -= ic - 1;
		unsigned int i, jc;
		for (i = 0; i < ic; i++)
		{
			jc = baseKernel[i]->getParameterDimension();
			if (index < jc)
			{
				baseKernel[i]->setParameter(index, value);
				break;
			}
			else index -= jc;
		}
	}
}

double WeightedSumKernel::eval(const Array<double>& x1, const Array<double>& x2) const
{
	double k = 0.0;
	int i, ic = baseKernel.size();
	for (i = 0; i < ic; i++) k += weight[i] * baseKernel[i]->eval(x1, x2);
	return k / weightsum;
}

double WeightedSumKernel::evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const
{
	derivative.resize(getParameterDimension(), false);

	int i, ic = baseKernel.size();
	std::vector<double> k_result(ic);
	Array<double> der;
	double k = 0.0;
	int p = ic - 1;
	int j, jc;
	for (i = 0; i < ic; i++)
	{
		k_result[i] = baseKernel[i]->evalDerivative(x1, x2, der);
		k += weight[i] * k_result[i];

		// compute the derivatives w.r.t. the parameters of the sub kernels
		jc = baseKernel[i]->getParameterDimension();
		for (j = 0; j < jc; j++)
		{
			derivative(p) = weight[i] * der(j) / weightsum;
			p++;
		}
	}

	// compute the derivatives w.r.t. the kernel weights
	for (i = 0; i < ic - 1; i++)
	{
		derivative(i) = weight[i+1] * (k_result[i+1] - k / weightsum) / weightsum;
	}

	return k / weightsum;
}

bool WeightedSumKernel::isFeasible()
{
	int i, ic = baseKernel.size();
	for (i = 0; i < ic; i++)
	{
		if (! baseKernel[i]->isFeasible()) return false;
	}
	return true;
}


////////////////////////////////////////////////////////////////////////////////


WeightedSumKernel2::WeightedSumKernel2(const std::vector<KernelFunction*>& base)
{
	int i, ic = base.size();
	if (ic == 0) throw SHARKEXCEPTION("[WeightedSumKernel2] There must be at least one sub-kernel.");
	int j, jc;
	int p, pc = ic;
	baseKernel.resize(ic);
	weight.resize(ic);
	for (i = 0; i < ic; i++)
	{
		pc += base[i]->getParameterDimension();
		baseKernel[i] = base[i];
		weight[i] = 1.0;
	}
	weightsum = ic;
	parameter.resize(pc, false);

	p = 0;
	for (i = 0; i < ic; i++)
	{
		parameter(p) = 0.0;
		p++;
	}
	for (i = 0; i < ic; i++)
	{
		jc = base[i]->getParameterDimension();
		for (j = 0; j < jc; j++)
		{
			parameter(p) = base[i]->getParameter(j);
			p++;
		}
	}
}

WeightedSumKernel2::~WeightedSumKernel2()
{
}


void WeightedSumKernel2::setParameter(unsigned int index, double value)
{
	parameter(index) = value;

	unsigned int ic = baseKernel.size();
	if (index < ic)
	{
		// compute the weights
		unsigned int i;
		double a;
		weightsum = 1.0;
		for (i = 0; i < ic; i++)
		{
			a = exp(parameter(i));
			weight[i] = a;
			weightsum += a;
		}
	}
	else
	{
		// redirect the parameter to the corresponding sub-kernel
		index -= ic;
		unsigned int i, jc;
		for (i = 0; i < ic; i++)
		{
			jc = baseKernel[i]->getParameterDimension();
			if (index < jc)
			{
				baseKernel[i]->setParameter(index, value);
				break;
			}
			else index -= jc;
		}
	}
}

double WeightedSumKernel2::eval(const Array<double>& x1, const Array<double>& x2) const
{
	double k = 0.0;
	int i, ic = baseKernel.size();
	for (i = 0; i < ic; i++) k += weight[i] * baseKernel[i]->eval(x1, x2);
	return k / weightsum;
}

double WeightedSumKernel2::evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const
{
	derivative.resize(getParameterDimension(), false);

	int i, ic = baseKernel.size();
	std::vector<double> k_result(ic);
	Array<double> der;
	double k = 0.0;
	int p = ic;
	int j, jc;
	for (i = 0; i < ic; i++)
	{
		k_result[i] = baseKernel[i]->evalDerivative(x1, x2, der);
		k += weight[i] * k_result[i];

		// compute the derivatives w.r.t. the parameters of the sub kernels
		jc = baseKernel[i]->getParameterDimension();
		for (j = 0; j < jc; j++)
		{
			derivative(p) = weight[i] * der(j) / weightsum;
			p++;
		}
	}

	// compute the derivatives w.r.t. the kernel weights
	for (i = 0; i < ic; i++)
	{
		derivative(i) = weight[i] * (k_result[i] - k / weightsum) / weightsum;
	}

	return k / weightsum;
}

bool WeightedSumKernel2::isFeasible()
{
	int i, ic = baseKernel.size();
	for (i = 0; i < ic; i++)
	{
		if (! baseKernel[i]->isFeasible()) return false;
	}
	return true;
}


////////////////////////////////////////////////////////////////////////////////


PrototypeKernel::PrototypeKernel(unsigned int setsize, unsigned int prototypeDim)
{
	if (prototypeDim == 0) prototypeDim = setsize;

	m_setsize = setsize;
	m_prototypeDim = prototypeDim;
	unsigned int i, total = setsize * prototypeDim;
	parameter.resize(total, false);
	parameter = 0.0;
	for (i=0; i<Shark::min(setsize, prototypeDim); i++) parameter(i * (prototypeDim + 1)) = 1.0;
}

PrototypeKernel::~PrototypeKernel()
{
}


double PrototypeKernel::eval(const Array<double>& x1, const Array<double>& x2) const
{
	unsigned int i = (unsigned int)floor(x1(0));
	unsigned int j = (unsigned int)floor(x2(0));
	RANGE_CHECK(i < m_setsize);
	RANGE_CHECK(j < m_setsize);

	double ret = 0.0;
	unsigned int p;
	for (p=0; p<m_prototypeDim; p++) ret += parameter(m_prototypeDim*i+p) * parameter(m_prototypeDim*j+p);
	return ret;
}

double PrototypeKernel::evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const
{
	unsigned int i = (unsigned int)floor(x1(0));
	unsigned int j = (unsigned int)floor(x2(0));
	RANGE_CHECK(i < m_setsize);
	RANGE_CHECK(j < m_setsize);

	derivative.resize(m_setsize * m_prototypeDim);
	derivative = 0.0;

	double ret = 0.0;
	unsigned int p;
	for (p=0; p<m_prototypeDim; p++)
	{
		unsigned int ii = m_prototypeDim * i + p;
		unsigned int jj = m_prototypeDim * j + p;
		double pi = parameter(ii);
		double pj = parameter(jj);
		ret += pi * pj;
		derivative(ii) += pj;
		derivative(jj) += pi;
	}
	return ret;
}

void PrototypeKernel::getPrototype(unsigned int n, Array<double>& vec)
{
	RANGE_CHECK(n < m_setsize);

	vec.resize(m_prototypeDim, false);
	unsigned int p;
	for (p=0; p<m_prototypeDim; p++) vec(p) = parameter(m_prototypeDim * n + p);
}

void PrototypeKernel::setPrototype(unsigned int n, const Array<double>& vec)
{
	SIZE_CHECK(vec.ndim() == 1);
	SIZE_CHECK(vec.dim(0) == m_prototypeDim);
	RANGE_CHECK(n < m_setsize);

	unsigned int p;
	for (p=0; p<m_prototypeDim; p++) parameter(m_prototypeDim * n + p) = vec(p);
}


////////////////////////////////////////////////////////////////////////////////


JointKernelFunction::JointKernelFunction(KernelFunction& inputkernel, KernelFunction& labelkernel)
: m_inputkernel(inputkernel)
, m_labelkernel(labelkernel)
{
	parameter.resize(inputkernel.getParameterDimension() + labelkernel.getParameterDimension(), false);
	unsigned int i;
	for (i=0; i<inputkernel.getParameterDimension(); i++)
	{
		parameter(i) = inputkernel.getParameter(i);
	}
	for (i=0; i<labelkernel.getParameterDimension(); i++)
	{
		parameter(inputkernel.getParameterDimension() + i) = labelkernel.getParameter(i);
	}
}

JointKernelFunction::~JointKernelFunction()
{
}


double JointKernelFunction::eval(const Array<double>& x1, const Array<double>& y1, const Array<double>& x2, const Array<double>& y2) const
{
	double k1 = m_inputkernel.eval(x1, x2);
	double k2 = m_labelkernel.eval(y1, y2);
	return k1 * k2;
}

double JointKernelFunction::evalDerivative(const Array<double>& x1, const Array<double>& y1, const Array<double>& x2, const Array<double>& y2, Array<double>& derivative) const
{
	Array<double> der1(m_inputkernel.getParameterDimension());
	Array<double> der2(m_labelkernel.getParameterDimension());
	double k1 = m_inputkernel.evalDerivative(x1, x2, der1);
	double k2 = m_labelkernel.evalDerivative(y1, y2, der2);

	unsigned int i;
	derivative.resize(getParameterDimension(), false);
	for (i=0; i<m_inputkernel.getParameterDimension(); i++)
		derivative(i) = k2 * der1(i);
	for (i=0; i<m_labelkernel.getParameterDimension(); i++)
		derivative(m_inputkernel.getParameterDimension() + i) = k1 * der2(i);

	return k1 * k2;
}

void JointKernelFunction::model(const Array<double>& input, Array<double> &output)
{
	throw SHARKEXCEPTION("[JointKernelFunction::model] never call this function - it is undefined.");
}

void JointKernelFunction::setParameter(unsigned int index, double value)
{
	Model::setParameter(index, value);
	if (index < m_inputkernel.getParameterDimension()) m_inputkernel.setParameter(index, value);
	else m_labelkernel.setParameter(index - m_inputkernel.getParameterDimension(), value);
}
