/*!
*  \file KTA.cpp
*
*  \brief Implementation of the (negative) Kernel Target Alignment (KTA) 
* 		   as proposed by Nello Cristianini
*
*  \author  T. Glasmachers
*  \date    2006, 2009
*
*  \par
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


#include <ReClaM/KTA.h>
#include <Array/ArrayOp.h>


NegativeKTA::NegativeKTA(unsigned int numberOfClasses)
{
	m_numberOfClasses = numberOfClasses;
	m_offdiag = -1.0 / (numberOfClasses - 1.0);
	m_2offdiag = 2.0 * m_offdiag;
	m_offdiag2 = m_offdiag * m_offdiag;
	m_2offdiag2 = 2.0 * m_offdiag2;
}

NegativeKTA::~NegativeKTA()
{
}


double NegativeKTA::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	// check the model type
	KernelFunction* pKernel = dynamic_cast<KernelFunction*>(&model);
	JointKernelFunction* pJointKernel = dynamic_cast<JointKernelFunction*>(&model);
	if (pKernel == NULL && pJointKernel == NULL) throw SHARKEXCEPTION("[NegativeKTA::error] model is not a valid KernelFunction or JointKernelFunction.");

	unsigned int i, j, ell = input.dim(0);
	double k;
	double DD = 0.0;
	double DK = 0.0;
	double KK = 0.0;

	if (pKernel != NULL)
	{
		for (i=0; i<ell; i++)
		{
			for (j=0; j<i; j++)
			{
				k = pKernel->eval(input[i], input[j]);
				if (target(i, 0) == target(j, 0))
				{
					DD += 2.0;
					DK += 2.0 * k;
				}
				else
				{
					DD += m_2offdiag2;
					DK += m_2offdiag * k;
				}
				KK += 2.0 * k * k;
			}
			k = pKernel->eval(input[i], input[i]);
			DD += 1.0;
			DK += k;
			KK += k * k;
		}

		return (-DK / sqrt(DD * KK));
	}
	else
	{
		for (i=0; i<ell; i++)
		{
			for (j=0; j<i; j++)
			{
				k = pJointKernel->eval(input[i], target[i], input[j], target[j]);
				if (target(i, 0) == target(j, 0))
				{
					DD += 2.0;
					DK += 2.0 * k;
				}
				else
				{
					DD += m_2offdiag2;
					DK += m_2offdiag * k;
				}
				KK += 2.0 * k * k;
			}
			k = pJointKernel->eval(input[i], target[i], input[i], target[i]);
			DD += 1.0;
			DK += k;
			KK += k * k;
		}

		return (-DK / sqrt(DD * KK));
	}
}

double NegativeKTA::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	// check the model type
	KernelFunction* pKernel = dynamic_cast<KernelFunction*>(&model);
	JointKernelFunction* pJointKernel = dynamic_cast<JointKernelFunction*>(&model);
	if (pKernel == NULL && pJointKernel == NULL) throw SHARKEXCEPTION("[NegativeKTA::errorDerivative] model is not a valid KernelFunction or JointKernelFunction.");

	unsigned int i, j, ell = input.dim(0);
	double k;
	double DD = 0.0;
	double DK = 0.0;
	double KK = 0.0;

	if (pKernel != NULL)
	{
		unsigned int pc = pKernel->getParameterDimension();
		derivative.resize(pc, false);
		Array<double> kd(pc);
		Array<double> der1(pc); der1 = 0.0;
		Array<double> der2(pc); der2 = 0.0;

		for (i=0; i<ell; i++)
		{
			for (j=0; j<i; j++)
			{
				k = pKernel->evalDerivative(input[i], input[j], kd);
				double _2k = 2.0 * k;

				if (target(i, 0) == target(j, 0))
				{
					DD += 2.0;
					DK += _2k;
					der1 += 2.0 * kd;
				}
				else
				{
					DD += m_2offdiag2;
					DK += m_2offdiag * k;
					der1 += m_2offdiag * kd;
				}
				KK += _2k * k;
				der2 += _2k * kd;
			}
			k = pKernel->evalDerivative(input[i], input[i], kd);
			DD += 1.0;
			DK += k;
			KK += k * k;
			der1 += kd;
			der2 += kd;
		}

		double denom = sqrt(DD * KK);
		double f1 = 1.0 / denom;
		double f2 = DK / (KK * denom);
		derivative = f2 * der2 - f1 * der1;

		return (-DK / denom);
	}
	else
	{
		unsigned int pc = pJointKernel->getParameterDimension();
		derivative.resize(pc, false);
		Array<double> kd(pc);
		Array<double> der1(pc); der1 = 0.0;
		Array<double> der2(pc); der2 = 0.0;

		for (i=0; i<ell; i++)
		{
			for (j=0; j<i; j++)
			{
				k = pJointKernel->evalDerivative(input[i], target[i], input[j], target[j], kd);
				double _2k = 2.0 * k;

				if (target(i, 0) == target(j, 0))
				{
					DD += 2.0;
					DK += _2k;
					der1 += 2.0 * kd;
				}
				else
				{
					DD += m_2offdiag2;
					DK += m_2offdiag * k;
					der1 += m_2offdiag * kd;
				}
				KK += _2k * k;
				der2 += _2k * kd;
			}
			k = pJointKernel->evalDerivative(input[i], target[i], input[i], target[i], kd);
			DD += 1.0;
			DK += k;
			KK += k * k;
			der1 += kd;
			der2 += kd;
		}

		double denom = sqrt(DD * KK);
		double f1 = 1.0 / denom;
		double f2 = DK / (KK * denom);
		derivative = f2 * der2 - f1 * der1;

		return (-DK / denom);
	}
}


////////////////////////////////////////////////////////////


NegativeBKTA::NegativeBKTA()
{
}

NegativeBKTA::~NegativeBKTA()
{
}


double NegativeBKTA::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	// check the model type
	C_SVM* pCSVM = dynamic_cast<C_SVM*>(&model);
	KernelFunction* pKernel = dynamic_cast<KernelFunction*>(&model);
	bool norm2 = false;
	if (pCSVM != NULL)
	{
		if (! pCSVM->is2norm()) throw SHARKEXCEPTION("[NegativeBKTA::error] C_SVM must use 2-norm slack penalty.");
		pKernel = pCSVM->getSVM()->getKernel();
		norm2 = true;
	}
	else if (pKernel == NULL) throw SHARKEXCEPTION("[NegativeBKTA::error] model is not a valid C_SVM or KernelFunction.");

	int i, j, T = input.dim(0);

	int n = 0;
	int p = 0;
	double mult_pp, mult_nn;
		for (i = 0; i < T; i++) if (target(i, 0) > 0.0) p++; else n++;
	mult_pp = ((double)n) / ((double)p);
	mult_nn = ((double)p) / ((double)n);

	double k, temp;
	double A = 0.0;
	double S = 0.0;
	for (i = 0; i < T; i++)
	{
		for (j = 0; j < i; j++)
		{
			k = pKernel->eval(input[i], input[j]);
			if (target(i, 0) > 0.0 && target(j, 0) > 0.0)
			{
				temp = 2.0 * mult_pp * k;
				A += temp;
				S += temp * k;
			}
			else if (target(i, 0) < 0.0 && target(j, 0) < 0.0)
			{
				temp = 2.0 * mult_nn * k;
				A += temp;
				S += temp * k;
			}
			else
			{
				temp = 2.0 * k;
				A -= temp;
				S += temp * k;
			}
		}
		k = pKernel->eval(input[i], input[i]);
		if (norm2) k += (target(i, 0) > 0.0) ? 1.0 / pCSVM->get_Cplus() : 1.0 / pCSVM->get_Cminus();
		if (target(i, 0) > 0.0)
		{
			temp = mult_pp * k;
			A += temp;
			S += temp * k;
		}
		else
		{
			temp = mult_nn * k;
			A += temp;
			S += temp * k;
		}
	}
	return -0.5 * A / (sqrt(S *((double)p) *((double)n)));
}

double NegativeBKTA::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	// check the model type
	C_SVM* pCSVM = dynamic_cast<C_SVM*>(&model);
	KernelFunction* pKernel = dynamic_cast<KernelFunction*>(&model);
	bool norm2 = false;
	if (pCSVM != NULL)
	{
		if (! pCSVM->is2norm()) throw SHARKEXCEPTION("[NegativeBKTA::errorDerivative] C_SVM must use 2-norm slack penalty.");
		pKernel = pCSVM->getSVM()->getKernel();
		norm2 = true;
	}
	else if (pKernel == NULL) throw SHARKEXCEPTION("[NegativeBKTA::errorDerivative] model is not a valid C_SVM or KernelFunction.");

	int i, j, T = input.dim(0);

	int n = 0;
	int p = 0;
	double mult, amult, mult_pp, mult_nn;
		for (i = 0; i < T; i++) if (target(i, 0) > 0.0) p++; else n++;
	mult_pp = ((double)n) / ((double)p);
	mult_nn = ((double)p) / ((double)n);

	int d;
	int mD = model.getParameterDimension();
	int kD = pKernel->getParameterDimension();
	double k;
	double A = 0.0;
	double S = 0.0;
	Array<double> a(kD);
	Array<double> b(kD);
	Array<double> der(kD);
	derivative.resize(mD, false);

	a = 0.0;
	b = 0.0;
	double aCplus = 0.0;
	double aCminus = 0.0;
	double bCplus = 0.0;
	double bCminus = 0.0;
	derivative = 0.0;

	for (i = 0; i < T; i++)
	{
		for (j = 0; j < i; j++)
		{
			k = pKernel->evalDerivative(input[i], input[j], der);
			if (target(i, 0) > 0.0 && target(j, 0) > 0.0) mult = amult = 2.0 * mult_pp;
			else if (target(i, 0) < 0.0 && target(j, 0) < 0.0) mult = amult = 2.0 * mult_nn;
			else
			{
				mult = -2.0; amult = 2.0;
			}

			for (d = 0; d < kD; d++)
			{
				a(d) += mult * der(d);
				b(d) += amult * k * der(d);
			}
			A += mult * k;
			S += amult * k * k;
		}
		k = pKernel->evalDerivative(input[i], input[i], der);
		if (norm2)
		{
			if (target(i, 0) > 0.0)
			{
				double invC = 1.0 / pCSVM->get_Cplus();
				k += invC;
				aCplus -= mult_pp * invC * invC;
				bCplus -= mult_pp * k * invC * invC;
			}
			else
			{
				double invC = 1.0 / pCSVM->get_Cminus();
				k += invC;
				aCminus -= mult_nn * invC * invC;
				bCminus -= mult_nn * k * invC * invC;
			}
		}
		if (target(i, 0) > 0.0) mult = mult_pp;
		else mult = mult_nn;
		for (d = 0; d < kD; d++)
		{
			a(d) += mult * der(d);
			b(d) += mult * k * der(d);
		}
		A += mult * k;
		S += mult * k * k;
	}

	double N = 2.0 * sqrt(((double)n) * ((double)p) * S);
	if (norm2)
	{
		double cr = pCSVM->getCRatio();
		derivative(0) = (A * (bCplus + bCminus / cr) / S - (aCplus + aCminus / cr)) / N;
		for (d = 0; d < kD; d++)
		{
			derivative(d + 1) = (A * b(d) / S - a(d)) / N;
		}
	}
	else
	{
		for (d = 0; d < kD; d++)
		{
			derivative(d) = (A * b(d) / S - a(d)) / N;
		}
	}

	return -A / N;
}

