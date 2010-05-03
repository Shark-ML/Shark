//===========================================================================
/*!
*  \file SpanBound2.cpp
*
*  \brief Compute the SpanBound for the 2-norm SVM
*
*  \author  T. Glasmachers
*  \date    2006
*
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


#include <ReClaM/SpanBound2.h>
#include <ReClaM/Svm.h>


SpanBound2::SpanBound2(bool verbose)
{
	this->verbose = verbose;
	maxIter = -1;
}

SpanBound2::~SpanBound2()
{
}


double SpanBound2::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	C_SVM* csvm = dynamic_cast<C_SVM*>(&model);
	if (csvm == NULL) throw SHARKEXCEPTION("[SpanBound2::error] model is not a valid C_SVM");

	bool norm2 = csvm->is2norm();
	SVM* svm = csvm->getSVM();

	if (verbose) { printf("[span bound] SVM training ..."); fflush(stdout); }
	SVM_Optimizer svmopt;
	svmopt.init(*csvm);
	svmopt.setMaxIterations(maxIter);
	svmopt.optimize(*svm, input, target);

	QpSvmDecomp* solver = (QpSvmDecomp*)svmopt.get_Solver();
	if (! solver->isOptimal()) return 1e100;
	iter = solver->iterations();

	int i, p, examples = input.dim(0);
	Array<double> linear(examples);
	Array<double> lower(examples);
	Array<double> upper(examples);
	Array<double> lambda(examples);

	// prepare the quadratic programs for the span
	linear = 0.0;

	// for all support vectors
	if (norm2)
	{
		if (verbose)
		{
			int sv = 0;
			for (i = 0; i < examples; i++) if (svm->getAlpha(i) != 0.0) sv++;
			printf(" #SV=%d ", sv);
			fflush(stdout);
		}

		int wrong = 0;
		for (p = 0; p < examples; p++)
		{
			double ap = fabs(svm->getAlpha(p));
			if (ap > 0.0)
			{
				double prediction = svm->model(input[p]);
				double threshold = target(p, 0) * prediction / ap;
				if (threshold < 0.0)
				{
					wrong++;
					if (verbose) { printf("x"); fflush(stdout); }
				}
				else
				{
					// prepare the quadratic program
					// and find a feasible point
					double sum = -1.0;
					for (i = 0; i < examples; i++)
					{
						double ai = fabs(svm->getAlpha(i));
						if (i == p)
						{
							lower(i) = -1.0;
							upper(i) = -1.0;
							lambda(i) = -1.0;
						}
						else if (ai != 0.0)
						{
							if (target(i, 0) == target(p, 0))
							{
								lower(i) = -ai / ap;
								upper(i) = 1e100;
							}
							else
							{
								lower(i) = -1e100;
								upper(i) = ai / ap;
							}
							if (sum < 0.0)
							{
								if (sum + upper(i) >= 0.0)
								{
									lambda(i) = -sum;
									sum = 0.0;
								}
								else
								{
									lambda(i) = upper(i);
									sum += upper(i);
								}
							}
							else
							{
								lambda(i) = 0.0;
							}
						}
						else
						{
							lower(i) = 0.0;
							upper(i) = 0.0;
							lambda(i) = 0.0;
						}
					}

					// compute the span and the coefficients lambda
					double span2 = -2.0 * solver->Solve(linear, lower, upper, lambda, 0.001, -0.5 * threshold);
					if (span2 >= threshold)
					{
						wrong++;
						if (verbose) { printf("x"); fflush(stdout); }
					}
					else
					{
						if (verbose) { printf("."); fflush(stdout); }
					}
				}
			}
		}
		if (verbose) printf(" %d\n", wrong);
		return ((double)wrong / (double)examples);
	}
	else
	{
		throw SHARKEXCEPTION("[SpanBound2::error] The span bound for the 1-norm-SVM is not implemented yet.");
	}

	return 0.0;		// dead code
}

