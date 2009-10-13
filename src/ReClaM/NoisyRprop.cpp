//===========================================================================
/*!
 *  \file NoisyRprop.cpp
 *
 *  \brief Rprop for noisy function evaluations
 *
 *  \author  T. Glasmachers
 *  \date    2007
 *
 *  \par Copyright (c) 1999-2007:
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
 *  <BR>
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


#include <ReClaM/NoisyRprop.h>


NoisyRprop::NoisyRprop()
{
}

NoisyRprop::~NoisyRprop()
{
}


void NoisyRprop::init(Model& model)
{
	initUserDefined(model);
}

void NoisyRprop::initUserDefined(Model& model, double delta0P)
{
	Array<double> d(model.getParameterDimension());
	d = delta0P;
	initUserDefined(model, d);
}

void NoisyRprop::initUserDefined(Model& model, const Array<double>& delta0P)
{
	minEpisodeLength = 40;

	int pc = model.getParameterDimension();

	delta.resize(pc, false);
	EpisodeLength.resize(pc, false);
	sample.resize(pc, false);
	sampleIteration.resize(pc, false);
	iteration.resize(pc, false);
	toTheRight.resize(pc, false);
	position.resize(pc, false);
	leftmost.resize(pc, false);
	rightmost.resize(pc, false);
	leftsum.resize(pc, false);
	rightsum.resize(pc, false);
	current.resize(pc, false);
	numberOfAverages.resize(pc, false);
	gradient.resize(pc, false);

	delta   = delta0P;
	EpisodeLength = minEpisodeLength;
	sample = 0;
	sampleIteration = minEpisodeLength / 100;
	iteration = 0;
	position = 0;
	leftmost = 0;
	rightmost = 0;
	toTheRight = 0;
	leftsum = 0.0;
	rightsum = 0.0;
	current = 0;
	numberOfAverages = 1;
	gradient = 0.0;
}

double NoisyRprop::optimize(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target)
{
	int p, pc = model.getParameterDimension();
	Array<double> dedw(pc);

	// compute noisy error function value and derivative
	double currentError = errorfunction.errorDerivative(model, input, target, dedw);

	// loop over all coordinates
	for (p=0; p<pc; p++)
	{
		gradient(p) += dedw(p);
		current(p)++;
		if (current(p) == numberOfAverages(p))
		{
			// basic Rprop algorithm
			double value = model.getParameter(p);
			if (gradient(p) < 0.0)
			{
				model.setParameter(p, value + delta(p));
				position(p)++;
				toTheRight(p)++;
				rightsum(p) -= gradient(p);
			}
			else if (gradient(p) > 0.0)
			{
				model.setParameter(p, value - delta(p));
				position(p)--;
				leftsum(p) += gradient(p);
			}

			// collect leftmost and rightmost of 100 position samples
			if (iteration(p) == sampleIteration(p))
			{
				if (position(p) > rightmost(p)) rightmost(p) = position(p);
				if (position(p) < leftmost(p)) leftmost(p) = position(p);
				sample(p)++;
				sampleIteration(p) = sample(p) * EpisodeLength(p) / 100;
				if (sampleIteration(p) <= iteration(p)) sampleIteration(p) = iteration(p) + 1;
			}

			if (! model.isFeasible())
			{
				model.setParameter(p, value);
			}

			// strategy adaptation
			iteration(p)++;
			if (iteration(p) == EpisodeLength(p))
			{
				// Compute the normalization of the toTheRight statistics
				// and compare it to its expected normal approximation
				int N = EpisodeLength(p);
				int newN = N;
				double normalized = fabs((toTheRight(p) - 0.5 * N) / sqrt((double)N));
				if (normalized > 1.64)				// 90% quantile
				{
					// increase the step size
					delta(p) *= 2.0;
					newN /= 2;
				}
				else
				{
					if (normalized < 0.25)			// 20% quantile
					{
						// decrease the step size
						delta(p) /= 2.0;
						newN *= 2;
					}

					// Compute the normalization of the spread statistics
					// and compare it to its expected distribution

					// compute approximate quantiles of the spread statistics
					int spread = rightmost(p) - leftmost(p);
					normalized = spread / sqrt((double)N);

					if (normalized < 0.95)			// 10% quantile
					{
						// decrease the number of iterations
						newN /= 2;
					}
					else if (normalized > 1.77)		// 75% quantile
					{
						// increase the number of iterations
						newN *= 2;
					}

					// Compute the normalization of the asymmetry statistics
					// and compare it to its expectred distribution
					normalized = fabs(rightsum(p) / toTheRight(p) - leftsum(p) / (N - toTheRight(p))) / (rightsum(p) + leftsum(p)) * N * sqrt((double)N);

					if (normalized < 0.29)			// 15% quantile
					{
						// decrease averaging
						if (numberOfAverages(p) > 1)
						{
							numberOfAverages(p) /= 2;
							newN *= 2;
						}
					}
					else if (normalized > 2.5)		// 90% quantile
					{
						// increase averaging
						numberOfAverages(p) *= 2;
						newN /= 2;
					}
				}
				if (newN < minEpisodeLength) newN = minEpisodeLength;
				EpisodeLength(p) = newN;

				// reset episode
				iteration(p) = 0;
				sample(p) = 0;
				sampleIteration(p) = minEpisodeLength / 100;
				position(p) = 0;
				leftmost(p) = 0;
				rightmost(p) = 0;
				toTheRight(p) = 0;
				leftsum(p) = 0.0;
				rightsum(p) = 0.0;
			}

			current(p) = 0;
			gradient(p) = 0.0;
		}
	}

	// return the previous error value
	return currentError;
}
