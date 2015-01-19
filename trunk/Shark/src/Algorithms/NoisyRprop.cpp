//===========================================================================
/*!
 * 
 *
 * \brief       NoisyRprop
 * 
 * 
 *
 * \author      O.Krause
 * \date        2010-2011
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#include <shark/Algorithms/GradientDescent/NoisyRprop.h>
#include <algorithm>

using namespace shark;

NoisyRprop::NoisyRprop(){
	m_features |= REQUIRES_FIRST_DERIVATIVE;
}

void NoisyRprop::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint){
	init(objectiveFunction,startingPoint,0.01);
}

void NoisyRprop::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint,double delta0P){
	checkFeatures(objectiveFunction);
	RealVector d(startingPoint.size());
	std::fill(d.begin(),d.end(),delta0P);
	init(objectiveFunction,startingPoint, d);
}

void NoisyRprop::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint,const RealVector& delta0P){
	SIZE_CHECK(startingPoint.size()==delta0P.size());
	m_best.point=startingPoint;
	m_minEpisodeLength = 40;

	size_t pc = startingPoint.size();

	m_episodeLength.resize(pc);
	m_sample.resize(pc);
	m_sampleIteration.resize(pc);
	m_iteration.resize(pc);
	m_toTheRight.resize(pc);
	m_position.resize(pc);
	m_leftmost.resize(pc);
	m_rightmost.resize(pc);
	m_leftsum.resize(pc);
	m_rightsum.resize(pc);
	m_current.resize(pc);
	m_numberOfAverages.resize(pc);
	m_gradient.resize(pc);

	m_delta   = delta0P;
	std::fill(m_episodeLength.begin(),m_episodeLength.end(),m_minEpisodeLength);
	std::fill(m_sample.begin(),m_sample.end(),0);
	std::fill(m_sampleIteration.begin(),m_sampleIteration.end(),m_minEpisodeLength / 100);
	std::fill(m_iteration.begin(),m_iteration.end(),0);
	std::fill(m_position.begin(),m_position.end(),0);
	std::fill(m_leftmost.begin(),m_leftmost.end(),0);
	std::fill(m_rightmost.begin(),m_rightmost.end(),0);
	std::fill(m_toTheRight.begin(),m_toTheRight.end(),0);
	m_leftsum.clear();
	m_rightsum.clear();
	std::fill(m_current.begin(),m_current.end(),0);
	std::fill(m_numberOfAverages.begin(),m_numberOfAverages.end(),1);
	m_gradient.clear();
}

void NoisyRprop::step(const ObjectiveFunctionType& objectiveFunction)
{
	size_t pc = m_best.point.size();
	ObjectiveFunctionType::FirstOrderDerivative dedw;

	// compute noisy error function value and derivative
	m_best.value = objectiveFunction.evalDerivative(m_best.point,dedw);

	// loop over all coordinates
	m_gradient += dedw;
	for (size_t p=0; p<pc; p++)
	{
		m_current[p]++;
		if (m_current[p] == m_numberOfAverages[p])
		{
			// basic Rprop algorithm
			double value = m_best.point(p);
			if (m_gradient(p) < 0.0)
			{
				m_best.point(p)+=m_delta(p);
				m_position[p]++;
				m_toTheRight[p]++;
				m_rightsum(p) -= m_gradient(p);
			}
			else if (m_gradient(p) > 0.0)
			{
				m_best.point(p)-=m_delta(p);
				m_position[p]--;
				m_leftsum(p) += m_gradient(p);
			}

			// collect leftmost and rightmost of 100 position samples
			if (m_iteration[p] == m_sampleIteration[p])
			{
				if (m_position[p] > m_rightmost[p]) m_rightmost[p] = m_position[p];
				if (m_position[p] < m_leftmost[p]) m_leftmost[p] = m_position[p];
				m_sample[p]++;
				m_sampleIteration[p] = m_sample[p] * m_episodeLength[p] / 100;
				if (m_sampleIteration[p] <= m_iteration[p]) m_sampleIteration[p] = m_iteration[p] + 1;
			}

			if (!objectiveFunction.isFeasible(m_best.point))
			{
				m_best.point(p)=value;
			}

			// strategy adaptation
			m_iteration[p]++;
			if (m_iteration[p] == m_episodeLength[p])
			{
				// Compute the normalization of the toTheRight statistics
				// and compare it to its expected normal approximation
				unsigned int N = m_episodeLength[p];
				unsigned int newN = N;
				double normalized = std::fabs((m_toTheRight[p] - 0.5 * N) / std::sqrt((double)N));
				if (normalized > 1.64)				// 90% quantile
				{
					// increase the step size
					m_delta(p) *= 2.0;
					newN /= 2;
				}
				else
				{
					if (normalized < 0.25)			// 20% quantile
					{
						// decrease the step size
						m_delta(p) /= 2.0;
						newN *= 2;
					}

					// Compute the normalization of the spread statistics
					// and compare it to its expected distribution

					// compute approximate quantiles of the spread statistics
					int spread = m_rightmost[p] - m_leftmost[p];
					normalized = spread / std::sqrt((double)N);

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
					// and compare it to its expected distribution
					normalized = std::fabs(m_rightsum[p] / m_toTheRight[p] - m_leftsum[p] / (N - m_toTheRight[p])) / (m_rightsum[p] + m_leftsum[p]) * N * std::sqrt((double)N);

					if (normalized < 0.29)			// 15% quantile
					{
						// decrease averaging
						if (m_numberOfAverages[p] > 1)
						{
							m_numberOfAverages[p] /= 2;
							newN *= 2;
						}
					}
					else if (normalized > 2.5)		// 90% quantile
					{
						// increase averaging
						m_numberOfAverages[p] *= 2;
						newN /= 2;
					}
				}
				if (newN < m_minEpisodeLength) newN = m_minEpisodeLength;
				m_episodeLength[p] = newN;

				// reset episode
				m_iteration[p] = 0;
				m_sample[p] = 0;
				m_sampleIteration[p] = m_minEpisodeLength / 100;
				m_position[p] = 0;
				m_leftmost[p] = 0;
				m_rightmost[p] = 0;
				m_toTheRight[p] = 0;
				m_leftsum(p) = 0.0;
				m_rightsum(p) = 0.0;
			}

			m_current[p] = 0;
			m_gradient(p) = 0.0;
		}
	}
}
