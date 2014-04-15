//===========================================================================
/*!
 * 
 *
 * \brief       Jaakkola's heuristic and related quantities for Gaussian kernel selection
 * 
 * 
 *
 * \author      T. Glasmachers, O. Krause, C. Igel
 * \date        2010
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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


#ifndef SHARK_ALGORITHMS_JAAKKOLAHEURISTIC_H
#define SHARK_ALGORITHMS_JAAKKOLAHEURISTIC_H


#include <shark/Data/Dataset.h>
#include <shark/Core/Traits/ProxyReferenceTraits.h>

#include <boost/range/adaptor/filtered.hpp>
#include <algorithm>

namespace shark{


/// \brief Jaakkola's heuristic and related quantities for Gaussian kernel selection
///
/// \par
/// Jaakkola's heuristic method for setting the width parameter of the
/// Gaussian radial basis function kernel is to pick a quantile (usually
/// the median) of the distribution of Euclidean distances between points
/// having different labels. The present implementation computes the kernel
/// width \f$ \sigma \f$ and the bandwidth
///    \f[ \gamma = \frac{1}{2 \sigma^2} \f]
/// based on the median or on any other quantile of the empirical
/// distribution.
///
/// By default, only the distance to the closest point with different
/// label is considered. This behavior can be turned off by an option
/// of the constructor. This is faster andin accordance with the
/// original paper.
class JaakkolaHeuristic
{
public:
	/// Constructor
	/// \param dataset           vector-valued input data
	/// \param nearestFalseNeighbor  if true, only the nearest neighboring point with different label is considered (default true)
	template<class InputType>
	JaakkolaHeuristic(LabeledData<InputType,unsigned int> const& dataset, bool nearestFalseNeighbor = true)
	{
		typedef typename LabeledData<InputType,unsigned int>::const_element_range Elements;
		typedef typename ConstProxyReference<InputType const>::type Element;
		Elements elements = dataset.elements();
		if(!nearestFalseNeighbor) {
			for(typename Elements::iterator it = elements.begin(); it != elements.end(); ++it){
				Element x = it->input;
				typename Elements::iterator itIn = it;
				itIn++;
				for (; itIn != elements.end(); itIn++) {
					if (itIn->label == it->label) continue;
					Element y = itIn->input;
					double dist = distanceSqr(x,y);
					m_stat.push_back(dist);
				}
			}

		} else {
			std::size_t classes = numberOfClasses(dataset);
			std::size_t dim = inputDimension(dataset);
			m_stat.resize(dataset.numberOfElements());
			std::fill(m_stat.begin(),m_stat.end(), std::numeric_limits<double>::max());
			std::size_t blockStart = 0;
			for(std::size_t c = 0; c != classes; ++c){
				
				typename Elements::iterator leftIt = elements.begin();
				typename Elements::iterator end = elements.end();
				while(leftIt != end){
					//todo: use a filter on the iterator
					//create the next batch containing only elements of class c as left argument to distanceSqr
					typename Batch<InputType>::type leftBatch(512, dim);
					std::size_t leftElements = 0;
					while(leftElements < 512 && leftIt != end){
						if(leftIt->label == c){
							row(leftBatch,leftElements) = leftIt->input;
							++leftElements;
						}
						++leftIt;
					}
					//now go through all elements and again create batches, this time of all elements which are not of class c
					typename Elements::iterator rightIt = elements.begin();
					while(rightIt != end){
						typename Batch<InputType>::type rightBatch(512, dim);
						std::size_t rightElements = 0;
						while(rightElements < 512 && rightIt != end){
							if(rightIt->label != c){
								row(rightBatch,rightElements) = rightIt->input;
								++rightElements;
							}
							++rightIt;
						}

						//now compute distances and update shortest distance
						RealMatrix distances = distanceSqr(leftBatch,rightBatch);
						for(std::size_t i = 0; i != leftElements;++i){
							m_stat[blockStart+i]=std::min(min(subrange(row(distances,i),0,rightElements)),m_stat[blockStart+i]);
						}
					}
					blockStart+= leftElements;
				}
			}
			
			//~ for(typename Elements::iterator it = elements.begin(); it != elements.end(); ++it){
				//~ double minDistSqr = std::numeric_limits<double>::max();//0;
				//~ Element x = it->input;
				//~ for (typename Elements::iterator itIn = elements.begin(); itIn != elements.end(); itIn++) {
					//~ if (itIn->label == it->label) continue;
					//~ Element y = itIn->input;
					//~ double dist = distanceSqr(x,y);
					//~ //if( (minDistSqr == 0) || (dist < minDistSqr))  minDistSqr = dist;
					//~ if(dist < minDistSqr)  minDistSqr = dist;
				//~ }
				//~ m_stat.push_back(minDistSqr);
			//~ }
			
		}
		std::sort(m_stat.begin(), m_stat.end());
	}
		
	/// Compute the given quantile (usually median)
	/// of the empirical distribution of Euclidean distances
	/// of data pairs with different labels.
	double sigma(double quantile = 0.5)
	{
		std::size_t ic = m_stat.size();
		SHARK_ASSERT(ic > 0);

		std::sort(m_stat.begin(), m_stat.end());

		if (quantile < 0.0)
		{
			// TODO: find minimum
			return std::sqrt(m_stat[0]);
		}
		if (quantile >= 1.0)
		{
			// TODO: find maximum
			return std::sqrt(m_stat[ic-1]);
		}
		else
		{
			// TODO: partial sort!
			double t = quantile * (ic - 1);
			std::size_t i = (std::size_t)floor(t);
			double rest = t - i;
			return ((1.0 - rest) * std::sqrt(m_stat[i]) + rest * std::sqrt(m_stat[i+1]));
		}
	}

	/// Compute the given quantile (usually the median)
	/// of the empirical distribution of Euclidean distances
	/// of data pairs with different labels converted into
	/// a value usable as the gamma parameter of the GaussianRbfKernel.
	double gamma(double quantile = 0.5)
	{
		double s = sigma(quantile);
		return 0.5 / (s * s);
	}


private:
	/// all pairwise distances
	std::vector<double> m_stat;
};

}
#endif
