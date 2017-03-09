/*!
 *
 * \author      O.Krause, T. Glasmachers
 * \date        2014-2016
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUME_CONTRIBUTION_2D_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUME_CONTRIBUTION_2D_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/utility/KeyValuePair.h>

#include <algorithm>
#include <vector>

namespace shark{
/// \brief Finds the smallest/largest Contributors given 2D points
///
/// The algorithm sorts the points by their first coordinate and then 
/// calculates the volume of the hypervolume dominated by each Pointseparately
/// returning the index of the elements with the minimum/maximum contribution.
struct HypervolumeContribution2D{
private:
	struct Point{
		Point(){}

		Point(double f1, double f2, std::size_t index)
		: f1(f1)
		, f2(f2)
		, index(index)
		{}

		bool operator<(Point const& rhs) const{//for lexicographic sorting
			if (f1 < rhs.f1) return true;
			if (f1 > rhs.f1) return false;
			return (f2 < rhs.f2);
		}

		double f1;
		double f2;
		std::size_t index;
	};
	
	///\brief Stores the k elements with the best contribution(highest,lowest) as indicated by comparator
	///
	/// The algorithm returns the indizes of the front front[i].index in order of contribution.
	/// The input is a set of points sorted by x-value. the edge-points are never selected.
	///
	/// This is implemented by using a min-heap that stores the k best elements,
	/// but having the smallest element on top so that we can quickly decide which
	/// element to remove.
	template<class Comparator>
	std::vector<KeyValuePair<double,std::size_t> > bestContributors( std::vector<Point> const& front, std::size_t k, Comparator comp)const{
		
		std::vector<KeyValuePair<double,std::size_t> > bestK(k+1);
		auto heapStart = bestK.begin();
		auto heapEnd = bestK.begin();
		
		auto pointComp = [&](KeyValuePair<double,std::size_t> const& lhs, KeyValuePair<double,std::size_t> const& rhs){return comp(lhs.key,rhs.key);};
		
		//compute the hypervalue contribution for each Pointexcept the endpoints;
		for(std::size_t i = 1; i < front.size()-1;++i){
			double contribution = (front[i+1].f1 - front[i].f1)*(front[i-1].f2 - front[i].f2);
			*heapEnd = makeKeyValuePair(contribution,front[i].index);
			++heapEnd;
			std::push_heap(heapStart,heapEnd,pointComp);
			
			if(heapEnd == bestK.end()){
				std::pop_heap(heapStart,heapEnd,pointComp);
				--heapEnd;
			}
		}
		std::sort_heap(heapStart,heapEnd,pointComp);
		bestK.pop_back();//remove the k+1th element
		return bestK;
	}
public:
	/// \brief Returns the index of the points with smallest contribution.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the smallest contributor.
	/// \param [in] k The number of points to select.
	/// \param [in] referencePointThe reference Point\f$\vec{r} \in \mathbb{R}^2\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$.
	template<typename Set, typename VectorType>
	std::vector<KeyValuePair<double,std::size_t> > smallest( Set const& points, std::size_t k, VectorType const& referencePoint)const{
		SHARK_RUNTIME_CHECK(points.size() >= k, "There must be at least k points in the set");
		std::vector<Point> front;
		front.emplace_back(0,referencePoint[1],points.size()+1);//add reference point
		for(std::size_t i  = 0; i != points.size(); ++i)
			front.emplace_back(points[i][0],points[i][1],i);
		front.emplace_back(referencePoint[0],0,points.size()+1);//add reference point
		std::sort(front.begin()+1,front.end()-1);
		
		return bestContributors(front,k,[](double con1, double con2){return con1 < con2;});
	}
	
	/// \brief Returns the index of the points with smallest contribution.
	///
	/// As no reference point is given, the edge points can not be computed and are not selected.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the smallest contributor.
	/// \param [in] k The number of points to select.
	template<typename Set>
	std::vector<KeyValuePair<double,std::size_t> > smallest( Set const& points, std::size_t k)const{
		SHARK_RUNTIME_CHECK(points.size() >= k, "There must be at least k points in the set");
		std::vector<Point> front;
		for(std::size_t i  = 0; i != points.size(); ++i)
			front.emplace_back(points[i][0],points[i][1],i);
		std::sort(front.begin(),front.end());
		
		return bestContributors(front,k,[](double con1, double con2){return con1 < con2;});
	}
	
	/// \brief Returns the index of the points with largest contribution.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the smallest contributor.
	/// \param [in] referencePointThe reference Point\f$\vec{r} \in \mathbb{R}^2\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$.
	template<typename Set, typename VectorType>
	std::vector<KeyValuePair<double,std::size_t> > largest( Set const& points, std::size_t k, VectorType const& referencePoint)const{
		SHARK_RUNTIME_CHECK(points.size() >= k, "There must be at least k points in the set");
		std::vector<Point> front;
		front.emplace_back(0,referencePoint[1],points.size()+1);//add reference point
		for(std::size_t i  = 0; i != points.size(); ++i)
			front.emplace_back(points[i][0],points[i][1],i);
		front.emplace_back(referencePoint[0],0,points.size()+1);//add reference point
		std::sort(front.begin()+1,front.end()-1);
		
		return bestContributors(front,k,[](double con1, double con2){return con1 > con2;});
	}
	
	/// \brief Returns the index of the points with largest contribution.
	///
	/// As no reference point is given, the edge points can not be computed and are not selected.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the smallest contributor.
	/// \param [in] k The number of points to select.
	template<typename Set>
	std::vector<KeyValuePair<double,std::size_t> > largest( Set const& points, std::size_t k)const{
		SHARK_RUNTIME_CHECK(points.size() >= k, "There must be at least k points in the set");
		std::vector<Point> front;
		for(std::size_t i  = 0; i != points.size(); ++i)
			front.emplace_back(points[i][0],points[i][1],i);
		std::sort(front.begin(),front.end());
		
		return bestContributors(front,k,[](double con1, double con2){return con1 > con2;});
	}

};

}
#endif
