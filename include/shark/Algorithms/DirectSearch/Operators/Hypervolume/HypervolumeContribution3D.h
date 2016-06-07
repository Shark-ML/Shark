/*!
 *
 * \author      O.Krause, T. Glasmachers
 * \date        2014-2016
 *
 *
 * \par Copyright 1995-2016 Shark Development Team
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUME_CONTRIBUTION_3D_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUME_CONTRIBUTION_3D_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/utility/KeyValuePair.h>

#include <algorithm>
#include <vector>
#include <deque>
#include <set>
#include <utility>

namespace shark {
/// \brief Finds the hypervolume contribution for points in 3DD
///
/// The algorithm sweeps ascending through the z-direction and keeps track of
/// of the current cut through the volume at a given z-value.
/// the cut is partitioned in boxes representing parts of the hypervolume
/// that is not hared with any other point. thus the sum of
/// the volume of all boxes belonging to a point is making
/// up its hypervolume.
///
/// The algorithm runs in O(n log(n)) time.
struct HypervolumeContribution3D {
private:
	struct Point{
		Point(){}

		Point(double f1, double f2, double f3, std::size_t index)
		: f1(f1)
		, f2(f2)
		, f3(f3)
		, index(index)
		{}
			
		bool operator<(Point const& rhs) const{//sort by x1 coordinate
			return f1 < rhs.f1;
		}
		
		template<class S>
		friend S& operator<<(S& s,Point const& p){
			s<<"("<<p.f1<<","<<p.f2<<","<<p.f3<<")";
			return s;
		}
		
		double f1;
		double f2;
		double f3;
		std::size_t index;
	};
	
	///\brief Box representing part of the hypervolume of a point
	///
	/// by convention a unclosed box has lower.f3=upper.f3 and when its
	/// upper boundary is found, it is set to the right value and its volume
	/// is computed.
	///
	/// Boxes are stored in boxlists sorted ascending by x-coordinate.
	struct Box{
		Point lower;
		Point upper;
		
		double volume()const{
			return (upper.f1-lower.f1)*(upper.f2-lower.f2)*(upper.f3-lower.f3);
		}
	};
	

	/// \brief Updates the volumes of the first nondominated neighbour of the new point to the right
	///
	/// The volume of the newly added point intersects with boxes of its neighbour in the front.
	/// Thus we remove all boxes intersecting with this one, compute their volume and add it to
	/// the total volume of right. The boxes are replaced by one new box representing the non-intersecting
	/// contribution that is still to be determined.
	double cutBoxesOnTheRight(std::deque<Box>& rightList, Point const& point, Point const& right)const{
		if(rightList.empty()) return 0;//nothing to do
		
		double addedContribution = 0;
		double xright = right.f1;
		//iterate through all partly covered boxes
		while(!rightList.empty()){
			Box& b = rightList.front();
			//if the box is not partially covered, we are done as by sorting all other boxes have smaller y-value
			if(b.upper.f2 <= point.f2)
				break;
				
			//add volume of box
			b.upper.f3 = point.f3;
			addedContribution += b.volume();
			xright = b.upper.f1;
			rightList.pop_front();
		}
		if(xright != right.f1){//if we removed a box
			//in the last step, we have removed boxes that were only partially
			//covered. replace them by a box that covers their area
			Box rightBox;
			rightBox.lower.f1 = right.f1;
			rightBox.lower.f2 = right.f2;
			rightBox.lower.f3 = point.f3;
			rightBox.upper.f1 = xright;
			rightBox.upper.f2 = point.f2;
			rightBox.upper.f3 = point.f3;
			//upper.f3 remains unspecified until the box is completed
			rightList.push_front(rightBox);
		}
		return addedContribution;
	}
	
	/// \brief Updates the volumes of the first neighbour of the new point to the left
	///
	/// The volume of the newly added point intersects with boxes of its neighbours in the front.
	/// On the left side, we can even completely dominate whole boxes which are removed.
	/// Otherwise, the boxes are intersected with the volume of the new point and shrunk to the remainder
	/// This method removes the volume of boxes removed that way.
	double cutBoxesOnTheLeft(std::deque<Box>& leftList, Point const& point)const{
		double addedContribution = 0;
		while(!leftList.empty()){
			Box& b= leftList.back();
			if(point.f1 < b.lower.f1 ){//is the box completely covered?
				b.upper.f3 = point.f3;
				addedContribution += b.volume();
				leftList.pop_back();
			}else if(point.f1 < b.upper.f1 ){//is the box partly covered?
				//Add contribution
				b.upper.f3 = point.f3;
				addedContribution += b.volume();
				//replace box by the new box that starts from the height of p
				b.upper.f1 = point.f1;
				b.lower.f3 = point.f3;
				break;
			}else{
				break;//uncovered
			}
		}
		return addedContribution;
	}
	
	std::vector<KeyValuePair<double,std::size_t> > allContributions(std::vector<Point> const& points)const{
		
		std::size_t n = points.size();
		//for every point we have a list of boxes that make up its contribution, L in the paper.
		//the list stores the boxes ordered by x-value + one additional (empty) list for the added corner points
		std::vector<std::deque<Box> > boxlists(n+1);
		//contributions are accumulated here for every point
		std::vector<KeyValuePair<double,std::size_t> > contributions(n+1,makeKeyValuePair(0.0,1));
		for(std::size_t i = 0; i != n; ++i){
			contributions[i].value = points[i].index;
		}
		
		//The tree stores values ordered by x-value and is our xy front.
		// even though we store 3D points, the third component is not relevant
		// thus the values are also ordered by y-component.
		std::multiset<Point> xyFront;
		//insert points indiating the reference frame, required for setting up the boxes.
		//The 0 stands for the reference point (0,0) in x-y coordinate
		//the -inf ensures that the point never becomes dominated
		double inf = std::numeric_limits<double>::max();//inf can be more costly to work with!
		xyFront.insert(Point(-inf,0,-inf,n));
		xyFront.insert(Point(0,-inf,-inf,n));
		
		//main loop
		for(std::size_t i = 0; i != n; ++i){
			Point const& point = points[i];
			
			//position of the point with next smaller x-value in the front
			auto left = xyFront.lower_bound(point);//gives larger or equal			
			--left;//first smaller element
			
			//check if the new point is dominated
			if(left->f2 < point.f2)
				continue;
			
			//find the indizes of all points dominated by the new point
			//and find the position of the next nondominated neighbour
			//with larger x-value
			std::vector<std::size_t> dominatedPoints;
			auto right= left; ++right;
			while((*right).f2 > point.f2){
				dominatedPoints.push_back(right->index);
				++right;
			}
			
			
			//erase all dominated points from the front
			xyFront.erase(std::next(left),right);
			
			//add point to the front
			xyFront.insert(Point(point.f1,point.f2,point.f3,i));
			//reorder dominated points so that the largest x-values are at the front
			std::reverse(dominatedPoints.begin(),dominatedPoints.end());
			
			
			//cut and remove neighbouring boxes
			contributions[left->index].key += cutBoxesOnTheLeft(boxlists[left->index],point);
			contributions[right->index].key += cutBoxesOnTheRight(boxlists[right->index],point,*right);
			
			//Remove dominated points with their boxes
			//and create new boxes for the new point
			double xright = right->f1;
			for(std::size_t domIndex:dominatedPoints){
				Point dominated = points[domIndex];
				auto& domList = boxlists[domIndex];
				for(Box& b:domList){
					b.upper.f3 = point.f3;
					contributions[domIndex].key += b.volume();
				}
				Box leftBox;
				leftBox.lower.f1 = dominated.f1;
				leftBox.lower.f2 = point.f2;
				leftBox.lower.f3 = point.f3;
				
				leftBox.upper.f1 = xright;
				leftBox.upper.f2 = dominated.f2;
				leftBox.upper.f3 = leftBox.lower.f3;
				//upper.f3 remains unspecified until the box is completed
				boxlists[i].push_front(leftBox);
				
				xright = dominated.f1;
			}
			//Add the new box created by this point
			Box newBox;
			newBox.lower.f1 = point.f1;
			newBox.lower.f2 = point.f2;
			newBox.lower.f3 = point.f3;
			newBox.upper.f1 = xright;
			newBox.upper.f2 = left->f2;
			newBox.upper.f3 = newBox.lower.f3;
			boxlists[i].push_front(newBox);
		}
		
		//go through the front and close all remaining boxes
		for(Point const& p: xyFront){
			std::size_t index = p.index;
			for(Box & b: boxlists[index]){
				b.upper.f3  = 0.0;
				contributions[index].key +=b.volume();
			}
		}
		
		//finally sort contributions descending and return it
		contributions.pop_back();//remove the superfluous last element
		std::sort(contributions.begin(),contributions.end());
		
		return contributions;
	}
	
public:
	/// \brief Returns the index of the points with smallest contribution as well as their contribution.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the smallest contributor.
	/// \param [in] k The number of points to select.
	/// \param [in] referencePointThe reference Point\f$\vec{r} \in \mathbb{R}^2\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$.
	template<class Set, typename VectorType>
	std::vector<KeyValuePair<double,std::size_t> > smallest(Set const& points, std::size_t k, VectorType const& ref)const{
		std::vector<Point> front;
		for(std::size_t i = 0; i != points.size(); ++i){
			front.emplace_back(points[i](0)-ref(0),points[i](1)-ref(1),points[i](2)-ref(2),i);
		}
		std::sort(
			front.begin(),front.end(),
			[](Point const& lhs, Point const& rhs){
				return lhs.f3 < rhs.f3;
			}
		);
		
		auto result = allContributions(front);
		result.erase(result.begin()+k,result.end());
		return result;
	}
	
	/// \brief Returns the index of the points with smallest contribution as well as their contribution.
	///
	/// As no reference point is given, the extremum points can not be computed and are never selected.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the smallest contributor.
	/// \param [in] k The number of points to select.
	template<class Set>
	std::vector<KeyValuePair<double,std::size_t> > smallest(Set const& points, std::size_t k)const{
		//reference point computation, and obtain the indizes of the extremum elements
		std::size_t minIndex[]={0,0,0};
		double minVal[]={points[0](0),points[0](1),points[0](2)};
		double ref[]={points[0](0),points[0](1),points[0](2)};
		for(std::size_t i = 0; i != points.size(); ++i){
			for(std::size_t j = 0; j != 3; ++j){
				if(points[i](j)< minVal[j]){
					minVal[j] = points[i](j);
					minIndex[j]=i;
				}
				ref[j] = std::max(ref[j],points[i](j));
			}
		}
		
		
		std::vector<Point> front;
		for(std::size_t i = 0; i != points.size(); ++i){
			front.emplace_back(points[i](0)-ref[0],points[i](1)-ref[1],points[i](2)-ref[2],i);
		}
		std::sort(
			front.begin(),front.end(),
			[](Point const& lhs, Point const& rhs){
				return lhs.f3 < rhs.f3;
			}
		);
		
		auto result = allContributions(front);
		for(std::size_t j = 0; j != 3; ++j){
			auto pos = std::find_if(
				result.begin(),result.end(),
				[&](KeyValuePair<double,std::size_t> const& p){
					return p.value == minIndex[j];
				}
			);
			if(pos != result.end())
				result.erase(pos);
		}
		result.erase(result.begin()+k,result.end());
		return result;
	}

	
	/// \brief Returns the index of the points with largest contribution as well as their contribution.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the largest contributor.
	/// \param [in] referencePointThe reference Point\f$\vec{r} \in \mathbb{R}^2\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$.
	template<class Set, typename VectorType>
	std::vector<KeyValuePair<double,std::size_t> > largest(Set const& points, std::size_t k, VectorType const& ref)const{
		std::vector<Point> front;
		for(std::size_t i = 0; i != points.size(); ++i){
			front.emplace_back(points[i](0)-ref(0),points[i](1)-ref(1),points[i](2)-ref(2),i);
		}
		std::sort(
			front.begin(),front.end(),
			[](Point const& lhs, Point const& rhs){
				return lhs.f3 < rhs.f3;
			}
		);
		
		auto result = allContributions(front);
		result.erase(result.begin(),result.end()-k);
		std::reverse(result.begin(),result.end());
		return result;
	}
	
	
	/// \brief Returns the index of the points with largest contribution as well as their contribution.
	///
	/// As no reference point is given, the extremum points can not be computed and are never selected.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the largest contributor.
	/// \param [in] k The number of points to select.
	template<class Set>
	std::vector<KeyValuePair<double,std::size_t> > largest(Set const& points, std::size_t k)const{
		//reference point computation, and obtain the indizes of the extremum elements
		std::size_t minIndex[]={0,0,0};
		double minVal[]={points[0](0),points[0](1),points[0](2)};
		double ref[]={points[0](0),points[0](1),points[0](2)};
		for(std::size_t i = 0; i != points.size(); ++i){
			for(std::size_t j = 0; j != 3; ++j){
				if(points[i](j)< minVal[j]){
					minVal[j] = points[i](j);
					minIndex[j]=i;
				}
				ref[j] = std::max(ref[j],points[i](j));
			}
		}
		
		
		std::vector<Point> front;
		for(std::size_t i = 0; i != points.size(); ++i){
			front.emplace_back(points[i](0)-ref[0],points[i](1)-ref[1],points[i](2)-ref[2],i);
		}
		std::sort(
			front.begin(),front.end(),
			[](Point const& lhs, Point const& rhs){
				return lhs.f3 < rhs.f3;
			}
		);
		
		auto result = allContributions(front);
		for(std::size_t j = 0; j != 3; ++j){
			auto pos = std::find_if(
				result.begin(),result.end(),
				[&](KeyValuePair<double,std::size_t> const& p){
					return p.value == minIndex[j];
				}
			);
			if(pos != result.end())
				result.erase(pos);
		}
		if(result.size() > k)
			result.erase(result.begin(),result.end()-k);
		std::reverse(result.begin(),result.end());
		return result;
	}
};

}
#endif