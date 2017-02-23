/*!
 *
 * \author      O.Krause
 * \date        2016
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMESUBSETSELECTION_2D_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMESUBSETSELECTION_2D_H

#include <shark/LinAlg/Base.h>

#include <algorithm>
#include <vector>
#include <deque>

namespace shark {
/// \brief Implementation of the exact hypervolume subset selection algorithm in 2 dimensions.
///
/// This algorithm solves the problem of selecting a subset of points with largest hypervolume in 2D.
/// The algorithm has complexity n (k+log(n)) 
///
/// While this algorithm accepts fronts with dominated points in it, the caller has to ensure
/// that after domination checks there are at least as many points left as there are to select. The
/// Algorithm will throw an exception otherwise.
///
/// This can easily be ensured by removing the nondominated points prior to calling this function.
///
/// The algorithm is described in:
/// Bringmann, Karl, Tobias Friedrich, and Patrick Klitzke. 
/// "Two-dimensional subset selection for hypervolume and epsilon-indicator."
/// Proceedings of the 2014 conference on Genetic and evolutionary computation. 
/// ACM, 2014.	
/// (although it is not very helpful)
struct HypervolumeSubsetSelection2D {
private:
	
	struct Point{
		Point(){}

		Point(double f1, double f2, std::size_t index)
		: f1(f1)
		, f2(f2)
		, index(index)
		, selected(false)
		{}
			
		bool operator<(Point const& rhs) const{//for lexicographic sorting
			if (f1 < rhs.f1) return true;
			if (f1 > rhs.f1) return false;
			return (f2 < rhs.f1);
		}
		
		double f1;
		double f2;
		std::size_t index;
		bool selected;
	};
	
	///\brief Linear function a*x+b where a is stored in first and b is stored in section.
	///
	/// The linear function also stores an index to uniquely identify it.
	///
	/// Linear functions are used in the algorithm to represent the 
	/// volume of a given set of points under the change of reference point.
	/// more formally, let H^l_i be the volume of a set of points of size l with largest
	/// x-value at the point (x_i,y_i) and reference point x_i(thus H^l_i can only use points
	/// 1,...,i). 
	/// Then for x>x_i we have
	/// f_i^l(x) = H_i^l+ y_i(x_i-x)=-x*y_i+y_i*x_i+H = a*x+b. 
	/// Later the algorithm will use an upper envelope over a set of those functions
	/// to decide which points to add to the sets until the size of the sets is k.
	///
	/// for this application the stored index is the same as index i of the point stated above.
	struct LinearFunction{
	
		double a;
		double b;
		std::size_t index;
		
		LinearFunction(double a, double b, std::size_t index = 0):a(a), b(b), index(index){}
		LinearFunction(){}
		
		double eval(double x)const{
			return a*x + b;
		}
	};

	/// \brief Returns the intersection of two linear functions
	double Intersection(LinearFunction f1, LinearFunction f2)const{
		return (f2.b - f1.b) / (f1.a - f2.a);
	}
	
	
	/// \brief  Calculates for each given x the maximum among the functions f, i.e. the upper envelope of f.
	/// 
	/// Algorithm 2 in the paper. Complexity O(n)
	/// given a set of functions f_1...f_n, ordered by slope such that f_1.a < f_2.a<...<f_n.a and points with x-coordinate x_1<...<x_n
	/// computes h_i = max_{1 <= j <= i} f_j(x_i) for i=1,...,n as well as the index of the function leading to the value h_i
	std::pair<std::vector<double>,std::vector<std::size_t> > upperEnvelope(
		std::vector<LinearFunction>const& functions,
		std::vector<Point> const& points
	)const{
		SHARK_ASSERT(functions.size() == points.size());
		std::size_t n = points.size();
		std::vector<double> h(n);
		std::vector<std::size_t> chosen(n);
		std::deque<LinearFunction> s;

		// This is the original algorithm 2 as in the paper. Even if the paper looks at maximum
		// hypervolume where domination is given when one point has LARGER function
		// values as the other, In section 3.2 they transform the problem to a problem
		// where domination is given by SMALLER function values and then accordingly
		// give the algorithm for this type. They just give the transformation but do not say
		// what the transformation does so it is not clear until you implement it.
		//
		// This is a super confusing part of the paper, please kids, do not be like Bringmann et al.
		// Keep it simple, stupid. Sometimes an additional index does not hurt.
		//
		//the algorithm works by inserting functions f_1 to f_i one-by-one, figuring out which functions
		// are dominated (not being part of the upper envelope) and removing all functions which for
		// function values x_i,x_i+1,... are already smaller than one of the other function.
		// using the ordering relations given the set s contains the function ordered by (current) function value.
		// so after iteration i we can just extract the largest function value for x_i by looking at the first element of s.
		for (std::size_t i = 0; i != n; ++i) {

			// remove dominated functions.
			// as we push back into s,
			// at the end of s are the functions with largest slope.
			// therefore if we have the last two elements as s_-1 and s_-2 and the new
			// function f, knowing that the intersection of s_-1 and f is smaller than the intersection
			// of s_-1 and s_-2 means that s_-1 is dominated  by s_-2 and f and thus can be removed.
			while (s.size() > 1 ) {
				
				double d1 = Intersection(functions[i], s.end()[-1]);
				double d2 = Intersection(s.end()[-2], s.end()[-1]);

				if (d1 <= d2 || std::abs(d1-d2) < 1.e-10) {//check for numeric stability
					s.pop_back();
				} else {
					break;
				}
			}
			//include the new function and store its index.
			s.push_back(functions[i]);
			s.back().index = i;
			// at the beginning of s are the functions with smallest slope
			// if the first function in s has a smaller function value for the current 
			// x_i than the second function,
			// we can safely remove it as it can not be part of the envelope any more
			// (We are only looking at function values >=x from now on and thus the larger slope domintes)
			while (s.size() > 1) {
				double d1 = s[0].eval(points[i].f1);
				double d2 = s[1].eval(points[i].f1);

				if (d1 < d2 || std::abs(d1-d2) < 1.e-10) {
					s.pop_front();
				} else {
					break;
				}
			}
			//assign maximum
			//the functions in s are ordered by function value  
			// the function with the largest function value is currently at the front
			h[i] = s[0].eval(points[i].f1);
			chosen[i] = s[0].index;
		}
		return std::make_pair(std::move(h),std::move(chosen));
	}
	
	
	/// Fast calculation O(n*k) for the hypervolume selection problem. 
	/// for the selected points, it sets selected=true.
	void hypSSP(std::vector<Point>& front,std::size_t k)const{
		SHARK_RUNTIME_CHECK( k > 0, "k must be non-zero");
		SHARK_RUNTIME_CHECK( k <= front.size(), "The front must have at least k nondominated points");
		
		std::size_t n = front.size();
		std::vector<LinearFunction> functions(n);
		
		std::vector<std::vector<std::size_t> > chosen;
		std::vector<double>  h(n,0.0);
		for(std::size_t j=0; j != k-1; ++j) {//compute until k-1 elements are chosen
			for(std::size_t i=0; i != n; ++i ) {
				functions[i] = LinearFunction( -front[i].f2, front[i].f1* front[i].f2 + h[i]);
			}
			auto result = upperEnvelope(functions, front);
			h = result.first;
			chosen.push_back(result.second);
		}
		
		//choose the last element by simply iterating over all elements
		std::size_t currentIndex = 0;
		double res = -1;
		for(std::size_t i=0; i != n; ++i ) {
			LinearFunction f(-front[i].f2, front[i].f1*front[i].f2 + h[i]);
			if(f.eval(0)  > res) {
				res = f.eval(0);
				currentIndex = i;
			}
		}
		front[currentIndex].selected = true;
		//iterate backwards to reconstruct chosen indizes
		for(auto pos = chosen.rbegin(); pos != chosen.rend(); ++pos){
			currentIndex = (*pos)[currentIndex];
			front[currentIndex].selected = true;
		}
	}
	
	template<typename Set>
	std::vector<Point> createFront(Set const& points, double refX, double refY)const{
		//copy points using the new reference frame with refPoint at (0,0). also store original index for later
		std::vector<Point> front;
		for(std::size_t i = 0; i != points.size(); ++i){
			front.emplace_back(points[i](0) - refX, points[i](1) - refY,i);
		}
		std::sort(front.begin(),front.end());//sort lexicographically
		//erase dominated points
		auto newEnd = std::unique(front.begin(),front.end(),[](Point const& x, Point const& y){
			return y.f2 >= x.f2;//by lexikographic sort we already have y.f1 >= x.f1
		});
		front.erase(newEnd,front.end());
		return front;
	}
public:
	/// \brief Executes the algorithm.
	/// While this algorithm in general accepts fronts with dominated points in it, the caller has to ensure
	/// that after domination checks there are at least as many points left as there are to select. The
	/// Algorithm will throw an exception otherwise.
	///
	/// This can easily be ensured by removing the nondominated points prior to calling this function.
	/// \param [in] points The set \f$S\f$ of points to select
	/// \param [out] selected set of the same size as the set of points indicating whether the point is selected (1) or not (0)
	/// \param [in] k number of points to select. Must be lrger than 0
	/// \param [in] refPoint The reference point \f$\vec{r} \in \mathbb{R}^2\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$. .
	template<typename Set, typename SelectedSet, typename VectorType >
	void operator()( Set const& points, SelectedSet& selected, std::size_t k, VectorType const& refPoint){
		SIZE_CHECK(points.size() == selected.size());
		SHARK_RUNTIME_CHECK(k > 0, "k must be >0");
		SHARK_RUNTIME_CHECK( k <= points.size(), "the number of points must be larger than k");
		SIZE_CHECK( points.begin()->size() == 2 );
		SIZE_CHECK( refPoint.size() == 2 );
		
		for(auto&& s: selected)
			s = false;
		
		std::vector<Point> front = createFront(points, refPoint(0), refPoint(1));
		
		//find the optimal set in the front. afterwards selected points have selected=true
		hypSSP(front,k);
		//mark selected points in the original front
		for(Point const& point: front){
			if(point.selected){
				selected[point.index] = true;
			}
		}
	}
	
	/// \brief Executes the algorithm.
	///
	/// This version does not use a reference point. instead the extreme points are always kept which  implicitely defines a reference point
	/// that after domination checks there are at least as many points left as there are to select. The
	/// Algorithm will throw an exception otherwise.
	///
	/// This can easily be ensured by removing the nondominated points prior to calling this function.
	///
	/// \param [in] points The set \f$S\f$ of points to select
	/// \param [out] selected set of the same size as the set of points indicating whether the point is selected (1) or not (0)
	/// \param [in] k number of points to select, must be larger or equal 2
	template<typename Set, typename SelectedSet>
	void operator()( Set const& points, SelectedSet& selected, std::size_t k){
		SIZE_CHECK(points.size() == selected.size());
		SHARK_RUNTIME_CHECK( k >= 2, "k must be larger or equal 2");
		SHARK_RUNTIME_CHECK( k <= points.size(), "the number of points mjust be larger than k");
		SIZE_CHECK(points.size() == selected.size());
		SIZE_CHECK( points.begin()->size() == 2 );
		
		for(auto&& s: selected)
			s = false;
		
		//create front using "fake ref"
		std::vector<Point> front = createFront(points, 0,0);
			
		//get reference value from extremal points
		double refX= front.back().f1;
		double refY= front.front().f2;
			
		for(auto&& point: front){
			point.f1 -= refX;
			point.f2 -= refY;
		}
		
		//mark the extrema as chosen and remove them from the front
		selected[front.front().index] = true;
		selected[front.back().index] = true;
		front.pop_back();
		front.erase(front.begin(),front.begin()+1);
		if(k == 2) return;
		
		//find the optimal set in the front. afterwards selected points have selected=true
		hypSSP(front,k-2);
		//mark selected points in the original front
		for(Point const& point: front){
			if(point.selected){
				selected[point.index] = true;
			}
		}
	}
};

}
#endif
