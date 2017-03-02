///
/// \brief       Implements a divide-and-conquer non-dominated sorting algorithm.
/// 
/// \author      T. Glasmachers
/// \date        2016
///
///
/// \par Copyright 1995-2017 Shark Development Team
/// 
/// <BR><HR>
/// This file is part of Shark.
/// <http://shark-ml.org/>
/// 
/// Shark is free software: you can redistribute it and/or modify
/// it under the terms of the GNU Lesser General Public License as published 
/// by the Free Software Foundation, either version 3 of the License, or
/// (at your option) any later version.
/// 
/// Shark is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU Lesser General Public License for more details.
/// 
/// You should have received a copy of the GNU Lesser General Public License
/// along with Shark.  If not, see <http://www.gnu.org/licenses/>.
///

#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_DOMINATION_DCNONDOMINATEDSORT_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_DOMINATION_DCNONDOMINATEDSORT_H

#include <shark/Algorithms/DirectSearch/Operators/Domination/ParetoDominance.h>
#include <shark/LinAlg/Base.h>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <utility>
#include <algorithm>
#include <functional>


namespace shark {


///
/// \brief Divide-and-conquer algorithm for non-dominated sorting.
///
/// Assembles subsets/fronts of mutually non-dominated individuals.
/// Afterwards every individual is assigned a rank by pop[i].rank() = frontIndex.
/// The front of non-dominated points has the value 1.
/// 
/// The algorithm is described in:
/// Generalizing the Improved Run-Time Complexity Algorithm for Non-Dominated Sorting.
/// Felix-Antoine Fortin, Simon Grenier, and Marc Parizeau,
/// Proceedings Genetic and Evolutionary Computation Conference (GECCO), pp. 615-622, 2013.
///
/// The algorithm is based on an earlier publication by Jensen, 2003. Its runtime
/// complexity for n points and m objectives is claimed to be as low as
/// \f$ \mathcal{O}(n \log(n)^{m-1}) \f$, which is often better than the complexity
/// \f$ \mathcal{O}(n^2 m) \f$ of the of "fast non-dominated sorting" algorithm,
/// in particular for small m.
///
/// However, the implementation has a complexity of \f$ \mathcal{O}(n \log(n)^{m-2}c) \f$,
/// where c is the number of distinct non-dominance ranks. If can hence be as bad as
/// \f$ \mathcal{O}(n^2 \log(n)^{m-2}) \f$, but in practice it is usually fast.
/// The same applies to the reference implementation in the DEAP library (at the time
/// of writing, March 2016).
///
class BaseDCNonDominatedSort
{
private:
	struct Point
	{
		Point(): frt(1){}

		Point(RealVector const& objvec)
		: obj(objvec.raw_storage().values)
		, frt(1)
		, m_size(objvec.size())
		{}
		
		bool operator<(Point const& rhs) const{//for lexicographic sorting
			for (std::size_t i=0; i<m_size; i++)
			{
				if (obj[i] < rhs.obj[i]) return true;
				if (obj[i] > rhs.obj[i]) return false;
			}
			return false;
		}
		bool operator==(Point const& rhs) const{//for std::unique
			for (std::size_t i=0; i<m_size; i++)
			{
				if (obj[i] != rhs.obj[i]) return false;
			}
			return true;
		}

		
		double const* obj;
		unsigned int frt;    // front index (1-based)
		std::size_t m_size;
	};

	typedef std::vector<Point*> ContainerType;
	typedef std::map<unsigned int, double> MapType;
	typedef std::set<std::size_t, std::greater<std::size_t> > SetType;
	typedef std::map<double, SetType > InverseMapType;

public:
	/// \brief Executes the non-dominated sorting algorithm.
	///
	/// Afterwards every individual is assigned a rank by pop[i].rank() = frontNumber.
	/// The front of dominating points has the value 1.
	///
	template<typename PointRange, typename RankRange>
	void operator () (PointRange const& pointRange, RankRange& rankRange)
	{
		SIZE_CHECK(pointRange.size() == rankRange.size());
		std::size_t m = pointRange[0].size();
		// prepare set of unique objective vectors
		std::vector<Point> points(pointRange.size());
		std::transform(pointRange.begin(),pointRange.end(),points.begin(),[](RealVector const& point){
			return Point(point);
		});
		std::sort(points.begin(),points.end());
		points.erase(std::unique(points.begin(),points.end()),points.end());
		ContainerType S(points.size());
		for (std::size_t i = 0; i != points.size(); ++i)
		{
			S[i] = &points[i];
		}

		// call recursive algorithm
		ndHelperA(S,m);

		// assign ranks to individuals
		for (std::size_t i = 0; i != pointRange.size(); ++i)
		{
			auto it = std::lower_bound(points.begin(), points.end(), Point(pointRange[i]));
			SHARK_ASSERT(it != points.end());
			rankRange[i] = it->frt;
		}
	}

private:
	// Dominance relation restricted to the first k components of the objective vector.
	static DominanceRelation dominance(Point const* lhs, Point const* rhs, std::size_t k)
	{
		return shark::dominance(blas::adapt_vector(k,lhs->obj),blas::adapt_vector(k,rhs->obj));
	}

	// figure 2 in the paper
	//
	// This function performs non-dominated sorting of S according to
	// the first k objectives, while taking previously set front index
	// values into account (which stem from dominance by points external
	// to S).
	void ndHelperA(ContainerType& S, std::size_t k)
	{
		if (S.size() < 2) return;
		if (S.size() == 2)
		{
			if (dominance(S[0], S[1], k) == LHS_DOMINATES_RHS)
			{
				S[1]->frt = std::max(S[1]->frt, S[0]->frt + 1);
			}
			return;
		}
		if (k == 2)
		{
			sweepA(S);
			return;
		}

		// check condition |\{s_k | s \in S\}| = 1
		bool k_equal = true;
		for (std::size_t i=1; i<S.size(); i++)
		{
			if (S[0]->obj[k-1] != S[i]->obj[k-1])
			{
				k_equal = false;
				break;
			}
		}

		if (k_equal)
		{
			ndHelperA(S, k-1);
			return;
		}
		else
		{
			ContainerType L, H;
			splitA(S, k, L, H);
			ndHelperA(L, k);
			ndHelperB(L, H, k-1);
			ndHelperA(H, k);
			return;
		}
	}

	// figure 3 in the paper
	//
	// Same functionality as ndHelperA, but a specialized sweeping
	// algorithm for two objectives.
	//
	// Note: the paper (and also the original paper by Jensen) states
	// that this is an O(n log(n)) operation. However, the reference
	// implementation in the DEAP library implements it as an O(n^2)
	// operation, or more exactly, as an O(n c) operation, where c is
	// the number of fronts. It seems that this is because Jensen's
	// O(n log(c)) algorithm works only if all initial FRT values
	// coincide, which holds in the bi-objective case, but not in the
	// more general case needed here.
	void sweepA(ContainerType& S)
	{
		MapType T;
		T[S[0]->frt] = S[0]->obj[1];
		for (std::size_t i=1; i<S.size(); i++)
		{
			// O(T.size()) operation, should be logarithmic...?
			unsigned int r = 0;
			for (auto p : T)
			{
				if (p.second <= S[i]->obj[1]) r = std::max(r, p.first);
			}

			if (r > 0) S[i]->frt = std::max(S[i]->frt, r + 1);

			T[S[i]->frt] = S[i]->obj[1];
		}
	}

	// median of the k-th objective over S
	double median(ContainerType const& S, std::size_t k)
	{
		std::vector<double> value(S.size());
		for (std::size_t i=0; i<S.size(); i++) value[i] = S[i]->obj[k];
		std::nth_element(value.begin(), value.begin() + value.size() / 2, value.end());
		double ret = value[value.size() / 2];
		if (S.size() & 1) return ret;
		ret += *std::max_element(value.begin(), value.begin() + value.size() / 2);
		return ret / 2.0;
	}

	// figure 5 in the paper
	//
	// Split the set S according to the median in objective k
	// (one-based index) as balanced as possible into subsets
	// L and H. The subsets remain lexicographically sorted.
	void splitA(ContainerType const& S, std::size_t k, ContainerType& L, ContainerType& H)
	{
		k--;   // index is zero-based
		SHARK_ASSERT(L.empty());
		SHARK_ASSERT(H.empty());
		double med = median(S, k);
		ContainerType La, Lb, Ha, Hb;
		for (std::size_t i=0; i<S.size(); i++)
		{
			double v = S[i]->obj[k];
			if (v < med)
			{
				La.push_back(S[i]);
				Lb.push_back(S[i]);
			}
			else if (v > med)
			{
				Ha.push_back(S[i]);
				Hb.push_back(S[i]);
			}
			else
			{
				La.push_back(S[i]);
				Hb.push_back(S[i]);
			}
		}
		if (Lb.size() < Ha.size())
		{
			using namespace std;
			swap(L, La);
			swap(H, Ha);
		}
		else
		{
			using namespace std;
			swap(L, Lb);
			swap(H, Hb);
		}
	}

	// figure 7 in the paper
	//
	// preconditions:
	// * sets L and H are lexicographically orderes
	// * all points in L are better than those in H according to objective k
	// * the front indices for L are final
	// * the front indices of H are all 1
	// *
	// postconditions:
	// * each front index of h \in H is assigned the max of the front indices of l \in L dominating h, plus 1
	void ndHelperB(ContainerType const& L, ContainerType& H, std::size_t k)
	{
		if (L.empty() || H.empty()) return;
		if (L.size() == 1 || H.size() == 1)
		{
			for (std::size_t j=0; j<H.size(); j++)
			{
				for (std::size_t i=0; i<L.size(); i++)
				{
					DominanceRelation rel = dominance(L[i], H[j], k);
					if (rel == LHS_DOMINATES_RHS || rel == EQUIVALENT)
					{
						H[j]->frt = std::max(H[j]->frt, L[i]->frt + 1);
					}
				}
			}
			return;
		}
		if (k == 2)
		{
			sweepB(L, H);
			return;
		}
		double minLk = L[0]->obj[k-1];
		double maxLk = L[0]->obj[k-1];
		for (std::size_t i=1; i<L.size(); i++)
		{
			double v = L[i]->obj[k-1];
			minLk = std::min(minLk, v);
			maxLk = std::max(maxLk, v);
		}
		double minHk = H[0]->obj[k-1];
		double maxHk = H[0]->obj[k-1];
		for (std::size_t i=1; i<H.size(); i++)
		{
			double v = H[i]->obj[k-1];
			minHk = std::min(minHk, v);
			maxHk = std::max(maxHk, v);
		}
		if (maxLk <= minHk)
		{
			ndHelperB(L, H, k-1);
			return;
		}
		if (minLk <= maxHk)
		{
			ContainerType L1, L2, H1, H2;
			splitB(L, H, k, L1, L2, H1, H2);
			ndHelperB(L1, H1, k);
			ndHelperB(L1, H2, k - 1);
			ndHelperB(L2, H2, k);
			return;
		}
	}

	// figure 8 in the paper
	//
	// Same as ndHelperB, but a specialized sweeping algorithm for two objectives.
	void sweepB(ContainerType const& L, ContainerType& H)
	{
		MapType T;
		std::size_t i = 0;
		for (std::size_t j=0; j<H.size(); j++)
		{
			while (i < L.size())
			{
				if (L[i]->obj[0] > H[j]->obj[0]) break;
				if (L[i]->obj[0] == H[j]->obj[0] && L[i]->obj[1] > H[j]->obj[1]) break;
				auto it = T.find(L[i]->frt);
				if (it == T.end() || L[i]->obj[1] < it->second)
				{
					T[L[i]->frt] = L[i]->obj[1];
				}
				i++;
			}

			// O(T.size()) operation, should be logarithmic...
			unsigned int r = 0;
			for (auto p : T)
			{
				if (p.second <= H[j]->obj[1]) r = std::max(r, p.first);
			}

			if (r > 0)   // U \not= \{\}
			{
				H[j]->frt = std::max(H[j]->frt, r + 1);
			}
		}
	}

	// figure 9 in the paper
	//
	// This function splits the sets L and H into subsets L1, L2 and H1, H2,
	// respectively, using a common split point in objective k. The split
	// point is optimized to balance the subsets. The subsets remain
	// lexicographically sorted.
	void splitB(ContainerType const& L, ContainerType const& H, std::size_t k,
			ContainerType& L1, ContainerType& L2, ContainerType& H1, ContainerType& H2)
	{
		k--;   // index is zero-based
		double pivot = median((L.size() > H.size()) ? L : H, k);

		ContainerType L1a, L1b, L2a, L2b;
		for (std::size_t i=0; i<L.size(); i++)
		{
			double v = L[i]->obj[k];
			if (v < pivot)
			{
				L1a.push_back(L[i]);
				L1b.push_back(L[i]);
			}
			else if (v > pivot)
			{
				L2a.push_back(L[i]);
				L2b.push_back(L[i]);
			}
			else
			{
				L1a.push_back(L[i]);
				L2b.push_back(L[i]);
			}
		}

		ContainerType H1a, H1b, H2a, H2b;
		for (std::size_t i=0; i<H.size(); i++)
		{
			double v = H[i]->obj[k];
			if (v < pivot)
			{
				H1a.push_back(H[i]);
				H1b.push_back(H[i]);
			}
			else if (v > pivot)
			{
				H2a.push_back(H[i]);
				H2b.push_back(H[i]);
			}
			else
			{
				H1a.push_back(H[i]);
				H2b.push_back(H[i]);
			}
		}

		if (L1b.size() + H1b.size() <= L2a.size() + H2a.size())
		{
			using namespace std;
			swap(L1, L1a);
			swap(L2, L2a);
			swap(H1, H1a);
			swap(H2, H2a);
		}
		else
		{
			using namespace std;
			swap(L1, L1b);
			swap(L2, L2b);
			swap(H1, H1b);
			swap(H2, H2b);
		}
	}
};


template<class PointRange, class RankRange>
void dcNonDominatedSort(PointRange const& points, RankRange& ranks) {
	BaseDCNonDominatedSort sorter;
	sorter(points,ranks);
}

}  // namespace shark
#endif
