//===========================================================================
/*!
*  \brief Test cases for nearest neighbor queries.
*
*  \author  T. Glasmachers
*  \date    2011
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
*  
*/
//===========================================================================

#define BOOST_TEST_MODULE Algorithms_NearestNeighbors
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <algorithm>

#include <shark/LinAlg/Base.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Trees/KDTree.h>
#include <shark/Models/Trees/LCTree.h>
#include <shark/Models/Trees/KHCTree.h>
#include <shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Core/Timer.h>


using namespace shark;


// helper class for std::sort
class Comparator
{
public:
	Comparator(std::vector<RealVector>& data, RealVector& test)
	: m_data(data)
	, m_test(test)
	{ }

	bool operator () (std::size_t index1, std::size_t index2) const
	{
		double d1 = distanceSqr(m_data[index1], m_test);
		double d2 = distanceSqr(m_data[index2], m_test);
		return (d1 < d2);
	}

protected:
	std::vector<RealVector>& m_data;
	RealVector& m_test;
};

const unsigned int TRAINING = 100000;
const unsigned int NEIGHBORS = 1000;

template<class Tree>
void testTree(
	Tree const& tree, char const* name, std::vector<RealVector> const& data, 
	RealVector const& test, std::vector<std::size_t> index,
	double time_reference
){
	// space for algorithm results
	std::vector<double> dist(NEIGHBORS);
	std::vector<RealVector const*> neighbor(NEIGHBORS);
	// query NEIGHBORS first neighbors of the test point
	double start = Timer::now();
	IterativeNNQuery<std::vector<RealVector> > query(&tree,data, test);
	for (std::size_t i=0; i != NEIGHBORS; i++){
		std::pair<double,std::size_t> ret = query.next();
		dist[i] = ret.first;
		neighbor[i] = &data[ret.second];
		std::cout<<dist[i]<<" "<<ret.second<<std::endl;
	}
	double time = Timer::now() - start;

	// check consistency with brute force algorithm
	/*for (std::size_t i=0; i<neighbors; i++)
	{
		if (i >= 5) { 
			boost_check_equal(&data[index[i]], neighbor[i]);
		}
		boost_check_small(distance(*neighbor[i], test) - dist[i], 1e-12);
	}

	 check consistency of ranking
	for (std::size_t i=1; i<neighbors; i++)
	{
		boost_check_le(dist[i-1], dist[i]);
	}*/

	// more than 40x faster on my machine
	BOOST_TEST_MESSAGE(name<<" time: " << time << " seconds, versus " << time_reference << " seconds for brute force search");
	std::cout << name<<" time: " << time << " seconds, versus " << time_reference << " seconds for brute force search" << std::endl;
}


//chck that the tree locates every of it's training points correctly
template<class Tree>
void testTreeStructure(
	Tree const& tree, std::vector<RealVector> const& data
){
	for (std::size_t i=0; i != data.size(); i++){
		BinaryTree<RealVector> const* node = &tree;
		while(! node->isLeaf()){
			if(node->isRight(data[i]))
				node = node->right();
			else
				node = node->left();
		}
		std::size_t index = node->index(0);
		if(i < 5)
			BOOST_CHECK_LE(index,5u);
		else
			BOOST_CHECK_EQUAL(index,i);

	}
}

// Generate 100.000 i.i.d. standard normally
// distributed samples in 3D, then query the
// 1.000 nearest neighbors of the origin.
BOOST_AUTO_TEST_CASE(IterativeNearestNeighborQueries)
{
	Rng::seed(42);
	// generate data set and test point
	std::vector<RealVector> data(100);
	for (std::size_t i=0; i<100; i++)
	{
		// multiple instances of the same point
		data[i].resize(1);
		data[i][0] = Rng::gauss();
		//data[i][1] = Rng::gauss();
	}

	UnlabeledData<RealVector> dataset = createDataFromRange(data);

	// test the KDTree
	KDTree<RealVector> kdtree(dataset);
	testTreeStructure(kdtree,data);
//	LCTree<RealVector> lctree(data);
//	LinearKernel<RealVector> kernel;
//	KHCTree<std::vector<RealVector> > khctree(data, &kernel);
//	testTree(kdtree,"KDTree",data,test,index,time_reference);
//	testTree(lctree,"LCTree",data,test,index,time_reference);
//	testTree(khctree,"KHCTree",data,test,index,time_reference);
}


// Generate 100.000 i.i.d. standard normally
// distributed samples in 3D, then query the
// 1.000 nearest neighbors of the origin.
//BOOST_AUTO_TEST_CASE(IterativeNearestNeighborQueries)
//{
//	double start;
//	Rng::seed(42);
//	// generate data set and test point
//	std::vector<RealVector> data(TRAINING);
//	for (std::size_t i=0; i<5; i++)
//	{
//		// multiple instances of the same point
//		data[i].resize(3);
//		data[i][0] = 0.0;
//		data[i][1] = 0.0;
//		data[i][2] = 0.0;
//	}
//	for (std::size_t i=0; i<TRAINING; i++)
//	{
//		// random data
//		data[i].resize(3);
//		data[i][0] = Rng::gauss();
//		data[i][1] = Rng::gauss();
//		data[i][2] = Rng::gauss();
//	}
//	RealVector test(3);
//	test[0] = 0.0;
//	test[1] = 0.0;
//	test[2] = 0.0;
//
//	// brute force sorting (for comparison)
//	std::vector<std::size_t> index(TRAINING);
//	start = Timer::now();
//	for (std::size_t i=0; i<TRAINING; i++) 
//		index[i] = i;
//	Comparator comparator(data, test);
//	std::sort(index.begin(), index.end(), comparator);
//	double time_reference = Timer::now() - start;
//
//	UnlabeledData<RealVector> dataset(data);
//
//	// test the KDTree
//	KDTree<RealVector> kdtree(dataset);
//	testTreeStructure(kdtree,data);
////	LCTree<RealVector> lctree(data);
////	LinearKernel<RealVector> kernel;
////	KHCTree<std::vector<RealVector> > khctree(data, &kernel);
////	testTree(kdtree,"KDTree",data,test,index,time_reference);
////	testTree(lctree,"LCTree",data,test,index,time_reference);
////	testTree(khctree,"KHCTree",data,test,index,time_reference);
//}
