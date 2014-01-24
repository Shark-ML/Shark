//===========================================================================
/*!
 * 
 * \file        HierarchicalClustering.tpp
 *
 * \brief       Example program for hierarchical clustering.
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2011
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

#include <shark/Models/Trees/LCTree.h>
#include <shark/Models/Clustering/HierarchicalClustering.h>
#include <shark/Models/Clustering/HardClusteringModel.h>


using namespace shark;


int main(int argc, char** argv)
{
	// create artificial data
	std::size_t trainingSize = 20;
	std::size_t testSize = 30;
	std::vector<RealVector> tr_d(trainingSize, RealVector(1, 0.0));
	std::vector<RealVector> te_d(testSize, RealVector(1, 0.0));
	for (std::size_t i=0; i<trainingSize; i++) 
		tr_d[i](0) = 100.0 * (i + 0.5) / (double)trainingSize;
	for (std::size_t i=0; i<testSize; i++) 
		te_d[i](0) = Rng::uni(0.0, 100.0);
		
	UnlabeledData<RealVector> training = createDataFromRange(tr_d);
	UnlabeledData<RealVector> test = createDataFromRange(te_d);

	// construct a hierarchical clustering with at most 3 points per cluster
	LCTree<RealVector> tree(training, TreeConstruction(0, 3));
	HierarchicalClustering<RealVector> clustering(&tree);
	HardClusteringModel<RealVector> model(&clustering);

	// output statistics
	std::cout << "number of tree nodes: " << tree.nodes() << std::endl;
	std::cout << "number of clusters: " << clustering.numberOfClusters() << std::endl;

	// output cluster assignments
	std::cout << "\ntraining data:\n";
	for (std::size_t i = 0; i != trainingSize; i++){
		unsigned int cluster = model(training.element(i));
		std::cout << "   point " << training.element(i)(0) << "  -->  cluster " << cluster << std::endl;
	}
	std::cout << "\ntest data:\n";
	for (std::size_t i=0; i<testSize; i++){
		unsigned int cluster = model(test.element(i));
		std::cout << "   point " << test.element(i)(0) << "  -->  cluster " << cluster << std::endl;
	}
}
