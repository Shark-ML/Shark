//===========================================================================
/*!
 * 
 *
 * \brief       k-means Clustering Tutorial Sample Code, requires the data
 * set faithful.csv
 * 
 * 
 *
 * \author      C. Igel
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

//###begin<includes>
#include <shark/Data/Csv.h> //load the csv file
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h> //normalize

#include <shark/Algorithms/KMeans.h> //k-means algorithm
#include <shark/Models/Clustering/HardClusteringModel.h>//model performing hard clustering of points
//###end<includes>

#include <iostream>

using namespace shark;
using namespace std;

int main(int argc, char **argv) {
	if(argc < 2) {
		cerr << "usage: " << argv[0] << " (filename)" << endl;
		exit(EXIT_FAILURE);
	}
	// read data
	UnlabeledData<RealVector> data;
	try {
	//###begin<import>
		importCSV(data, argv[1], ' ');
	//###end<import>
	} 
	catch (...) {
		cerr << "unable to read data from file " <<  argv[1] << endl;
		exit(EXIT_FAILURE);
	}
	std::size_t elements = data.numberOfElements();

	// write statistics of input data
	cout << "number of data points: " << elements << " dimensions: " << dataDimension(data) << endl;

	// normalize data
	//###begin<normalization>
	Normalizer<> normalizer;
	NormalizeComponentsUnitVariance<> normalizingTrainer(true);//zero mean
	normalizingTrainer.train(normalizer, data);
	data = normalizer(data);
	//###end<normalization>

	// compute centroids using k-means clustering
	//###begin<clustering>
	Centroids centroids;
	size_t iterations = kMeans(data, 2, centroids);
	//###end<clustering>
	// report number of iterations by the clustering algorithm
	cout << "iterations: " << iterations << endl;

	// write cluster centers/centroids
	//###begin<print_cluster_means>
	Data<RealVector> const& c = centroids.centroids();
	cout<<c<<std::endl;
	//###end<print_cluster_means>

	// cluster data
	//###begin<assign_cluster>
	HardClusteringModel<RealVector> model(&centroids);
	Data<unsigned> clusters = model(data);
	//###end<assign_cluster>
	
	// write results to files
	ofstream c1("cl1.csv");
	ofstream c2("cl2.csv");
	ofstream cc("clc.csv");
	//###begin<print_cluster_assignment>
	for(std::size_t i=0; i != elements; i++) {
		if(clusters.element(i)) 
			c1 << data.element(i)(0) << " " << data.element(i)(1) << endl;
		else 
			c2 << data.element(i)(0) << " " << data.element(i)(1) << endl;
	}
	//###end<print_cluster_assignment>
	cc << c.element(0)(0) << " " << c.element(0)(1) << endl;
	cc << c.element(1)(0) << " " << c.element(1)(1) << endl;
}
