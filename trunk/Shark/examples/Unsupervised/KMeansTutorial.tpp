//===========================================================================
/*!
 *  \brief k-means Clustering Tutorial Sample Code, requires the data
 *  set faithful.csv
 *
 *  \author C. Igel
 *  \date 2011
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

#include <shark/Data/Csv.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>

#include <shark/Algorithms/KMeans.h>
#include <shark/Models/Clustering/HardClusteringModel.h>

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
		import_csv(data, argv[1], " ");
	} 
	catch (...) {
		cerr << "unable to read data from file " <<  argv[1] << endl;
		exit(EXIT_FAILURE);
	}
	std::size_t elements = data.numberOfElements();

	// write statistics of input data
	cout << "number of data points: " << elements << " dimensions: " << dataDimension(data) << endl;

	// normalize data
	Normalizer<> normalizer;
	NormalizeComponentsUnitVariance<> normalizingTrainer;
	normalizingTrainer.train(normalizer, data);
	data = normalizer(data);

	// compute centroids using k-means clustering
	Centroids centroids;
	size_t iterations = kMeans(data, 2, centroids);

	// report number of iterations by the clustering algorithm
	cout << "iterations: " << iterations << endl;

	// write cluster centers/centroids
	Data<RealVector> const& c = centroids.centroids();
	cout<<c<<std::endl;

	// cluster data
	HardClusteringModel<RealVector> model(&centroids);
	Data<unsigned> clusters = model(data);

	// write results to files
	ofstream c1("cl1.csv");
	ofstream c2("cl2.csv");
	ofstream cc("clc.csv");
	for(std::size_t i=0; i != elements; i++) {
		if(clusters.element(i)) 
			c1 << data.element(i)(0) << " " << data.element(i)(1) << endl;
		else 
			c2 << data.element(i)(0) << " " << data.element(i)(1) << endl;
	}
	cc << c.element(0)(0) << " " << c.element(0)(1) << endl;
	cc << c.element(1)(0) << " " << c.element(1)(1) << endl;
}
