/*!
*  \file KNN.cpp
*
*  \author Tobias Glasmachers
*
*  \par
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR> 
*
*  \par Project:
*      ReClaM
*
*
*  <BR>
*
*
*  <BR><HR>
*  This file is part of ReClaM. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 2, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, write to the Free Software
*  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/

#include <Rng/GlobalRng.h>
#include <ReClaM/ArtificialDistributions.h>
#include <ReClaM/Dataset.h>
#include <ReClaM/KernelNearestNeighbor.h>
#include <ReClaM/ClassificationError.h>
#include <stdio.h>
#include <iostream>


using namespace std;


int main()
{
	Rng::seed(10);

	double gamma = 0.5;
	RBFKernel k(gamma);

	cout << endl;
	cout << "*** kernel nearest neighbor classifier ***" << endl;
	cout << endl;

	// create the xor problem with uniformly distributed examples
	unsigned int n = 3;

	cout << "Generating 100 training and 10000 test examples ..." << flush;
	Chessboard chess(2, 2);
	Dataset dataset;
	dataset.CreateFromSource(chess, 100, 10000);
	const Array<double>& x = dataset.getTrainingData();
	const Array<double>& y = dataset.getTrainingTarget();
	cout << " done." << endl;

	// create the kernel mean classifier
	cout << "Creating the 3-nearest-neighbor classifier ..." << flush;
	KernelNearestNeighbor knn(x, y, &k, n);
	cout << " done." << endl;

	// estimate the accuracy on the test set
	cout << "Testing ..." << flush;
	ClassificationError ce;
	double acc = 1.0 - ce.error(knn, dataset.getTestData(), dataset.getTestTarget());
	cout << " done." << endl;
	cout << "Estimated accuracy: " << 100.0 * acc << "%" << endl << endl;

	// lines below are for self-testing this example, please ignore
	if (acc >= 0.92) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
