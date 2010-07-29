//===========================================================================
/*!
 *  \file LinearClassiferTest.cpp
 *
 *  \author: Tobias Glasmachers
 *
 *  \par Copyright (c) 1998-2008:
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
#include <ReClaM/Dataset.h>
#include <ReClaM/LinearModel.h>
#include <ReClaM/LDA.h>
#include <ReClaM/WTA.h>


// Simple data generating distrubution: Gaussians around standard basis vectors
class MultiClassProblem : public DataSource
{
public:
	MultiClassProblem()
	{
		dataDim = 2;
		targetDim = 3;
	}

	bool GetData(Array<double>& data, Array<double>& target, int count)
	{
		data.resize(count, 2, false);
		target.resize(count, 3, false);
		target = 0.0;
		int i, c;
		for (i=0; i<count; i++)
		{
			c = Rng::discrete(0, 2);
			data(i, 1) = 0.7 * Rng::gauss();
			data(i, 0) = 0.7 * Rng::gauss() - 1.0 * data(i, 1);
			if (c == 0) data(i, 0) -= 1.0;
			else if (c == 1) data(i, 1) += 2.0;
			else if (c == 2) data(i, 0) += 1.0;
			target(i, c) = 1.0;
		}
		return true;
	}
};


int main(int argc, char** argv)
{
	// generate multi class dataset with 100 training and 1000 test examples
	MultiClassProblem problem;
	Dataset dataset;
	dataset.CreateFromSource(problem, 100, 1000);

	// construct model and optimizer for LDA with 2 dimensions and 3 classes
	LinearClassifier model(2, 3);
	LDA optimizer;
	optimizer.init(model);

	// train the model
	std::cout << "LDA training ..." << std::flush;
	optimizer.optimize(model, dataset.getTrainingData(), dataset.getTrainingTarget());
	std::cout << " done." << std::endl;

	// count the errors on the test set
	WTA err;
	double e = err.error(model, dataset.getTestData(), dataset.getTestTarget());
	std::cout << "fraction of errors: " << e << std::endl;

	// lines below are for self-testing this example, please ignore
	if (e <= 0.081) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
