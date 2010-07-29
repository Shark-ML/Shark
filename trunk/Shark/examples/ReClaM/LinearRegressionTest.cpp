//===========================================================================
/*!
 *  \file LinearRegressionTest.cpp
 *
 *  \author Tobias Glasmachers
 *
 *  \par Copyright (c) 1998-2007:
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
//===========================================================================


#include <Rng/GlobalRng.h>
#include <ReClaM/LinearModel.h>
#include <ReClaM/LinearRegression.h>
#include <ReClaM/Dataset.h>
#include <ReClaM/MeanSquaredError.h>


// Gaussian Noise on linear function
class RegressionProblem : public DataSource
{
public:
	RegressionProblem()
	{
		dataDim = 1;
		targetDim = 1;
	}

	bool GetData(Array<double>& data, Array<double>& target, int count)
	{
		data.resize(count, 1, false);
		target.resize(count, 1, false);
		int i;
		for (i=0; i<count; i++)
		{
			data(i, 0) = Rng::gauss();
			target(i, 0) = 0.5 * data(i, 0) + 1.0 + 0.1 * Rng::gauss();
		}
		return true;
	}
};


int main(int argc, char** argv)
{
	// generate multi class dataset with 100 training and 1000 test examples
	RegressionProblem problem;
	Dataset dataset;
	dataset.CreateFromSource(problem, 100, 1000);

	// construct model and optimizer
	AffineLinearFunction model(1);
	LinearRegression optimizer;
	optimizer.init(model);

	// train the model
	std::cout << "Linear Regression ..." << std::flush;
	optimizer.optimize(model, dataset.getTrainingData(), dataset.getTrainingTarget());
	std::cout << " done." << std::endl;

	// output training and test errors:
	MeanSquaredError mse;
	double train = mse.error(model, dataset.getTrainingData(), dataset.getTrainingTarget());
	double test = mse.error(model, dataset.getTestData(), dataset.getTestTarget());
	std::cout << "Training MSE: " << train << std::endl;
	std::cout << "    Test MSE: " << test << std::endl;
	std::cout << "slope: " << model.getParameter(0) << "  ---  optimal: 0.5" << std::endl;
	std::cout << "offset: " << model.getParameter(1) << "  ---  optimal: 1.0" << std::endl;

	// lines below are for self-testing this example, please ignore
	if (train <= 0.00926082) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
