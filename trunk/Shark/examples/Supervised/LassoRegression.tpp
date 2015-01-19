//===========================================================================
/*!
 * 
 *
 * \brief       LASSO Regression
 * 
 * This program demonstrates LASSO regression for the identification
 * of sparse coefficient vectors.
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2013
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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

#include <shark/Data/DataDistribution.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
//###begin<core>
#include <shark/Algorithms/Trainers/LassoRegression.h>
//###end<core>

#include <iostream>
#include <fstream>

using namespace shark;
using namespace std;


class TestProblem : public LabeledDataDistribution<RealVector, RealVector>
{
public:
	TestProblem(size_t informative, size_t nnz, size_t dim)
	: m_informative(informative)
	, m_nnz(nnz)
	, m_dim(dim)
	{ }


	void draw(RealVector& input, RealVector& label) const
	{
		input.resize(m_dim);
		input.clear();
		label.resize(1);

		// we have one informative component per example
		double g = Rng::gauss();;
		size_t i = Rng::discrete(0, m_informative-1);
		input(i) = g;
		label(0) = g;

		// the rest is non-informative
		for (size_t n=1; n<m_nnz; n++)
		{
			size_t i = Rng::discrete(m_informative, m_dim-1);
			input(i) = Rng::gauss();
		}
	}

protected:
	size_t m_informative;
	size_t m_nnz;
	size_t m_dim;
};


int main(int argc, char** argv)
{
	// Define a test problem with 10 out of 1000 informative
	// components. Each instance contains one informative and
	// 49 noise components. 10000 instances are drawn.
	TestProblem prob(10, 50, 1000);
	cout << "generating 100000 points ..." << flush;
	RegressionDataset data = prob.generateDataset(100000);
	cout << " done." << endl;

	// Set the regularization parameter.
	// For this problem the LASSO method identifies the correct
	// subset of 10 informative coefficients for a large range
	// of parameter values.
	//###begin<core>
	double lambda = 1.0;
	//###end<core>

	// trainer and model
	//###begin<core>
	LinearModel<> model;
	LassoRegression<> trainer(lambda);
	//###end<core>

	// train the model
	cout << "LASSO training ..." << flush;
	//###begin<core>
	trainer.train(model, data);
	//###end<core>
	cout << " done." << endl;

	// check non-zero coefficients
	RealMatrix m = model.matrix();
	size_t nnz = 0;
	size_t correct = 0;
	size_t wrong = 0;
	for (size_t j=0; j<m.size2(); j++)
	{
		if (m(0, j) != 0.0)
		{
			nnz++;
			if (j < 10) correct++;
			else wrong++;
		}
	}
	cout << "solution statistics:" << endl;
	cout << "  number of non-zero coefficients: " << nnz << endl;
	cout << "  correctly identified coefficients: " << correct << endl;
	cout << "  wrongly identified coefficients: " << wrong << endl;
}
