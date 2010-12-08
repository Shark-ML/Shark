//===========================================================================
/*!
*  \file GridSearch.cpp
*
*  \brief optimization by grid or point set search
*
*  This file provides a collection of quite simple optimizers.
*  It provides a basic grid search, a nested grid search and
*  a search on a predefined set of points.
*
*  \author  T. Glasmachers
*  \date    2006
*
*  \par Copyright (c) 1999-2006:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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


#include <ReClaM/GridSearch.h>


GridSearch::GridSearch(int verbosity)
{
	this->verbosity = verbosity;
}

GridSearch::~GridSearch()
{
}


void GridSearch::init(Model& model)
{
	SHARKEXCEPTION("[GridSearch::init] A default initialization is impossible for the grid optimizer.");
}

void GridSearch::init(int params, double min, double max, int nodes)
{
	RANGE_CHECK(nodes > 1);
	if (params < 1 || min >= max) SHARKEXCEPTION("[GridSearch::init] invalid intialization");

	numberOfValues.resize(params, false);
	numberOfValues = nodes;
	nodeValues.resize(params, nodes, false);
	int i, j;
	for (j = 0; j < params; j++)
	{
		for (i = 0; i < nodes; i++)
		{
			if ( nodes != 1 ) nodeValues(j, i) = min + i * (max - min) / (nodes - 1.0);
			else nodeValues(j, i) = (min + max)  / 2.0;
		}
	}
}

void GridSearch::init(int params, const Array<double>& min, const Array<double>& max, const Array<int>& nodes)
{
	SIZE_CHECK(nodes.ndim() == 1);
	SIZE_CHECK(min.ndim() == 1);
	SIZE_CHECK(max.ndim() == 1);
	SIZE_CHECK(nodes.dim(0) == (unsigned)params);
	SIZE_CHECK(min.dim(0) == (unsigned)params);
	SIZE_CHECK(max.dim(0) == (unsigned)params);

	int i, j;
	int nmax = 0;
	for (i = 0; i < params; i++) if (nodes(i) > nmax) nmax = nodes(i);
	numberOfValues.resize(params, false);
	nodeValues.resize(params, nmax, false);
	for (j = 0; j < params; j++)
	{
		for (i = 0; i < nodes(j); i++)
		{
			if ( nodes(j) != 1 )
				nodeValues(j, i) = min(j) + i * (max(j) - min(j)) / (nodes(j) - 1.0);
			else
				nodeValues(j, i) = ( min(j) + max(j) ) / 2.0;
		}
		numberOfValues(j) = nodes(j);
	}
}




void GridSearch::init(int params, const Array<double>& values)
{
	int d = values.dim(0);
	int i, j;
	numberOfValues.resize(params, false);
	numberOfValues = d;
	nodeValues.resize(params, d, false);
	for (j = 0; j < params; j++)
	{
		for (i = 0; i < d; i++)
		{
			nodeValues(j, i) = values(i);
		}
	}
}

void GridSearch::init(int params, const Array<int>& nodes, const Array<double>& values)
{
	if (params != (int)nodes.dim(0) || params != (int)values.dim(0))
		SHARKEXCEPTION("[GridSearch::init] Dimension conflict");

	numberOfValues = nodes;
	nodeValues = values;
}

void GridSearch::init(const Model& model, unsigned max_values)
{
	if ( max_values < 1 ) throw SHARKEXCEPTION("[GridSearch::init]"
		"need at least one set of parameters for the grid.");
	unsigned num_params = model.getParameterDimension();
	if ( ( !model.isFeasible() ) || ( num_params < 1 ) ) throw SHARKEXCEPTION("[GridSearch::init]"
		"need a feasible model with at least one parameter.");
		
	//size and fill member arrays
	numberOfValues.resize( num_params, false );
	nodeValues.resize( num_params, max_values, false );
	for (unsigned i = 0; i < num_params; i++) {
		numberOfValues(i) = 1;
		nodeValues(i, 0) = model.getParameter(i);
	}
}

void GridSearch::assignLinearRange(unsigned index, unsigned no_of_points, double min, double max)
{
	RANGE_CHECK( (no_of_points >= 1.0) && (no_of_points <= nodeValues.dim(1)) );
	RANGE_CHECK( min <= max );
	RANGE_CHECK( index < nodeValues.dim(0) );
	if ( no_of_points == 1 ) {
		nodeValues(index, 0) = ( min+max) / 2.0;
	}
	else {
		for (unsigned j = 0; j < no_of_points; j++)
			nodeValues(index, j) = min + j*( max-min ) / ( no_of_points-1.0 );
	}
	numberOfValues(index) = no_of_points;
}

void GridSearch::assignExponentialRange(unsigned index, double base_factor, double exp_base, int min, int max)
{
	RANGE_CHECK( min <= max );
	RANGE_CHECK( index < nodeValues.dim(0) );
	for (int j = 0; j <= (max-min); j++)
		nodeValues(index, j) = base_factor * pow( exp_base, j+min );
	numberOfValues(index) = max-min+1;
}


double GridSearch::optimize(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target)
{
	int params = numberOfValues.dim(0);
	Array<int> index(params);
	int i;
	double e;
	double best = 1e100;
	Array<double> best_param(params);
	for (i = 0; i < params; i++) best_param(i) = model.getParameter(i);

	// loop through all grid points
	index = 0;
	while (true)
	{
		// define the parameters
		for (i = 0; i < params; i++) model.setParameter(i, nodeValues(i, index(i)));

		// evaluate the model
		if (model.isFeasible())
		{
			e = errorfunction.error(model, input, target);
			if (e < best)
			{
				best = e;
				for (i = 0; i < params; i++) best_param(i) = model.getParameter(i);
			}

			// progress output
			if (verbosity == 1)
			{
				printf(".");
				fflush(stdout);
			}
			else if (verbosity == 2)
			{
				printf("params = (");
				for (i = 0; i < params; i++) printf(" %g", model.getParameter(i));
				printf(" ) error = %g\n", e);
				fflush(stdout);
			}
		}

		// next index
		for (i = 0; i < params; i++)
		{
			index(i)++;
			if (index(i) < numberOfValues(i)) break;
			index(i) = 0;
		}
		if (i == params) break;
	}

	// write the best parameters into the model
	for (i = 0; i < params; i++) model.setParameter(i, best_param(i));

	return best;
}


////////////////////////////////////////////////////////////


NestedGridSearch::NestedGridSearch(int verbosity)
{
	this->verbosity = verbosity;
}

NestedGridSearch::~NestedGridSearch()
{
}


void NestedGridSearch::init(Model& model)
{
	SHARKEXCEPTION("[NestedGridSearch::init] A default initialization is impossible for the grid optimizer.");
}

void NestedGridSearch::init(Model& model, const Array<double>& min, const Array<double>& max)
{
	SIZE_CHECK(min.ndim() == 1);
	SIZE_CHECK(max.ndim() == 1);

	unsigned int d, dc = min.dim(0);
	SIZE_CHECK(max.dim(0) == dc);
	RANGE_CHECK(model.getParameterDimension() >= dc);

	minimum = min;
	maximum = max;

	step.resize(dc, false);
	std::vector<unsigned int> D(dc);
	for (d = 0; d < dc; d++)
	{
		model.setParameter(d, 0.5 *(min(d) + max(d)));
		step(d) = 0.25 * (max(d) - min(d));
		D[d] = 5;
	}

	landscape.resize(D, false);
	landscape = 1e100;
}

double NestedGridSearch::optimize(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target)
{
	SIZE_CHECK(step.ndim() == 1);
	SIZE_CHECK(minimum.ndim() == 1);
	SIZE_CHECK(maximum.ndim() == 1);

	unsigned int d, dc = step.dim(0);
	bool bNew;
	double value;
	double e, best = 1e99;
	std::vector<unsigned int> index(dc);
	std::vector<unsigned int> best_index(dc);
	Array<double> best_param(dc);
	Array<double> old_param(dc);

	SIZE_CHECK(minimum.dim(0) == dc);
	SIZE_CHECK(maximum.dim(0) == dc);
	RANGE_CHECK(model.getParameterDimension() >= dc);

	// initialize variables
	for (d = 0; d < dc; d++)
	{
		value = model.getParameter(d);
		old_param(d) = value;
		best_param(d) = value;
		index[d] = 0;
		best_index[d] = 2;
	}

	// loop through the grid
	while (true)
	{
		// compute the grid point,
		// define it as the model parameters
		// and check whether the computation
		// has to be done at all
		bNew = true;
		e = landscape(index);

		if (e < 1e100) bNew = false;
		else
		{
			// set the parameters
			for (d = 0; d < dc; d++)
			{
				value = old_param(d) + (index[d] - 2.0) * step(d);
				if (value < minimum(d) || value > maximum(d)) bNew = false;
				model.setParameter(d, value);
			}
		}

		// evaluate the grid point
		if (bNew)
		{
			if (model.isFeasible())
			{
				e = errorfunction.error(model, input, target);
				landscape(index) = e;

				// progress output
				if (verbosity == 1)
				{
					printf(".");
					fflush(stdout);
				}
				else if (verbosity == 2)
				{
					printf("params = (");
					for (unsigned int i = 0; i < model.getParameterDimension(); i++) printf(" %g", model.getParameter(i));
					printf(" ) error = %g\n", e);
					fflush(stdout);
				}
			}
		}

		// remember the best solution
		if (e < best)
		{
			best = e;
			for (d = 0; d < dc; d++)
			{
				best_index[d] = index[d];
				best_param(d) = old_param(d) + (index[d] - 2.0) * step(d);
			}
		}

		// move to the next grid point
		for (d = 0; d < dc; d++)
		{
			index[d]++;
			if (index[d] <= 4) break;
			index[d] = 0;
		}
		if (d == dc) break;
	}

	// zoom into the error landscape array
	Array<double> zoomed = landscape;
	std::vector<unsigned int> zoomed_index(dc);
	for (d = 0; d < dc; d++) zoomed_index[d] = 0;
	// loop through the grid
	while (true)
	{
		// if appropriate, copy the error landscape point
		for (d = 0; d < dc; d++)
		{
			if ((zoomed_index[d] & 1) == 1) break;
			index[d] = best_index[d] + (zoomed_index[d] / 2) - 1;
			if (index[d] > 4) break;
		}
		if (d == dc) zoomed(zoomed_index) = landscape(index);
		else zoomed(zoomed_index) = 1e100;

		// move to the next grid point
		for (d = 0; d < dc; d++)
		{
			zoomed_index[d]++;
			if (zoomed_index[d] <= 4) break;
			zoomed_index[d] = 0;
		}
		if (d == dc) break;
	}
	landscape = zoomed;

	// decrease the step sizes
	for (d = 0; d < dc; d++) step(d) *= 0.5;

	// load the best parameter configuration into the model
	for (d = 0; d < dc; d++) model.setParameter(d, best_param(d));

	// return the lowest error value archieved
	return best;
}


////////////////////////////////////////////////////////////


PointSearch::PointSearch(int verbosity)
{
	this->verbosity = verbosity;
}

PointSearch::~PointSearch()
{
}


void PointSearch::init(Model& model)
{
	SHARKEXCEPTION("[PointSearch::init] A default initialization is impossible for the point search optimizer.");
}

void PointSearch::init(Array<double>& values)
{
	nodes = values;
}

double PointSearch::optimize(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target)
{
	int t, tc = nodes.dim(0);
	int p, pc = nodes.dim(1);
	double e;
	double best = 1e100;
	int best_index = -1;

	// loop through all points
	for (t = 0; t < tc; t++)
	{
		// define the parameters
		for (p = 0; p < pc; p++) model.setParameter(p, nodes(t, p));

		// evaluate the model
		if (model.isFeasible())
		{
			e = errorfunction.error(model, input, target);
			if (e < best)
			{
				best = e;
				best_index = t;
			}

			// progress output
			if (verbosity == 1)
			{
				printf(".");
				fflush(stdout);
			}
			else if (verbosity == 2)
			{
				printf("params = (");
				for (unsigned int i = 0; i < model.getParameterDimension(); i++) printf(" %g", model.getParameter(i));
				printf(" ) error = %g\n", e);
				fflush(stdout);
			}
		}
	}

	// write the best parameters into the model
	for (p = 0; p < pc; p++) model.setParameter(p, nodes(best_index, p));

	return best;
}

