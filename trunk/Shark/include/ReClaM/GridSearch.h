/*!
*  \file GridSearch.h
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
*  This file is part of Shark. This library is free software;
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


#ifndef _GridSearch_H_
#define _GridSearch_H_


#include <SharkDefs.h>
#include <ReClaM/Optimizer.h>


//!
//! \brief Optimize by trying out a grid of configurations
//!
//! \par
//! The GridSearch class allows for the definition of a grid in
//! parameter space. It does a simple one-step optimization over
//! the grid by trying out every possible parameter combination.
//! Please note that the computation effort grows exponentially
//! with the number of parameters.
//!
//! \par
//! If you only want to try a subset of the grid, consider using
//! the #PointSearch class instead.
//! A more sophisticated (less exhaustive) grid search variant is
//! available with the #NestedGridSearch class.
//!
class GridSearch : public Optimizer
{
public:
	//! Constructor
	//!
	//! \param  verbosity  0=none, 1=dots, 2=values
	//!
	GridSearch(int verbosity = 0);

	//! Destructor
	~GridSearch();


	//! There is no useful default initialization for this optimizer.
	//! Thus, this member throws an exception.
	void init(Model& model);

	//! uniform initialization for all parameters
	//! \param  params  number of model parameters to optimize
	//! \param  min     smallest parameter value
	//! \param  max     largest parameter value
	//! \param  nodes   total number of values in the interval
	void init(int params, double min, double max, int nodes);

	//! individual definition for every parameter
	//! \param  params  number of model parameters to optimize
	//! \param  min     smallest value for every parameter
	//! \param  max     largest value for every parameter
	//! \param  nodes   total number of values for every parameter
	void init(int params, const Array<double>& min, const Array<double>& max, const Array<int>& nodes);

	//! uniform definition of the values to test for all parameters
	//! \param  params  number of model parameters to optimize
	//! \param  values  values used for every coordinate
	void init(int params, const Array<double>& values);

	//! individual definition for every parameter
	//! \param  params  number of model parameters to optimize
	//! \param  nodes   total number of values for every parameter
	//! \param  values  values used. The first dimension is the parameter, the second dimension is the node. Unused entries are ignored.
	void init(int params, const Array<int>& nodes, const Array<double>& values);

	/*! Construct naive grid with as many dimensions as parameters in the model 
	 *  passed. For each dimension, only one parameter is assigned: the current value 
	 *  of the respective model parameter. To make the grid non-trivial, combine 
	 *  this init routine with the functions #assignLinearRange or #assignExponentialRange.
	 *  \param model the model from which the trivial grid will be constructed
	 *  \param max_values maximum number of grid values that are later allowed for each dimension of this grid. */
	void init(const Model& model, unsigned max_values);

	/*! Assign linearly progressing grid values to one certain parameter only. 
	 *  This is especially useful in combination with init'ing from a model.
	 *  \param index the index of the parameter to which grid values are assigned
	 *  \param no_of_points how many grid points should be assigned to that parameter
	 *  \param min smallest value for that parameter
	 *  \param max largest value for that parameter */
	void assignLinearRange(unsigned index, unsigned no_of_points, double min, double max);
	
	/*! Set exponentially progressing grid values for one certain parameter only. 
	 *  This is especially useful in combination with init'ing from a model. The 
	 *  grid points will be filled with values \f$ base_factor \cdot exp_base ^i \f$, 
	 *  where i does integer steps between min and max.
	 *  \param index the index of the parameter that gets new grid values
	 *  \param base_factor the value that the exponential base grid should be multiplied by
	 *  \param exp_base the exponential grid will progress on this base (e.g. 2, 10)
	 *  \param min the smallest exponent for #exp_base
	 *  \param max the largest exponent for #exp_base  */
	void assignExponentialRange(unsigned index, double base_factor, double exp_base, int min, int max);

	//! Please note that for the grid search optimizer it does
	//! not make sense to call #optimize more than once, as the
	//! solution does not improve iteratively.
	double optimize(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target);

protected:
	//! The array holds the number of grid values for every parameter axis.
	Array<int> numberOfValues;

	//! The array columns contain the grid values for the corresponding parameter axis.
	//! As all columns have the same size, some values may be meaningless.
	Array<double> nodeValues;

	//! verbosity level
	int verbosity;
};


//!
//! \brief Nested grid search
//!
//! \par
//! The NestedGridSearch class is an iterative optimizer,
//! doing one grid search in every iteration. In every
//! iteration, it halves the grid extent doubling the
//! resolution in every coordinate.
//!
//! \par
//! Although nested grid search is much less exhaustive
//! than standard grid search, it still suffers from
//! exponential time and memory complexity in the number
//! of variables optimized. Therefore, if the number of
//! variables is larger than 2 or 3, consider using the
//! CMA instead.
//!
//! \par
//! Nested grid search works as follows: The optimizer
//! defined a 5x5x...x5 equi-distant grid (depending on
//! the search space dimension) on an initially defined
//! search cube. During every grid search iteration,
//! the error is computed for all  grid points not seen
//! during previous iterations. Then the grid is moved
//! to the best grid point found so far and contracted
//! by a factor of two in each dimension. Each call to
//! the optimize() function performs one such step.
//!
//! \par
//! Let N denote the number of parameters to optimize.
//! To compute the error landscape at the current
//! zoom level, the algorithm has to do \f$ 5^N \f$
//! error function evaluation in the first iteration,
//! and roughtly \f$ 5^N - 3^N \f$ evaluations in
//! subsequent iterations.
//!
//! \par
//! The grid is always centered around the best
//! solution currently known. If this solution is
//! located at the boundary, the landscape may exceed
//! the parameter range defined #minimum and #maximum.
//! These invalid landscape values are not used and
//! are always set to 1e100, indicating non-optimality.
//!
class NestedGridSearch : public Optimizer
{
public:
	//! Constructor
	//!
	//! \param  verbosity  0=none, 1=dots, 2=values
	//!
	NestedGridSearch(int verbosity = 0);

	//! Destructor
	~NestedGridSearch();


	//! There is no useful default initialization for this optimizer.
	//! Thus, this member throws an exception.
	void init(Model& model);

	//!
	//! \brief Initialization of the nested grid search.
	//!
	//! \par
	//! The min and max arrays define ranges for every parameter to optimize.
	//! These ranges are strict, that is, the algorithm will not try values
	//! beyond the range, even if is finds a boundary minimum.
	//!
	//! \param  model  #Model to optimize
	//! \param  min    lower end of the parameter range
	//! \param  max    upper end of the parameter range
	void init(Model& model, const Array<double>& min, const Array<double>& max);

	//! Every call of the optimization member computes the
	//! error landscape on the current grid. It picks the
	//! best error value and zooms into the error landscape
	//! by a factor of 2 reusing already computed grid points.
	double optimize(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target);

protected:
	//! minimum parameter value to check
	Array<double> minimum;

	//! maximum parameter value to check
	Array<double> maximum;

	//! current step size for every parameter
	Array<double> step;

	//! \brief error landscape
	//!
	//! \par
	//! The landscape array has as many dimensions as
	//! there are parameters to optimize. In every array
	//! dimension there are 5 entries.
	Array<double> landscape;

	//! verbosity level
	int verbosity;
};


//!
//! \brief Optimize by trying out predefined configurations
//!
//! \par
//! The PointSearch class is similair to the #GridSearch class
//! by the property that it optimizes a model in a single pass
//! just trying out a predefined number of parameter configurations.
//! The main difference is that every parameter configuration has
//! to be explicitly defined. It is not possible to define a set
//! of values for every axis; refer to GridSearch for this purpose.
//! Thus, the optPointSearch class allows for more flexibility.
//!
class PointSearch : public Optimizer
{
public:
	//! Constructor
	//!
	//! \param  verbosity  0=none, 1=dots, 2=values
	//!
	PointSearch(int verbosity);

	//! Destructor
	~PointSearch();


	//! There is no useful default initialization for this optimizer.
	//! Thus, this member throws an exception.
	void init(Model& model);

	//! Initialization of the search points.
	//! \param  values  two-dimensional array; every column contains one parameter configuration.
	void init(Array<double>& values);

	//! Please note that for the point search optimizer it does
	//! not make sense to call #optimize more than once, as the
	//! solution does not improve iteratively.
	double optimize(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target);

protected:
	//! The array holds one parameter configuration in every column.
	Array<double> nodes;

	//! verbosity level
	int verbosity;
};


#endif

