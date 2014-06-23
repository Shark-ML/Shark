//===========================================================================
/*!
 * 
 *
 * \brief       GridSearch.h
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010
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

#ifndef SHARK_ALGORITHMS_GRIDSEARCH_H
#define SHARK_ALGORITHMS_GRIDSEARCH_H


#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Rng/GlobalRng.h>
#include <boost/foreach.hpp>

#include <boost/serialization/vector.hpp>


namespace shark {

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
    //! the PointSearch class instead.
    //! A more sophisticated (less exhaustive) grid search variant is
    //! available with the NestedGridSearch class.
    //!
    class GridSearch : public AbstractSingleObjectiveOptimizer<RealVector >
    {
    public:
	GridSearch(){
		m_configured=false;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "GridSearch"; }

	//! uniform initialization for all parameters
	//! \param  params  number of model parameters to optimize
	//! \param  min     smallest parameter value
	//! \param  max     largest parameter value
	//! \param  numSections   total number of values in the interval
	void configure(size_t params, double min, double max, size_t numSections)
	{
	    SIZE_CHECK(params>=1);
	    RANGE_CHECK(min<=max);
	    SIZE_CHECK(numSections>=1);
	    m_nodeValues.resize(params);
	    for (size_t i = 0; i < numSections; i++)
		{
		    double section = min + i * (max - min) / (numSections - 1.0);
		    BOOST_FOREACH(std::vector<double>& node,m_nodeValues)
			node.push_back(section);
		}
	    m_configured=true;
	}

	//! individual definition for every parameter
	//! \param  min     smallest value for every parameter
	//! \param  max     largest value for every parameter
	//! \param  sections   total number of values for every parameter
	void configure(const std::vector<double>& min, const std::vector<double>& max, const std::vector<size_t>& sections)
	{
	    size_t params = min.size();
	    SIZE_CHECK(sections.size() == params);
	    SIZE_CHECK(max.size() == params);
	    RANGE_CHECK(min <= max);

	    m_nodeValues.resize(params);
	    for (size_t dimension = 0; dimension < params; dimension++)
		{
		    size_t numSections = sections[dimension];
		    std::vector<double>& node = m_nodeValues[dimension];
		    node.resize(numSections);

		    if ( numSections == 1 )
			{
			    node[0] = (( min[dimension] + max[dimension] ) / 2.0);
			}
		    else for (size_t section = 0; section < numSections; section++)
			     {
				 node[section] = (min[dimension] + section * (max[dimension] - min[dimension]) / (numSections - 1.0));
			     }
		}
	    m_configured=true;
	}


	//! special case for 2D grid, individual definition for every parameter
	//! \param  min1     smallest value for first parameter
	//! \param  max1     largest value for first parameter
	//! \param  sections1   total number of values for first parameter
	//! \param  min2     smallest value for second parameter
	//! \param  max2     largest value for second parameter
	//! \param  sections2   total number of values for second parameter
	void configure(double min1, double max1, size_t sections1, double min2, double max2, size_t sections2)
	{
		RANGE_CHECK(min1<=max1);
		RANGE_CHECK(min2<=max2);
		RANGE_CHECK(sections1 > 0);
		RANGE_CHECK(sections2 > 0);

		m_nodeValues.resize(2u);

		if ( sections1 == 1 ) {
			m_nodeValues[0].push_back(( min1 + max1 ) / 2.0);
		} else {
			for (size_t section = 0; section < sections1; section++)
				m_nodeValues[0].push_back(min1 + section * (max1 - min1) / (sections1 - 1.0));
		}

		if ( sections2 == 1 ) {
			m_nodeValues[1].push_back(( min2 + max2 ) / 2.0);
		} else {
			for (size_t section = 0; section < sections2; section++)
				m_nodeValues[1].push_back(min2 + section * (max2 - min2) / (sections2 - 1.0));
		}
	}

	//! special case for line search
	//! \param  min1     smallest value for first parameter
	//! \param  max1     largest value for first parameter
	//! \param  sections1   total number of values for first parameter
	void configure(double min1, double max1, size_t sections1)
	{
		RANGE_CHECK(min1<=max1);
		RANGE_CHECK(sections1 > 0);

		m_nodeValues.resize(1u);

		if ( sections1 == 1 ) {
			m_nodeValues[0].push_back(( min1 + max1 ) / 2.0);
		} else {
			for (size_t section = 0; section < sections1; section++)
				m_nodeValues[0].push_back(min1 + section * (max1 - min1) / (sections1 - 1.0));
		}
	}


	//! uniform definition of the values to test for all parameters
	//! \param  params  number of model parameters to optimize
	//! \param  values  values used for every coordinate
	void configure(size_t params, const std::vector<double>& values)
	{
	    SIZE_CHECK(params > 0);
	    SIZE_CHECK(values.size() > 0);
	    m_nodeValues.resize(params);
	    BOOST_FOREACH(std::vector<double>& node,m_nodeValues)
		node=values;
	    m_configured=true;
	}

	//! individual definition for every parameter
	//! \param  values  values used. The first dimension is the parameter, the second dimension is the node.
	void configure(const std::vector<std::vector<double> >& values)
	{
	    SIZE_CHECK(values.size() > 0);
	    m_nodeValues = values;
	    m_configured=true;
	}

	//from ISerializable
	virtual void read( InArchive & archive )
	{
	    archive>>m_nodeValues;
	    archive>>m_configured;
	    archive>>m_best.point;
	    archive>>m_best.value;
	}

	virtual void write( OutArchive & archive ) const
	{
	    archive<<m_nodeValues;
	    archive<<m_configured;
	    archive<<m_best.point;
	    archive<<m_best.value;
	}

	/*! If Gridsearch wasn't configured before calling this method, it is default constructed
	 *  as a net spanning the range [-1,1] in all dimensions with 5 searchpoints (-1,-0.5,0,0.5,1).
	 *  so don't forget to scale the parameter-ranges of the objective function!
	 *  The startingPoint can actually be anything, only its dimension has to be correct.
	 */
	virtual void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint) {
	    (void) objectiveFunction;

	    if(!m_configured)
            configure(startingPoint.size(),-1,1,5);
	    SIZE_CHECK(startingPoint.size() == m_nodeValues.size());
	    m_best.point=startingPoint;
	}
	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	/*! Assign linearly progressing grid values to one certain parameter only.
	 *  This is especially useful if one parameter needs special treatment
	 *  \param index the index of the parameter to which grid values are assigned
	 *  \param noOfSections how many grid points should be assigned to that parameter
	 *  \param min smallest value for that parameter
	 *  \param max largest value for that parameter */
	void assignLinearRange(size_t index, size_t noOfSections, double min, double max)
	{
	    SIZE_CHECK( noOfSections >= 1);
	    RANGE_CHECK( min <= max );
	    SIZE_CHECK( index < m_nodeValues.size() );

	    m_nodeValues[index].clear();
	    if ( noOfSections == 1 ) {
		    m_nodeValues[index].push_back(( min+max) / 2.0);
		}
	    else {
		    m_nodeValues[index].reserve(noOfSections);
		    for (size_t section = 0; section < noOfSections; section++)
			m_nodeValues[index].push_back(min + section*( max-min ) / ( noOfSections-1.0 ));
		}
	}

	/*! Set exponentially progressing grid values for one certain parameter only.
	 *  This is especially useful if one parameter needs special treatment. The
	 *  grid points will be filled with values \f$ factor \cdot expbase ^i \f$,
	 *  where i does integer steps between min and max.
	 *  \param index the index of the parameter that gets new grid values
	 *  \param factor the value that the exponential base grid should be multiplied by
	 *  \param exp_base the exponential grid will progress on this base (e.g. 2, 10)
	 *  \param min the smallest exponent for exp_base
	 *  \param max the largest exponent for exp_base  */
	void assignExponentialRange(size_t index, double factor, double exp_base, int min, int max)
	{
	    SIZE_CHECK( min <= max );
	    RANGE_CHECK( index < m_nodeValues.size() );

	    m_nodeValues[index].clear();
	    m_nodeValues[index].reserve(max-min);
	    for (int section = 0; section <= (max-min); section++)
		m_nodeValues[index].push_back( factor * std::pow( exp_base, section+min ));
	}

	//! Please note that for the grid search optimizer it does
	//! not make sense to call step more than once, as the
	//! solution does not improve iteratively.
	void step(const ObjectiveFunctionType& objectiveFunction) {
	    size_t dimensions = m_nodeValues.size();
	    std::vector<size_t> index(dimensions, 0);
	    m_best.value = 1e100;
	    RealVector point(dimensions);

	    // loop through all grid points
	    while (true)
		{
		    // define the parameters
		    for (size_t dimension = 0; dimension < dimensions; dimension++)
                point(dimension) = m_nodeValues[dimension][index[dimension]];

		    // evaluate the model
		    if (objectiveFunction.isFeasible(point))
			{
			    double error = objectiveFunction.eval(point);

#ifdef SHARK_CV_VERBOSE_1
			    std::cout << "." << std::flush;
#endif
#ifdef SHARK_CV_VERBOSE
			    std::cout << point << "\t" << error << std::endl;
#endif
			    if (error < m_best.value)
				{
				    m_best.value = error;
				    m_best.point = point;		// [TG] swap() solution is out, caused ugly memory bug, I changed this back
				}
			}

		    // next index
		    size_t dimension = 0;
		    for (; dimension < dimensions; dimension++)
			{
			    index[dimension]++;
			    if (index[dimension] < m_nodeValues[dimension].size()) break;
			    index[dimension] = 0;
			}
		    if (dimension == dimensions) break;
		}
#ifdef SHARK_CV_VERBOSE_1
        std::cout << std::endl;
#endif
	}

    protected:

	//! The array columns contain the grid values for the corresponding parameter axis.
	std::vector<std::vector<double> > m_nodeValues;

	bool m_configured;
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
    //! the error is computed for all  grid points.
    //!Then the grid is moved
    //! to the best grid point found so far and contracted
    //! by a factor of two in each dimension. Each call to
    //! the optimize() function performs one such step.
    //!
    //! \par
    //! Let N denote the number of parameters to optimize.
    //! To compute the error landscape at the current
    //! zoom level, the algorithm has to do \f$ 5^N \f$
    //! error function evaluations in every iteration.
    //!
    //! \par
    //! The grid is always centered around the best
    //! solution currently known. If this solution is
    //! located at the boundary, the landscape may exceed
    //! the parameter range defined m_minimum and m_maximum.
    //! These invalid landscape values are not used.
    //!
    class NestedGridSearch : public AbstractSingleObjectiveOptimizer<RealVector >
    {
    public:
	//! Constructor
	NestedGridSearch()
	{
		m_configured=false;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NestedGridSearch"; }

	void configure(PropertyTree const&){}
	//!
	//! \brief Initialization of the nested grid search.
	//!
	//! \par
	//! The min and max arrays define ranges for every parameter to optimize.
	//! These ranges are strict, that is, the algorithm will not try values
	//! beyond the range, even if is finds a boundary minimum.
	//!
	//! \param  min    lower end of the parameter range
	//! \param  max    upper end of the parameter range
	void configure(const std::vector<double>& min, const std::vector<double>& max)
	{
	    size_t dimensions = min.size();
	    SIZE_CHECK(max.size() == dimensions);

	    m_minimum = min;
	    m_maximum = max;

	    m_stepsize.resize(dimensions);
	    m_best.point.resize(dimensions);
	    for (size_t dimension = 0; dimension < dimensions; dimension++)
		{
		    m_best.point(dimension)=0.5 *(min[dimension] + max[dimension]);
		    m_stepsize[dimension] = 0.25 * (max[dimension] - min[dimension]);
		}
	    m_configured=true;
	}

	//!
	//! \brief Initialization of the nested grid search.
	//!
	//! \par
	//! The min and max values define ranges for every parameter to optimize.
	//! These ranges are strict, that is, the algorithm will not try values
	//! beyond the range, even if is finds a boundary minimum.
	//!
	//! \param parameters number of parameters to optimize
	//! \param  min    lower end of the parameter range
	//! \param  max    upper end of the parameter range
	void configure(size_t parameters, double min, double max)
	{
	    SIZE_CHECK(parameters > 0);

	    m_minimum=std::vector<double>(parameters,min);
	    m_maximum=std::vector<double>(parameters,max);
	    m_stepsize=std::vector<double>(parameters,0.25 * (max - min));

	    m_best.point.resize(parameters);

	    double start=0.5 *(min + max);
	    BOOST_FOREACH(double& value,m_best.point)
		value=start;
	    m_configured=true;
	}

	//from ISerializable
	virtual void read( InArchive & archive )
	{
	    archive>>m_minimum;
	    archive>>m_maximum;
	    archive>>m_stepsize;
	    archive>>m_configured;
	    archive>>m_best.point;
	    archive>>m_best.value;
	}

	virtual void write( OutArchive & archive ) const
	{
	    archive<<m_minimum;
	    archive<<m_maximum;
	    archive<<m_stepsize;
	    archive<<m_configured;
	    archive<<m_best.point;
	    archive<<m_best.value;
	}

	//! if NestedGridSearch was not configured before this call, it is default initialized ti the range[-1,1] for every parameter
	virtual void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint) {
	    (void) objectiveFunction;

	    if(!m_configured)
		configure(startingPoint.size(),-1,1);
	    SIZE_CHECK(m_stepsize.size()==startingPoint.size());

	}
	using AbstractSingleObjectiveOptimizer<RealVector >::init;


	//! Every call of the optimization member computes the
	//! error landscape on the current grid. It picks the
	//! best error value and zooms into the error landscape
	//! by a factor of 2.
	void step(const ObjectiveFunctionType& objectiveFunction) {
	    size_t dimensions = m_stepsize.size();
	    SIZE_CHECK(m_minimum.size() == dimensions);
	    SIZE_CHECK(m_maximum.size() == dimensions);

	    m_best.value = 1e99;
	    std::vector<size_t> index(dimensions,0);

	    RealVector point=m_best.point;

	    // loop through the grid
	    while (true)
		{
		    // compute the grid point,

		    // set the parameters
		    bool compute=true;
		    for (size_t d = 0; d < dimensions; d++)
			{
			    point(d) += (index[d] - 2.0) * m_stepsize[d];
			    if (point(d) < m_minimum[d] || point(d) > m_maximum[d])
				{
				    compute = false;
				    break;
				}
			}

		    // evaluate the grid point
		    if (compute && objectiveFunction.isFeasible(point))
			{
			    double error = objectiveFunction.eval(point);

			    // remember the best solution
			    if (error < m_best.value)
				{
				    m_best.value = error;
				    m_best.point=point;
				}
			}
		    // move to the next grid point
		    size_t d = 0;
		    for (; d < dimensions; d++)
			{
			    index[d]++;
			    if (index[d] <= 4) break;
			    index[d] = 0;
			}
		    if (d == dimensions) break;
		}
	    // decrease the step sizes
	    BOOST_FOREACH(double& step,m_stepsize)
		step *= 0.5;
	}

    protected:
	//! minimum parameter value to check
	std::vector<double> m_minimum;

	//! maximum parameter value to check
	std::vector<double> m_maximum;

	//! current step size for every parameter
	std::vector<double> m_stepsize;

	bool m_configured;
    };


    //!
    //! \brief Optimize by trying out predefined configurations
    //!
    //! \par
    //! The PointSearch class is similair to the GridSearch class
    //! by the property that it optimizes a model in a single pass
    //! just trying out a predefined number of parameter configurations.
    //! The main difference is that every parameter configuration has
    //! to be explicitly defined. It is not possible to define a set
    //! of values for every axis; see GridSearch for this purpose.
    //! Thus, the PointSearch class allows for more flexibility.
    //!
    //! If no configure method is called, this class just samples random points.
    //! They are uniformly distributed in [-1,1].
    //! parameters^2 points but minimum 20 are sampled in this case.
    //!
    class PointSearch : public AbstractSingleObjectiveOptimizer<RealVector >
    {
    public:
	//! Constructor
	PointSearch() {
	    m_configured=false;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "PointSearch"; }

	//! Initialization of the search points.
	//! \param  points  array of points to evaluate
	void configure(const std::vector<RealVector>& points) {
	    m_points=points;
	    m_configured=true;
	}

	//!samples random points in the range [min,max]^parameters
	void configure(size_t parameters,size_t samples, double min,double max) {
	    RANGE_CHECK(min<=max);
	    m_points.resize(samples);
	    for(size_t sample=0;sample!=samples;++sample)
		{
		    m_points[sample].resize(parameters);
		    for(size_t param=0;param!=parameters;++param)
			{
			    m_points[sample](param)=Rng::uni(min,max);
			}
		}
	    m_configured=true;
	}

	virtual void read( InArchive & archive )
	{
	    archive>>m_points;
	    archive>>m_configured;
	    archive>>m_best.point;
	    archive>>m_best.value;
	}

	virtual void write( OutArchive & archive ) const
	{
	    archive<<m_points;
	    archive<<m_configured;
	    archive<<m_best.point;
	    archive<<m_best.value;
	}

	//! If the class wasn't configured before, this method samples random uniform distributed points in [-1,1]^n.
	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint) {
	    (void) objectiveFunction;

	    if(!m_configured)
		{
		    size_t parameters=startingPoint.size();
		    size_t samples=std::min(sqr(parameters),(size_t)20);
		    configure(parameters,samples,-1,1);
		}
	}
	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	//! Please note that for the point search optimizer it does
	//! not make sense to call step more than once, as the
	//! solution does not improve iteratively.
	void step(const ObjectiveFunctionType& objectiveFunction) {
	    size_t numPoints = m_points.size();
	    m_best.value = 1e100;
	    size_t bestIndex=0;

	    // loop through all points
	    for (size_t point = 0; point < numPoints; point++)
		{
		    // evaluate the model
		    if (objectiveFunction.isFeasible(m_points[point]))
			{
			    double error = objectiveFunction.eval(m_points[point]);
			    if (error < m_best.value)
				{
				    m_best.value = error;
				    bestIndex=point;
				}
			}
		}
	    m_best.point=m_points[bestIndex];
	}

    protected:
	//! The array holds one parameter configuration in every column.
	std::vector<RealVector> m_points;

	//! verbosity level
	bool m_configured;
    };


}
#endif
