//===========================================================================
/*!
 *  \file ObjectiveFunction.h
 *
 *  \brief General objective function class
 *
 *  \author  Christian Igel, Tobias Glasmachers
 *  \date    2008
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *
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
 *
 *
 */
//===========================================================================

#ifndef OBJECTIVE_FUNCTION_H
#define OBJECTIVE_FUNCTION_H

#include <Rng/GlobalRng.h>
#include <SharkDefs.h>
#include <vector>
#include <string>  
#include <Array/ArrayOp.h>
#include <Array/ArrayIo.h>
#include <LinAlg/VecMat.h>


//! \brief Constraint handler class for objective functions
//!
//! \par
//! The ConstraintHandler class knows two types of simple
//! constraint handling techniques. First, it can tell
//! whether a given point is feasible. If the point is not
//! feasible, this information allows the evolutionary
//! alrorithm to re-sample the point.
//!
//! \par
//! The second type of constraint handling is optional,
//! that is, it is not required by all constraint handlers.
//! The handler may be asked to return the closest feasible
//! point, given a non-feasible candidate. This leads to a
//! modified and often beneficial change of the sampling of
//! points at the boundary of the feasible region.
template<class T>
class ConstraintHandler
{
public:
	ConstraintHandler() { }
	virtual ~ConstraintHandler() { }

	virtual bool isFeasible(const T& point) const { return true; }
	virtual bool closestFeasible(T& point) const { return false; }
};


//! \brief Constraint handler for box regions
//!
//! Most simple non-trivial ConstraintHandler:
//! The feasible region is an axis parallel box.
//!
//! This class should be used as a base class for more
//! complicated feasible regions with known bounding box.
class BoxConstraintHandler : public ConstraintHandler<double*>
{
public:
	//! Constructor
	//!
	//! \param  dim    number of coordinates
	//! \param  lower  lower bound for all coordinates
	//! \param  upper  upper bound for all coordinates
	BoxConstraintHandler(unsigned int dim, double lower, double upper);

	//! Constructor
	//!
	//! \param  dim             number of coordinates
	//! \param  lower           lower bound for all coordinates
	//! \param  upper           upper bound for all coordinates
	//! \param  exception       coordinate index with different bounds
	//! \param  exceptionLower  lower bound for the exception coordinate
	//! \param  exceptionUpper  upper bound for the exception coordinate
	BoxConstraintHandler(unsigned int dim, double lower, double upper, unsigned int exception, double exceptionLower, double exceptionUpper);

	//! \param  dim              number of coordinates
	//! \param  lower            lower bound for all coordinates
	//! \param  upper            upper bound for all coordinates
	//! \param  exception1       first coordinate index with different bounds
	//! \param  exception1Lower  lower bound for the first exception coordinate
	//! \param  exception1Upper  upper bound for the first exception coordinate
	//! \param  exception2       second coordinate index with different bounds
	//! \param  exception2Lower  lower bound for the second exception coordinate
	//! \param  exception2Upper  upper bound for the second exception coordinate
	BoxConstraintHandler(unsigned int dim, double lower, double upper, unsigned int exception1, double exception1Lower, double exception1Upper, unsigned int exception2, double exception2Lower, double exception2Upper);

	//! Constructor
	//!
	//! \param  dim    number of coordinates
	//! \param  lower  component wise lower bound
	//! \param  upper  component wise upper bound
	BoxConstraintHandler(const std::vector<double>& lower, const std::vector<double>& upper);

	//! Destructor
	~BoxConstraintHandler();


	//! checks whether the point is inside the box
	bool isFeasible(double* const& point) const;

	//! returns the closest feasible point in the box
	bool closestFeasible(double*& point) const;

	//! access to the search space dimension
	inline unsigned int dimension() const { return m_dimension; }

	//! access to the box bounds
	inline double lowerBound(unsigned int index) const { return m_lower[index]; }

	//! access to the box bounds
	inline double upperBound(unsigned int index) const { return m_upper[index]; }

protected:
	//! dimension of the search space
	unsigned int m_dimension;

	//! component wise lower bounds
	std::vector<double> m_lower;

	//! component wise upper bounds
	std::vector<double> m_upper;
};


//! \brief Base class of all objective functions
//!
//! \par
//! The ObjectiveFunction base class does not provide any
//! interface for function evaluations. It just holds a
//! variable to count the number of function evaluations.
//! The main purpose of this class is to serve as a general
//! non-template superclass of all objective functions,
//! such that any objective function can be cast into this
//! type. As a convenience feature each objective function
//! can be given a name. This should be done in the
//! constructor of a sub-class.
class ObjectiveFunction {
public:
	//! Constructor
	ObjectiveFunction();

	//! Destructor
	virtual ~ObjectiveFunction();


	//! return the number of function evaluations so far
	inline unsigned timesCalled() const { return m_timesCalled; }

	//! reset the number of function evaluations to zero
	inline void resetTimesCalled() { m_timesCalled = 0; }

	//! return the name of this functions
	inline const std::string& name() const { return m_name; }

	//! return the number of objectives to optimize
	virtual unsigned int objectives() const = 0;

protected:
	//! number of function evaluations
	unsigned m_timesCalled;

	//! function name
	std::string m_name;
};


//! \brief Single- or multi-objective fitness function on an arbitrary search space
//!
//! \par
//! The template class ObjectiveFunctionT is the base class
//! of all objective function implementations. Its template
//! parameter defines the search space or domain of the function.
//! The image can consist of the reals for a standard single
//! objective task, or of a real vector space for multi-objective
//! problems.
//!
//! \par
//! Constraints handling is also incorporated at this level.
//! It is possible to check the feasibility of a search point.
//! As an option the objective function can support the search
//! algorithm by providing the closest feasible point for a
//! non-feasible candidate.
//!
//! \par
//! Some standard benchmark problems come with a fixed
//! distribution of starting points for the search algorithm.
//! This feature is also supported, but it remains optional.
template<class T>
class ObjectiveFunctionT : public ObjectiveFunction {
public:
	//! Constructor
	ObjectiveFunctionT(ConstraintHandler<T>* constrainthandler = NULL)
	{
		this->constrainthandler = constrainthandler;
	}

	//! Descructor
	~ObjectiveFunctionT() { }

	//! check the feasibility of #point
	bool isFeasible(const T& point) const
	{
		if (constrainthandler == NULL) return true;
		else return constrainthandler->isFeasible(point);
	}

	//! Replace #point with the closest feasible point.
	//! Return false if this operation is not supported.
	bool closestFeasible(T& point) const
	{
		if (constrainthandler == NULL) return false;
		else return constrainthandler->closestFeasible(point);
	}

	//! objective function evaluation
	virtual void result(const T& point, std::vector<double>& value) = 0;

	//! objective function evaluation in the single objective case
	double operator() (const T& point) {
		std::vector<double> value(1);
		result(point, value);
		return value[0];
	}

	//! The objective function can propose a (random)
	//! starting point for optimization runs. This is
	//! useful for some standard benchmarks. If the
	//! function returns false, then no starting point
	//! is proposed, such that subclasses are not forced
	//! to provide this information.
	virtual bool ProposeStartingPoint(T& point) const
	{
		return false;
	}

	//! return the constraint handler or NULL
	inline const ConstraintHandler<T>* getConstraintHandler() const
	{ return constrainthandler; }

protected:
	//! registered constraint handler or NULL
	ConstraintHandler<T>* constrainthandler;
};


//! \brief Objective function on a vector space (VS)
//!
//! \par
//! The important special case that the search space is
//! embedded in a real vector space is captured by this class.
//! Simple c-arrays, vectors, or Shark Arrays can be used to
//! represent points in the search space.
template<class T>
class ObjectiveFunctionVS : public ObjectiveFunctionT<T*> {
public:
	//! Constructor
	ObjectiveFunctionVS(unsigned d = 0, ConstraintHandler<T*>* constrainthandler = NULL)
	: ObjectiveFunctionT<T*>(constrainthandler)
	, m_dimension(d)
	{ };

	//! Destructor
	~ObjectiveFunctionVS() { };


	//! Define the search space dimension
	void init(unsigned d) {
		m_dimension = d;
	}

	//! Retreive the search space dimension
	inline unsigned dimension() const { return m_dimension; }

	//! overloaded evaluation of a single objective function
	double operator()(const Array<T>& point) {
		SIZE_CHECK(point.ndim() == 1 || point.dim(0) == m_dimension);
		return (*this)(const_cast<T*>(&point(0)));
	}

	//! overloaded evaluation of a single objective function
	double operator()(const std::vector<T>& point) {
		SIZE_CHECK(point.size() == m_dimension);
		return (*this)(const_cast<T*>(&point[0]));
	}

	//! evaluation of a single objective function
	double operator() (double* const& point) {
		// it makes no sence - but re-declaring this operator
		// avoids stupid compiler errors :(
		std::vector<double> value(1);
		((ObjectiveFunctionT<const T*>*)this)->result(point, value);
		return value[0];
	}

	//! objective function evaluation
	void result(const Array<T>& point, std::vector<double>& value) {
		SIZE_CHECK(point.ndim() == 1 || point.dim(0) == m_dimension);
		((ObjectiveFunctionT<T*>*)this)->result((T* const&)const_cast<T*>(&point(0)), value);
	}

	//! objective function evaluation
	void result(const std::vector<T>& point, std::vector<double>& value) {
		SIZE_CHECK(point.size() == m_dimension);
		((ObjectiveFunctionT<T*>*)this)->result(const_cast<T*>(&point[0]), value);
	}

	//! For a BoxConstraintHandler we can automatically
	//! propose a starting point.
	bool ProposeStartingPoint(T*& point) const
	{
		if (ObjectiveFunctionT<T*>::constrainthandler == NULL) return false;
		BoxConstraintHandler* bch = dynamic_cast<BoxConstraintHandler*>(ObjectiveFunctionT<T*>::constrainthandler);
		if (bch == NULL) return false;
		int i, ic = bch->dimension();
		while (true)
		{
			for (i=0; i<ic; i++) point[i] = (T)Rng::uni(bch->lowerBound(i), bch->upperBound(i));
			if (bch->isFeasible((double*&)point)) break;
		}
		return true;
	}

	//! feasibility check
	inline bool isFeasible(const Array<T>& point) const {
		SIZE_CHECK(point.ndim() == 1 || point.dim(0) == m_dimension);
		return ((ObjectiveFunctionT<T*>*)this)->isFeasible((T* const&)const_cast<T*>(&point(0)));
	}

	//! feasibility check
	inline bool isFeasible(const std::vector<T>& point) const {
		SIZE_CHECK(point.size() == m_dimension);
		return ((ObjectiveFunctionT<T*>*)this)->isFeasible((T* const&)const_cast<T*>(&point[0]));
	}

	//! if possible, return the closest feasible point
	inline bool closestFeasible(Array<T>& point) const {
		SIZE_CHECK(point.ndim() == 1 || point.dim(0) == m_dimension);
		return ObjectiveFunctionT<T*>::ClosestFeasible(&point(0));
	}

	//! if possible, return the closest feasible point
	inline bool closestFeasible(std::vector<T>& point) const {
		SIZE_CHECK(point.size() == m_dimension);
		T* p = &point[0];
		return ObjectiveFunctionT<T*>::closestFeasible(p);
	}

	//! if possible, propose a starting point
	inline bool ProposeStartingPoint(Array<T>& point) const {
		SIZE_CHECK(point.ndim() == 1 || point.dim(0) == m_dimension);
		T* p = &point(0);
		return ProposeStartingPoint(p);
	}

	//! if possible, propose a starting point
	inline bool ProposeStartingPoint(std::vector<T>& point) const {
		SIZE_CHECK(point.size() == m_dimension);
		T* p = &point[0];
		return ProposeStartingPoint(p);
	}

	//! If possible, this method returns (a bound on) the best
	//! possible fitness. For the single objective case the
	//! function should return the fitness in the global optimum.
	//! For the multi objective case this function should return
	//! a utopian point.
	//!
	//! \param  fitness  component wise utopian fitness
	//! \return          true if an utopian point is known, false otherwise
	virtual bool utopianFitness(std::vector<double>& value) const { return false; }

	//! If possible, this method returns (a bound on) the worst
	//! possible fitness. For single objective optimization this
	//! information is rarely of any use, while for multi objective
	//! optimization a so-called nadir point is important for the
	//! computation of standard quality indicators like the
	//! hypervolume.
	//!
	//! \param  fitness  component wise worst fitness
	//! \return          true if a nadir point is known, false otherwise
	virtual bool nadirFitness(std::vector<double>& value) const { return false; }

protected:
	//! search space dimension
	unsigned m_dimension;
};


//! \brief objective function with a linear transformation applied to the search space \f$ R^n \f$
//!
//! \par
//! This specialized class represents an objective function
//! with a linear transformation of the search vector space.
class TransformedObjectiveFunction : public ObjectiveFunctionVS<double> {
public:
	//! Constructor
	TransformedObjectiveFunction(ObjectiveFunctionVS<double>& base, unsigned d = 0);

	//! Constructor
	TransformedObjectiveFunction(ObjectiveFunctionVS<double>& base, const Array<double>& transformation);

	//! Destructor
	~TransformedObjectiveFunction();

	//! Explicitly set the transformation
	void init(const Array<double>& transformation);

	//! Create a random orthogonal transformation
	void initRandomRotation(unsigned d = 0);

	//! Retreive the transformation matrix
	inline const Matrix& Transformation() const { return m_Transformation; }

	//! objective function evaluation
	inline void result(const Array<double>& point, std::vector<double>& value)
	{
		SIZE_CHECK(point.ndim() == 1 || point.dim(0) == m_dimension);
		result((double*)&point(0), value);
	}

	//! objective function evaluation
	inline void result(const std::vector<double>& point, std::vector<double>& value)
	{
		SIZE_CHECK(point.size() == m_dimension);
		result((double*)&point[0], value);
	}

	//! objective function evaluation
	void result(double* const& point, std::vector<double>& value);

	//! propose a transformed starting point
	bool ProposeStartingPoint(double*& point) const;

	//! #point is multiplied with m_Transformation
	inline void transform(Array<double>& point) const {
		SIZE_CHECK(point.ndim() == 1 || point.dim(0) == m_dimension);
		std::vector<double> tmp(m_dimension);
		transform(&point(0), tmp);
		unsigned i;
		for (i=0; i<m_dimension; i++) point(i) = tmp[i];
	}

	//! #point is multiplied with m_Transformation
	inline void transform(std::vector<double>& point) const {
		SIZE_CHECK(point.size() == m_dimension);
		std::vector<double> tmp(m_dimension);
		transform(&point[0], tmp);
		unsigned i;
		for (i=0; i<m_dimension; i++) point[i] = tmp[i];
	}

	//! #point is multiplied with m_Transformation
	inline void transform(double* point) const {
		std::vector<double> tmp(m_dimension);
		transform(point, tmp);
		unsigned i;
		for (i=0; i<m_dimension; i++) point[i] = tmp[i];
	}

	//! call to utopianFitness of the base objective function
	bool utopianFitness(std::vector<double>& value) const;

	//! call to nadirFitness of the base objective function
	bool nadirFitness(std::vector<double>& value) const;

	//! return the number of objectives to optimize
	unsigned int objectives() const;

protected:
	//! transform #in, resulting in #out
	void transform(const double* in, std::vector<double>& out) const;

	//! inverse transform #in, resulting in #out
	void transformInverse(const std::vector<double>& in, double* out) const;

	//! non-transformed objective function
	ObjectiveFunctionVS<double>& baseObjective;

	//! transformation matrix
	Matrix m_Transformation;
};


#endif
