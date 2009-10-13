//===========================================================================
/*!
 *  \file ObjectiveFunctions.h
 *
 *  \brief standard benchmark functions
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

#ifndef OBJECTIVE_FUNCTIONS_H
#define OBJECTIVE_FUNCTIONS_H

#include <EALib/ObjectiveFunction.h>


/*!
 * \brief Sphere function
 *
 * Rotation invariant Paraboloid f(x) = &lt;x, x&gt;.
 *
 */
class Sphere : public ObjectiveFunctionVS<double> {
public:
	//! Constructor
	Sphere(unsigned d);

	//! Destructor
	~Sphere();

	unsigned int objectives() const;
	void result(double* const& point, std::vector<double>& value);
	bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
};

/*!
 * \brief Noisy Sphere function
 *
 * \f[
 * 		f_{NoisySphere}(\vec{x}) = \sum_{i=1}^{n} x_i^2 + \frac{1}{2n}\xi\sum_{i=1}^{n} x_i^2
 * \f]
 * where \f$ \xi \f$ denotes a random variable drawn according to the 
 *   standard Cauchy distribution.
 */
class NoisySphere : public ObjectiveFunctionVS<double> {
public:
	//! Constructor
	NoisySphere(unsigned d);

	//! Destructor
	~NoisySphere();

	unsigned int objectives() const;
	void result(double* const& point, std::vector<double>& value);
	bool ProposeStartingPoint(double*& point) const;
};


/*!
 * \brief Sheered Paraboloid (Ellipsoid)
 *
 * Sheered Paraboloid f(x) = &lt;Mx, Mx&gt;,
 * the conditioning number of the matrix M can
 * be controlled.
 *
 * \f[
 *     f_{elli}(x) = \sum_{i=1}^{n}c^{\frac{i-1}{n-1}}  y_i^2
 * \f]
 *
 * where c is a given constant (usually \f$ c=1000 \f$) and y is defined by:
 * \f[
 *     y= Ox
 * \f]
 * where O is an arbitrary orthogonal rotation matrix.  
 */
class Paraboloid : public TransformedObjectiveFunction {
public:
	//! Constructor
	Paraboloid(unsigned d, double c = 1000.0);

	//! Destructor
	~Paraboloid();

// 	unsigned int objectives() const;
// 	bool ProposeStartingPoint(double*& point) const;

protected:
	//! non-transformed base model
	Sphere base;
};


/*!
 * \brief Tablet function
 *
 * The Tablet function:
 *
 * \f[
 *    f_{Tablet}(x) = (c*y_1)^2 + \sum_{i=2}^{n} y_i^2
 * \f]
 * where c is a given constant (usually \f$ c=1000 \f$) and y is defined by:
 * \f[
 *     y= Ox
 * \f]
 * where O is an arbitrary orthogonal rotation matrix.  
 */

class Tablet : public TransformedObjectiveFunction {
public:
	//! Constructor
	Tablet( unsigned d, double c = 1000.0 );

	//! Destructor
	~Tablet();

// 	unsigned int objectives() const;
// 	bool ProposeStartingPoint(double*& point) const;

protected:
	//! non-transformed base model
	Sphere base;
};


/*!
 * \brief Cigar function
 *
 * The Cigar function:
 *
 * \f[
 *    f_{Cigar}(x) = y_1^2 + \sum_{i=2}^{n}(c *y_i)^2
 * \f]
 * where c is a given constant (usually \f$ c=1000 \f$) and y is defined by:
 * \f[
 *     y= Ox
 * \f]
 * where O is an arbitrary orthogonal rotation matrix.  
 */
class Cigar : public TransformedObjectiveFunction {
public:
     //! Constructor
	Cigar( unsigned d, double c = 1000.0);

	//! Destructor
	~Cigar();

// 	unsigned int objectives() const;
// 	bool ProposeStartingPoint(double*& point) const;

protected:
	//! non-transformed base model
	Sphere base;
};


/*!
 * \brief Twoaxis function
 *
 * \brief The Twoaxis function:
 *
 * \f[
 *     f_{Twoaxis}(x)= \sum_{i=1}^{\lfloor\frac{n}{2}\rfloor} c x_1^2 + \sum_{i=\lceil\frac{n}{2}\rceil}^{n} x_i^2
 * \f]
 * where c is a given constant (usually \f$ c=1000 \f$) and y is defined by:
 * \f[
 *     y= Ox
 * \f]
 * where O is an arbitrary orthogonal rotation matrix.
 */
class Twoaxis : public TransformedObjectiveFunction {
public:
     //! Constructor
	Twoaxis( unsigned d, double c = 1000.0);

	//! Destructor
	~Twoaxis();

// 	unsigned int objectives() const;
// 	bool ProposeStartingPoint(double*& point) const;

protected:
	//! non-transformed base model
	Sphere base;
};


 


/*!
 * \brief Generalized Ackley's function
 *  
 * The generalized Ackley's function:
 *
 *  \f[
 *       f_{Ackley}(x)=-20\cdot \exp\left(-0.2\sqrt{\frac{1}{n}\cdot\sum\limits_{i=1}^{n}x_{i}^{2}}\right)\\
 *                      - \exp\left(\frac{1}{n}\cdot\sum\limits_{i=1}^{n}\cos(2\pi
 *                      x_{i})\right) + 20 + \exp(1)\enspace
 *  \f] 
 *   where
 *  \f[
 *      -32 \leq x_{i} \leq 32
 *  \f]
 */
class Ackley : public ObjectiveFunctionVS<double> {
public:
	//! Constructor
	Ackley( unsigned d = 30 ); 

	//! Destructor
	~Ackley();

	unsigned int objectives() const;
	void result(double* const& point, std::vector<double>& value);
	bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& fitness) const;
};



/*!
 * \brief Generalized Rastrigin's function
 *
 * The generalized Rastrigin's function:
 *
 * \f[
 *      f_{Rastrigin}(x)= 10n + \sum_{i=1}^{n}\big(x_i^2-10 \cos(2\pi x_i)\big)\enspace 
 * \f]
 *   where
 *  \f[
 *      -5.12 \leq x_{i} \leq 5.12
 *  \f]
 */
class Rastrigin : public ObjectiveFunctionVS<double> {
public:
	//! Constructor
	Rastrigin( unsigned d = 10 );

	//! Destructor
	~Rastrigin();

	unsigned int objectives() const;
	void result(double* const& point, std::vector<double>& value);
	bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& fitness) const;
};



/*!
 * \brief Generalized Griewangk's function
 *
 * The generalized Griewangk's function:
 *
 *  \f[
 *      f_{Griewangk}(x)= 1 + \sum\limits_{i = 1}^n \frac{x_{i}^{2}}{4000}-\prod\limits_{i = 1}^n \cos(\frac{x_{i}}{\sqrt{i}})\enspace
 *  \f]
 *   where
 *  \f[
 *      -600 \leq x_{i} \leq 600
 *  \f]
 */
class Griewangk : public ObjectiveFunctionVS<double> {
public:
	//! Constructor
	Griewangk( unsigned d = 30 );

	//! Destructor
	~Griewangk();

	unsigned int objectives() const;
	void result(double* const& point, std::vector<double>& value);
	bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& fitness) const;
};


/*!
 * \brief Generalized Rosenbrock's function 
 *
 * The Generalized Rosenbrock's function:
 *  
 * \f[ 
 *      f_{Rosenbrock}(x)= \sum_{i=1}^{n-1}\left(100\cdot(x_i^2-x_{i+1})^2+(x_i-1)^2\right)
 * \f]
 */
class Rosenbrock : public ObjectiveFunctionVS<double> {
public:
	//! Constructor
	Rosenbrock( unsigned d = 29 );

	//! Destructor
	~Rosenbrock();

	unsigned int objectives() const;
	void result(double* const& point, std::vector<double>& value);
	bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& fitness) const;
};


/*!
 * \brief Generalized Rosenbrock's function 
 *
 * The Generalized Rosenbrock's function with a linear transformation applied to the search space :
 *  
 * \f[ 
 *      f_{Rosenbrock}(x)= \sum_{i=1}^{n-1}\left(100\cdot(y_i^2-y_{i+1})^2+(y_i-1)^2\right)
 * \f]
 * where y is defined by:
 * \f[
 *     y= Ox
 * \f]
 * where O is an arbitrary orthogonal rotation matrix.
 */
class RosenbrockRotated : public TransformedObjectiveFunction {
public:
	//! Constructor
	RosenbrockRotated( unsigned d = 29 );

	//! Destructor
	~RosenbrockRotated();

// 	unsigned int objectives() const;
// 	bool ProposeStartingPoint(double*& point) const;

protected:
	//! non-transformed base model
	Rosenbrock base;
};




/*!
 * \brief Different powers function
 * 
 * The Different powers function:
 *
 * \f[
 *      f_{diff pow}(x)= \sum_{i=1}^{n} |x_i|^{2+10*\frac{i-1}{n-1}}
 * \f]
 */

class DiffPow : public ObjectiveFunctionVS<double> {
public:
	//! Constructor
	DiffPow( unsigned d );

	//! Destructor
	~DiffPow();

	unsigned int objectives() const;
	void result(double* const& point, std::vector<double>& value);
	bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& fitness) const;
};



/*!
 * \brief Different powers function
 * 
 * The DiffPow function with a linear transformation applied to the search space:
 *
 * \f[
 *      f_{diff pow}(x)= \sum_{i=1}^{n} |y_i|^{2+10*\frac{i-1}{n-1}}
 * \f]
 * where y is defined by:
 * \f[
 *     y= Ox
 * \f]
 * where O is an arbitrary orthogonal rotation matrix.
 */
class DiffPowRotated : public TransformedObjectiveFunction {
public:
	//! Constructor
	DiffPowRotated(unsigned d);

	//! Destructor
	~DiffPowRotated();

// 	unsigned int objectives() const;
// 	bool ProposeStartingPoint(double*& point) const;

protected:
	//! non-transformed base model
	DiffPow base;
};




/*!
 * \brief Generalized Schwefel's problem
 *
 * The Generalized Schwefel's problem is given by:
 *
 * \f[
 *      f_{Schwefel}(x)= \sum_{i=1}^{n-1}\left(x_1\cdot\sin{(\sqrt{\left| x_i \right|}})\right) 
 * \f]
 */
class Schwefel : public ObjectiveFunctionVS<double> {
public:
	//! Constructor
	Schwefel( unsigned d );

	//! Destructor
	~Schwefel();

	unsigned int objectives() const;
	void result(double* const& point, std::vector<double>& value);
	bool ProposeStartingPoint(double*& point) const;
};



/*!
 * \brief Schwefel's ellipsoid function
 *
 * Schwefel's ellipsoid function:
 *
 * \f[
 *     f_{Schwefel}(x)= \sum_{i=1}^{n}(\sum_{j=1}^{i} x_j)^2
 * \f]
 */
class SchwefelEllipsoid : public ObjectiveFunctionVS<double> {
public:
	//! Constructor
	SchwefelEllipsoid(unsigned d);

	//! Destructor
	~SchwefelEllipsoid();

	unsigned int objectives() const;
	void result(double* const& point, std::vector<double>& value);
	bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& fitness) const;
};


/*!
 * \brief Schwefel's ellipsoid function.
 *
 * Schwefel's ellipsoid function with a linear transformation applied to the search space:
 *
 * \f[
 *     f_{Schwefel}(x)= \sum_{i=1}^{n}(\sum_{j=1}^{i} y_j)^2
 * \f]
 * where y is defined by:
 * \f[
 *     y= Ox
 * \f]
 * where O is an arbitrary orthogonal rotation matrix.
 */
class SchwefelEllipsoidRotated : public TransformedObjectiveFunction {
public:
	//! Constructor
	SchwefelEllipsoidRotated(unsigned d);

	//! Destructor
	~SchwefelEllipsoidRotated();

// 	unsigned int objectives() const;
// 	bool ProposeStartingPoint(double*& point) const;

protected:
	//! non-transformed base model
	SchwefelEllipsoid base;
};


//!
//! Random fitness function
//!
class RandomFitness : public ObjectiveFunctionVS<double> {
public:
	//! Constructor
	RandomFitness( unsigned d );
	
	//! Destructor
	~RandomFitness();

	unsigned int objectives() const;
	void result( double* const& point, std::vector<double>& value );
	bool ProposeStartingPoint(double*& point) const;
};


#endif
