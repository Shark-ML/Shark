//===========================================================================
/*!
 *  \file MultiObjectiveFunctions.h
 *
 *  \brief standard multi-objective benchmark functions
 *
 *  \author  Asja Fischer
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


#ifndef MULTIOBJECTIVEFUNCTIONS_H_
#define MULTIOBJECTIVEFUNCTIONS_H_


#include <EALib/ObjectiveFunction.h>


/*!
 * \brief ConstraintHandler suitable for BhinKorn
 */

class BhinKornConstraintHandler : public ConstraintHandler<double*>
{
public:
	BhinKornConstraintHandler() { }
	~BhinKornConstraintHandler() { }

	bool isFeasible(double* const& point) const;

protected:
	unsigned int m_dimension;
	double m_lower;
	double m_upper;
};



/*!
 * \brief Bhin and Korn 1
 *  
 * \f[
 *     f_{1}(x) = (x_{1} -2)^{2} + (x_{2}-1)^{2} +2
 * \f]
 * \f[
 *     f_{2}(x) = 9x_{1} - (x_{2}-1)^{2}
 * \f]
 * where the solutions are subject to the following constraints
 * \f[
 *     x_{1}^{2} + x_{2}^{2} - 255 \leq 0
 * \f]
 * \f[
 *     x_{1}+ 3x_{2} +10 \leq 0
 * \f]
 * and
 * \f[
 *    -20 \leq x_i \leq 20
 * \f]
 */
class BhinKorn : public  ObjectiveFunctionVS<double> {
public:
	//! Constructor
	BhinKorn();

	//! Destructor
	~BhinKorn();

	unsigned int objectives() const;
	void result(double* const& point, std::vector<double>& value);
	bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};




/*!
 * \brief Schaffer`s f2
 *  
 * \f[
 *     f_{2}(x)=g(x)+h(x)
 * \f]
 * where
 * \f[
 *     g(x)=x^2
 * \f]
 * and
 * \f[
 *     h(x)=(x-2)^2
 * \f]
 * \f[
 *     -6 \leq x \leq 6
 * \f]
 * 
 */
class Schaffer : public  ObjectiveFunctionVS<double> {
public:
	//! Constructor
	Schaffer();

	//! Destructor
	~Schaffer();

	unsigned int objectives() const;
	void result(double* const& point, std::vector<double>& value);
	bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};



/*!
 * \brief multi-objective problem ZDT1
 *
 * \f[
 *      f_{1}(x)=x_1
 * \f]
 * \f[
 *      f_{2}(x)=g(x)( 1 - \sqrt{ \frac{ x_1 }{ g(x) } } )
 * \f]
 * where
 * \f[
 *      g(x)=1+\frac{9}{n-1} \sum_{i=2}^{n} x_i ^2
 * \f]
 * \f[
 *      0 \leq  x_i  \leq 1
 * \f]
 */
 
class ZDT1 : public ObjectiveFunctionVS<double> {
public:
	//! Constructor
	ZDT1( unsigned d );

	//! Destructor
	~ZDT1();

	unsigned int objectives() const;
	void result(double* const& point, std::vector<double>& value);
	bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem ZDT2
 *
 * \f[
 *      f_{1}(x)=x_1
 * \f]
 * \f[
 *      f_{2}(x)=g(x)( 1 - ( \frac{ x_1 }{ g(x) } )^2 )
 * \f]
 * where
 * \f[
 *      g(x)=1+\frac{9}{n-1} \sum_{i=2}^{n} x_i ^2
 * \f]
 * \f[
 *      0 \leq  x_i  \leq 1
 * \f]
 */
class ZDT2 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZDT2( unsigned d );

    //! Destructor
    ~ZDT2();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem ZDT3
 *
 * \f[
 *      f_{1}(x)=x_1
 * \f]
 * \f[
 *      f_{2}(x)=g(x)(1-\sqrt{\frac{x_1}{g(x)}}-\frac{x_1}{g(x)}\sin{(10 \pi x_1)})
 * \f]
 * where
 * \f[
 *      g(x)=1+\frac{9}{n-1} \sum_{i=2}^{n} x_i ^2
 * \f]
 * \f[
 *      0 \leq  x_i  \leq 1
 * \f]
 */
class ZDT3 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZDT3( unsigned d );

    //! Destructor
    ~ZDT3();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem ZDT4 
 *
 * \f[
 *      f_{1}(x)=x_1
 * \f]
 * \f[
 *      f_2({\bf x})=g({\bf x})(1-{(\frac{x_1}{g({\bf x})})}^2)
 * \f]
 * where
 * \f[
 *     g(x)=1+10(n-1)+\sum_{i=2}^{n}(x_i^2 -10\cos{(4 \pi x_i)})
 * \f]
 * \f$ 0 \leq  x_1  \leq 1  \f$ and \f$ x_2, \ldots, x_{10} \in [-5,5] \f$
 *
 */
class ZDT4 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZDT4( unsigned d );

    //! Destructor
    ~ZDT4();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


// /*!
//  * \brief ConstraintHandler for the feasible region being a cuboid.
//  *
//  * The feasible region is an n-dimensional cuboid with the
//  * side length in the first dimension differing from the side
//  * lengths in the other dimensions.
//  */
// class CuboidConstraintHandler : public ConstraintHandler<double*>
// {
// public:
// 	CuboidConstraintHandler( unsigned int dimension, double lower1, double upper1, double lower2, double upper2 )
// 	: m_dimension( dimension )
// 	, m_lower_1( lower1 )
// 	, m_upper_1( upper1 )
// 	, m_lower_2( lower2 )
// 	, m_upper_2( upper2 )
// 	{ }
// 
// 	~CuboidConstraintHandler() { }
// 
// 	bool isFeasible(double* const& point) const;
// 	bool closestFeasible(double*& point) const;
// 
// protected:
// 	unsigned int m_dimension;
// 	double m_lower_1;
// 	double m_upper_1;
// 	double m_lower_2;
// 	double m_upper_2;
// };

// /*!
//  * \brief ConstraintHandler suitable for LZ07_F6
//  *
//  * The feasible region is an n-dimensional cuboid
//  * with the side length in the first and second dimension
//  * differing from the side lengths in the other dimensions.
//  */
// class CuboidConstraintHandler_2 : public ConstraintHandler<double*>
// {
// public:
// 	CuboidConstraintHandler_2( unsigned int dimension, double lower1, double upper1, double lower2, double upper2 )
// 	: m_dimension( dimension )
// 	, m_lower_1( lower1 )
// 	, m_upper_1( upper1 )
// 	, m_lower_2( lower2 )
// 	, m_upper_2( upper2 )
// 	{ }
// 
// 	~CuboidConstraintHandler_2() { }
// 
// 	bool isFeasible(const double*& point) const;
// 	bool closestFeasible(double*& point) const;
// 
// protected:
// 	unsigned int m_dimension;
// 	double m_lower_1;
// 	double m_upper_1;
// 	double m_lower_2;
// 	double m_upper_2;
// };


/*!
 * \brief multi-objective problem ZDT5
 *
 * \f[
 *     f_1({\bf x}) = 1+u(x_1)
 * \f]
 * \f[
 *    f_2({\bf x})=g({\bf x})(\frac{1}{f_1(x)})
 * \f]
 * where 
 * \f[
 *   g({\bf x})=\sum_{i=2}^{11} v(u(x_i))
 * \f]
 * \f$ u(x_i)\f$ gives the number of ones in the bit vector \f$ x_i \f$ (unitation funcion), 
 * \f[
 *    v(u(x_i))=\left\{\begin{array}{ll} 2+u(x_i), & \mbox{if } u(x_i) \le 5  \\ 
 *    1, & \mbox{if} u(x_i)=5 \end{array}\right.
 * \f]
 * \f$ x_1 \in \{0,1\}^{30} \f$ and \f$ x_2, \ldots, x_{11} \in \{0,1\}^5 \f$
 * and usually \f$ n=11 \f$
 */
class ZDT5 : public ObjectiveFunctionVS<long> {
public:
	//! Constructor
	ZDT5( unsigned d = 11);

	//! Destructor
	~ZDT5();

	unsigned int objectives() const;
	void result( long* const& point, std::vector<double>& value );
	bool ProposeStartingPoint( long*& point ) const;

private:
	unsigned u( long x ) const;
};


/*!
 * \brief ConstraintHandler suitable for ZDT5
 *
 * ConstraintHandler for ZDT5 where the decision variable space
 * is made up of bit strings. The feasible region is an
 * n-dimensional cuboid with the side length in the first
 * dimension differing from the side lengths in the other
 * dimensions.
 */
class ZDT5ConstraintHandler : public ConstraintHandler<long*>
{
public:
	ZDT5ConstraintHandler( unsigned int dimension, long lower1, long upper1, long lower2, long upper2 )
	: m_dimension( dimension )
	, m_lower_1( lower1 )
	, m_upper_1( upper1 )
	, m_lower_2( lower2 )
	, m_upper_2( upper2 )
	{ }

	~ZDT5ConstraintHandler() { }

	bool isFeasible(long* const& point) const;
	bool closestFeasible(long*& point) const;

protected:
	unsigned int m_dimension;
	long m_lower_1;
	long m_upper_1;
	long m_lower_2;
	long m_upper_2;
};


/*!
 * \brief multi-objective problem ZDT6
 *
 * \f[
 *      f_{1}(x)=1-\exp (-4 x_1) \sin^6 (6 \pi x_1)
 * \f]
 * \f[
 *      f_{2}(x)=g(x)( 1 - ( \frac{f_1(x)}{g(x)} )^2 )
 * \f]
 * where
 * \f[
 *      g(x)=1+9*( \sum_{i=2}^{n} \frac{x_i}{n-1})^{0.25}
 * \f]
 * \f[
 *      0 \leq  x_i  \leq 1
 * \f]
 */
class ZDT6 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZDT6( unsigned d );

    //! Destructor
    ~ZDT6();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem IHR1.
 *
 *  The rotated variant of the ZDT1 problem:
 *
 * \f[
 *      f_{1}(x)= |y_1|
 * \f]
 * \f[
 *      f_{2}(x)=g(y)h_f( 1 - \sqrt{ \frac{h(y_1) }{ g(y) } } )
 * \f]
 * \f[
 *      g(y)=1+\frac{9 \sum_{i=2}^{n}h_g(y_i)}{(n-1)} 
 * \f]
 * with auxiliary functions:
 * \f[
 *     h: \mathbf{R} \rightarrow [0,1], x \rightarrow (1+exp(\frac{-x}{ \sqrt{n} }))^{-1}
 * \f]
 * \f[
 *     h_f: \mathbf{R} \rightarrow \mathbf{R}, x \rightarrow \left\{\begin{array}{ll} x, & \mbox{if } |y_1|\leq y_{max}  \\ 
 *          1+|y_1|, & \mbox{ otherwise} \end{array}\right.
 * \f]
 * \f[
 *     h_g: \mathbf{R} \rightarrow \mathbf{R}_{\leq 0}, x \rightarrow \frac{x^2}{|x|+0.1}
 * \f]
 * where
 * \f$  x\in [0,1]^n \f$,
 * y is defined by \f$ y= Ox \f$ 
 * where O is an arbitrary orthogonal rotation matrix and 
 * \f$ y_{max} = 1/max_j (|o_{1j}|)\f$
 * and usually \f$ n=10 \f$
 */
class IHR1 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    IHR1( unsigned d=10 );

    //! Destructor
    ~IHR1();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;

	//! Create a random orthogonal transformation
	void initRandomRotation();

protected:
	//! transform #in, resulting in #out
	void transform(const double* in, std::vector<double>& out) const;
    double h(double x, double n);
	double hf(double x, double y0, double ymax);
	double hg(double x);

	//! transformation matrix
	Matrix m_Transformation;
};


/*!
 * \brief multi-objective problem IHR2.
 *
 *  The rotated variant of the ZDT2 problem:
 *
 * \f[
 *      f_{1}(x)= |y_1|
 * \f]
 * \f[
 *      f_{2}(x)=g(y)h_f( 1 - (\frac{y_1}{ g(y) } )^2 )
 * \f]
 * \f[
 *      g(y)=1+\frac{9 \sum_{i=2}^{n}h_g(y_i)}{(n-1)} 
 * \f]
 * with auxiliary functions:
 * \f[
 *     h_f: \mathbf{R} \rightarrow \mathbf{R}, x \rightarrow \left\{\begin{array}{ll} x, & \mbox{if } |y_1|\leq y_{max}  \\ 
 *          1+|y_1|, & \mbox{ otherwise} \end{array}\right.
 * \f]
 * \f[
 *     h_g: \mathbf{R} \rightarrow \mathbf{R}_{\leq 0}, x \rightarrow \frac{x^2}{|x|+0.1}
 * \f]
 * where
 * \f$  x\in [0,1]^n \f$ ,
 * y is defined by \f$ y= Ox \f$ 
 * where O is an arbitrary orthogonal rotation matrix and 
 * \f$ y_{max} = 1/max_j (|o_{1j}|)\f$
 * and usually \f$ n=10 \f$
 */
class IHR2 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    IHR2( unsigned d =10);

    //! Destructor
    ~IHR2();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
    
	//! Create a random orthogonal transformation
	void initRandomRotation();

protected:
	//! transform #in, resulting in #out
	void transform(const double* in, std::vector<double>& out) const;
	double hf(double x, double y0, double ymax);
	double hg(double x);

	//! transformation matrix
	Matrix m_Transformation;
};


/*!
 * \brief multi-objective problem IHR3.
 *
 *  The rotated variant of the ZDT3 problem:
 *
 * \f[
 *      f_{1}(x)= |y_1|
 * \f]
 * \f[
 *      f_{2}(x)=g(y)h_f( 1 - \sqrt{\frac{y_1}{ g(y) }} - \frac{h(y_1)}{g(y)}sin(10 \pi y_1))
 * \f]
 * \f[
 *      g(y)=1+\frac{9 \sum_{i=2}^{n}h_g(y_i)}{(n-1)} 
 * \f]
 * with auxiliary functions:
 * \f[
 *     h: \mathbf{R} \rightarrow [0,1], x \rightarrow (1+exp(\frac{-x}{ \sqrt{n} }))^{-1}
 * \f]
 * \f[
 *     h_f: \mathbf{R} \rightarrow \mathbf{R}, x \rightarrow \left\{\begin{array}{ll} x, & \mbox{if } |y_1|\leq y_{max}  \\ 
 *          1+|y_1|, & \mbox{ otherwise} \end{array}\right.
 * \f]
 * \f[
 *     h_g: \mathbf{R} \rightarrow \mathbf{R}_{\leq 0}, x \rightarrow \frac{x^2}{|x|+0.1}
 * \f]
 * where
 * \f$  x\in [0,1]^n \f$ ,
 * y is defined by \f$ y= Ox \f$ 
 * where O is an arbitrary orthogonal rotation matrix and 
 * \f$ y_{max} = 1/max_j (|o_{1j}|)\f$
 * and usually \f$ n=10 \f$
 */
class IHR3 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    IHR3( unsigned d =10);

    //! Destructor
    ~IHR3();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
    
	//! Create a random orthogonal transformation
	void initRandomRotation();

protected:
	//! transform #in, resulting in #out
	void transform(const double* in, std::vector<double>& out) const;
    double h(double x, double n);
	double hf(double x, double y0, double ymax);
	double hg(double x);

	//! transformation matrix
	Matrix m_Transformation;
};



/*!
 * \brief multi-objective problem IHR6.
 *
 *  The rotated variant of the ZDT6 problem:
 *
 * \f[
 *      f_{1}(x)= 1-exp(-4|y_1|)sin^6(6 \pi y_1)
 * \f]
 * \f[
 *      f_{2}(x)=g(y)h_f( 1 - (\frac{y_1}{ g(y) })^2) 
 * \f]
 * \f[
 *      g(y)=1+9 [\frac{\sum_{i=2}^{n}h_g(y_i)}{(n-1)}]^{0.25} 
 * \f]
 * with auxiliary functions:
 * \f[
 *     h_f: \mathbf{R} \rightarrow \mathbf{R}, x \rightarrow \left\{\begin{array}{ll} x, & \mbox{if } |y_1|\leq y_{max}  \\ 
 *          1+|y_1|, & \mbox{ otherwise} \end{array}\right.
 * \f]
 * \f[
 *     h_g: \mathbf{R} \rightarrow \mathbf{R}_{\leq 0}, x \rightarrow \frac{x^2}{|x|+0.1}
 * \f]
 * where
 * \f$  x\in [0,1]^n \f$ ,
 * y is defined by \f$ y= Ox \f$ 
 * where O is an arbitrary orthogonal rotation matrix and 
 * \f$ y_{max} = 1/max_j (|o_{1j}|)\f$
 * and usually \f$ n=10 \f$
 */
class IHR6 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    IHR6( unsigned d =10);

    //! Destructor
    ~IHR6();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
    
	//! Create a random orthogonal transformation
	void initRandomRotation();

protected:
	//! transform #in, resulting in #out
	void transform(const double* in, std::vector<double>& out) const;
    double h(double x, double n);
	double hf(double x, double y0, double ymax);
	double hg(double x);

	//! transformation matrix
	Matrix m_Transformation;
};


//  Problems with prescirbed Pareto Set (PS) :

//  Problems with linear Ps shapes


/*!
 * \brief  multi-objective problem ZZJ07_F1 
 *  
 * Multi objective problem with linear Pareto Set shape.
 * The ZZJ07_F1 problem is given by:
 *
 * \f[
 *      f_{1}(x)=x_1
 * \f]
 * \f[
 *      f_{2}(x)=g(x)( 1 - \sqrt{ \frac{x_1}{g(x)} } )
 * \f]
 * where
 * \f[
 *      g(x)=1+\frac{9}{n-1} \sum_{i=2}^{n} (x_i - x_1)^2
 * \f]
 * \f[
 *      0 \leq  x_i  \leq 1
 * \f]
 * and usually \f$ n=30 \f$
 */
class ZZJ07_F1 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZZJ07_F1(unsigned d = 30);

    //! Destructor
    ~ZZJ07_F1();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};

/*!
 * \brief multi-objective problem ZZJ07_F2  
 *
 * Multi objective problem with linear Pareto Set shape.
 * The ZZJ07_F2 problem is given by:
 *
 * \f[
 *      f_{1}(x)=x_1
 * \f]
 * \f[
 *      f_{2}(x)=g(x)( 1 - ( \frac{x_1}{g(x)} )^2 )
 * \f]
 * where
 * \f[
 *      g(x)=1+\frac{9}{n-1} \sum_{i=2}^{n} (x_i - x_1)^2
 * \f]
 * \f[
 *      0 \leq  x_i  \leq 1
 * \f]
 * and usually \f$ n=30 \f$
 */
class ZZJ07_F2 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZZJ07_F2( unsigned d = 30);

    //! Destructor
    ~ZZJ07_F2();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem ZZJ07_F3
 *
 * Multi objective problem with linear Pareto Set shape.
 * The ZZJ07_F3 problem is given by:
 *
 * \f[
 *      f_{1}(x)=1-\exp (-4 x_1) \sin^6 (6 \pi x_1)
 * \f]
 * \f[
 *      f_{2}(x)=g(x)( 1 - ( \frac{f_1(x)}{g(x)} )^2 )
 * \f]
 * where
 * \f[
 *      g(x)=1+9 ( \sum{i=2}^{n} (x_i - x_1)^2 / 9)^{0.25}
 * \f]
 * \f[
 *      0 \leq  x_i  \leq 1
 * \f]
 * and usually \f$ n=30 \f$
 */
class ZZJ07_F3 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZZJ07_F3(unsigned d = 30);

    //! Destructor
    ~ZZJ07_F3();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem ZZJ07_F4
 * 
 * Multi objective problem with linear Pareto Set shape.
 * The ZZJ07_F4 problem is given by:
 *
 * \f[
 *      f_{1}(x)=\cos (\frac{\pi}{2}*x_1)  \cos (\frac{\pi}{2}*x_2)(1+ g(x))
 * \f]
 * \f[
 *      f_{2}(x)=\cos (\frac{\pi}{2}*x_1)  \sin (\frac{\pi}{2}*x_2)(1+ g(x))
 * \f]
 * \f[
 *      f_{3}(x)= \sin (\frac{\pi}{2}*x_1)(1+ g(x))
 * \f]
 * where
 * \f[
 *      g(x)= \sum{i=3}^{n} (x_i - x_1)^2 
 * \f]
 * \f[
 *      0 \leq  x_i  \leq 1
 * \f]
 * and usually \f$ n=30 \f$
 */
class ZZJ07_F4 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZZJ07_F4(unsigned d = 30);

    //! Destructor
    ~ZZJ07_F4();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};

//  Problems with quadratic PS shapes

/*!
 * \brief multi-objective problem ZZJ07_F5 
 *  
 * Multi objective problem with quadratic Pareto Set shape.
 * The ZZJ07_F5 problem is given by:
 *
 * \f[
 *      f_{1}(x)=x_1
 * \f]
 * \f[ 
 *      f_{2}(x)=g(x)( 1 - \sqrt{ \frac{x_1}{g(x)} } )
 * \f]
 * where
 * \f[
 *      g(x)=1+\frac{9}{n-1} \sum_{i=2}^{n} (x_i^2 - x_1)^2
 * \f]
 * \f[
 *      0 \leq  x_i  \leq 1
 * \f]
 * and usually \f$ n=30 \f$
 */
class ZZJ07_F5 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZZJ07_F5(unsigned d = 30);

    //! Destructor
    ~ZZJ07_F5();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem ZZJ07_F6 
 *  
 * Multi objective problem with quadratic Pareto Set shape.
 * The ZZJ07_F6 problem is given by:
 *
 *
 * \f[
 *      f_{1}(x)= \sqrt{x_1}
 * \f]
 * \f[
 *      f_{2}(x)=g(x)( 1 - ( \frac{f_1(x)}{g(x)} )^2 )
 * \f]
 * where
 * \f[
 *      g(x)=1+\frac{9}{n-1} \sum_{i=2}^{n} (x_i^2-x_1)^2
 * \f]
 * \f[
 *      0 \leq  x_i  \leq 1
 * \f]
 * and usually \f$ n=30 \f$ 
 */
class ZZJ07_F6 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZZJ07_F6(unsigned d = 30);

    //! Destructor
    ~ZZJ07_F6();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem ZZJ07_F7
 *  
 * Multi objective problem with quadratic Pareto Set shape.
 * The ZZJ07_F7 problem is given by:
 *
 *
 * \f[
 *      f_{1}(x)=1-\exp (-4 x_1) \sin^6 (6 \pi x_1)
 * \f]
 * \f[
 *      f_{2}(x)=g(x)( 1 - ( \frac{f_1(x)}{g(x)} )^2 )
 * \f]
 * where
 * \f[
 *     g(x) = 1+9(\sum_{i=2}^{n}(x_{i}^2-x_{1})^{2}/9)^{0.25}
 * \f]
 * \f[
 *      0 \leq  x_i  \leq 1 
 * \f]
 * and usually  \f$ n=30 \f$
 */
class ZZJ07_F7 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZZJ07_F7(unsigned d = 30);

    //! Destructor
    ~ZZJ07_F7();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem ZZJ07_F8
 *  
 * Multi objective problem with quadratic Pareto Set shape.
 * The ZZJ07_F8 problem is given by:
 *
 *
 * \f[
 *      f_{1}(x)=\cos (\frac{\pi}{2}*x_1)  \cos (\frac{\pi}{2}*x_2)(1+ g(x))
 * \f]
 * \f[
 *      f_{2}(x)=\cos (\frac{\pi}{2}*x_1)  \sin (\frac{\pi}{2}*x_2)(1+ g(x))
 * \f]
 * \f[
 *      f_{3}(x)= \sin (\frac{\pi}{2}*x_1)(1+ g(x))
 * \f]
 * where
 * \f[
 *      g(x)= \sum_{i=3}^{n} (x_i^2 - x_1)^2
 * \f]
 * \f[
 *      0 \leq  x_i  \leq 1
 * \f]
 * and usually \f$ n=30 \f$
 */
class ZZJ07_F8 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZZJ07_F8(unsigned d = 30);

    //! Destructor
    ~ZZJ07_F8();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem ZZJ07_F9
 *    
 * Multi objective problem with quadratic Pareto Set shape.
 * The ZZJ07_F9 problem is given by:
 *
 * \f[
 *      f_{1}(x)=x_1
 * \f]
 * \f[
 *      f_{2}(x)=g(x)(1- \sqrt{\frac{x_1}{g(x)}})
 * \f]
 * where
 * \f[
 *      g(x) = \frac{1}{4000}\sum_{i=2}^{n}(x_{i}^{2}-x_{1})^{2}-\prod_{i=2}^{n}cos(\frac{x_{i}^{2}-x_{1}}{\sqrt{i-1}}) +2
 * \f]
 * \f[ 
 *      x\in [0,1]\times [0,10]^{n-1}
 * \f]
 * and usually \f$ n=10 \f$
 */
class ZZJ07_F9 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZZJ07_F9(unsigned d = 10);

    //! Destructor
    ~ZZJ07_F9();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem ZZJ07_F10
 *  
 * Multi objective problem with quadratic Pareto Set shape.
 * The ZZJ07_F10 problem is given by:
 *
 *
 * \f[
 *      f_1(x) = x_1
 * \f]
 * \f[
 *      f_{2}(x)=g(x)(1- \sqrt{\frac{x_1}{g(x)}})
 * \f]
 * where
 * \f[
 *      g(x)=1+10(n-1) +  \sum_{i=2}^{n} ((x_i^2 - x_1)^2- 10\cos(2\pi (x_i^2 - x_1))) 
 * \f]
 * \f[
 *       x\in [0,1]\times [0,10]^{n-1}  
 * \f]
 * and usually \f$ n=10 \f$
 */
class ZZJ07_F10 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
    ZZJ07_F10(unsigned d = 10);

    //! Destructor
    ~ZZJ07_F10();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem LZ06_F1
 *  
 * Multi objective problem with complicated Pareto Set shape.
 * The LZ06_F1 problem is given by:
 *
 * \f[
 *      f_1(x) = x_1
 * \f]
 * \f[
 *      f_2(x) = 1 - \sqrt{ \frac{ x_{1} }{ g(x) } }
 * \f]
 * where
 * \f[
 *      g(x) = 1+\frac{1}{n-1}\sum_{i=2}^{n}|x_{1}-sin(0.5x_{i}\pi)| 
 * \f]
 * \f[
 *      x\in [0,1]^{n} 
 * \f]
 * and usually \f$n = 30\f$
 */
class LZ06_F1 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	LZ06_F1( unsigned d = 30 );

    //! Destructor
    ~LZ06_F1();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem LZ06_F2 
 *  
 *  Multi objective problem with complicated Pareto Set shape.
 *  The LZ06_F2 problem is given by:
 *
 * \f[
 *      f_1(x) = x_1
 * \f]
 * \f[
 *      f_{2}(x)=1-(\frac{x_{1}}{g(x)})^2
 * \f]
 * where
 * \f[
 *      g(x) = 1+\frac{1}{n-1}\sum_{i=2}^{n}|x_{1}-sin(0.5x_{i}\pi)| 
 * \f]
 * \f[
 *      x\in [0,1]^{n} 
 * \f]
 * and usually \f$n = 30\f$
 */
class LZ06_F2 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	LZ06_F2( unsigned d = 30 );

    //! Destructor
    ~LZ06_F2();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};



/*!
 * \brief  multi-objective problem LZ07_F1 
 *  
 * Multi objective problem with complicated Pareto Set shape.
 * The LZ07_F1 problem is given by:
 * \f[
 *       f_1(x) = x_1 + \frac{2}{|J_1|} \sum_{j \in J_1} (x_j - x_1^{0.5(1.0 + \frac{3(j-2)}{n-2})})^2 \\      
 * \f]
 * \f[
 *       f_2(x) = 1 - \sqrt{x_1} + \frac{2}{|J_2|} \sum_{j \in J_2} (x_j - x_1^{0.5(1.0 + \frac{3(j-2)}{n-2})})^2
 * \f]
 * where
 * \f[
 *      x = (x_1, \ldots,x_n) \in [0,1]^n
 * \f]
 * \f[
 *      J_1=\{j|j \mbox{ is odd and } 2 \le j \le n\}
 * \f]
 * and 
 * \f[
 *      J_2=\{j|j \mbox{ is even and } 2 \le j \le n\}
 * \f]
 * and usually \f$n = 30\f$
 */
class LZ07_F1 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	LZ07_F1( unsigned d = 30 );

    //! Destructor
    ~LZ07_F1();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem LZ07_F2
 *  
 * Multi objective problem with complicated Pareto Set shape.
 * The LZ07_F2 problem is given by:
 * \f[
 *      f_1(x) = x_1 + \frac{2}{|J_1|} \sum_{j \in J_1} (x_j - \sin(6 \pi x_1 + \frac{j \pi}{n}))^2
 * \f]
 * \f[
 *      f_2(x) = 1 - \sqrt{x_1} + \frac{2}{|J_2|} \sum_{j \in J_2} (x_j - \sin(6 \pi x_1 + \frac{j \pi}{n}))^2
 * \f]
 * where
 * \f[
 *      x= (x_1, \ldots,x_n) \in [0,1]\times[-1,1]^{n-1}
 * \f]
 * \f[
 *      J_1=\{j|j \mbox{ is odd and } 2 \le j \le n\}
 * \f]
 * and 
 * \f[
 *      J_2=\{j|j \mbox{ is even and } 2 \le j \le n\}
 * \f]
 * and usually \f$n = 30\f$
 */
class LZ07_F2 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	LZ07_F2( unsigned d = 30 );

    //! Destructor
    ~LZ07_F2();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem LZ07_F3
 *  
 * Multi objective problem with complicated Pareto Set shape.
 * The LZ07_F3 problem is given by:
 *
 * \f[
 *      f_1(x) = x_1 + \frac{2}{|J_1|} \sum_{j \in J_1} (x_j - 0.8 x_1 \cos(6 \pi x_1 + \frac{j \pi}{n}))^2
 * \f]
 * \f[
 *      f_2(x) = 1 - \sqrt{x_1} + \frac{2}{|J_2|} \sum_{j \in J_2} (x_j - 0.8 x_1 \sin(6 \pi x_1 + \frac{j \pi}{n}))^2
 * \f]
 * where
 * \f[
 *      x= (x_1, \ldots,x_n) \in [0,1]\times[-1,1]^{n-1}
 * \f]
 * \f[
 *      J_1=\{j|j \mbox{ is odd and } 2 \le j \le n\}
 * \f]
 * and 
 * \f[
 *      J_2=\{j|j \mbox{ is even and } 2 \le j \le n\}
 * \f]
 * and usually \f$n = 30\f$
 */
class LZ07_F3 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	LZ07_F3( unsigned d = 30 );

    //! Destructor
    ~LZ07_F3();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem LZ07_F4
 *  
 * Multi objective problem with complicated Pareto Set shape.
 * The LZ07_F4 problem is given by:
 * \f[
 *      f_1(x) = x_1 + \frac{2}{|J_1|} \sum_{j \in J_1} (x_j - 0.8 x_1 \cos(\frac{6 \pi x_1 + \frac{j \pi}{n}}{3}))^2
 * \f]
 * \f[
 *      f_2(x) = 1 - \sqrt{x_1} + \frac{2}{|J_2|} \sum_{j \in J_2} (x_j - 0.8 x_1 \sin(6 \pi x_1 + \frac{j \pi}{n}))^2
 * \f]
 * where
 * \f[
 *      x= (x_1, \ldots,x_n) \in [0,1]\times[-1,1]^{n-1}
 * \f]
 * \f[
 *      J_1=\{j|j \mbox{ is odd and } 2 \le j \le n\}
 * \f]
 * and 
 * \f[
 *      J_2=\{j|j \mbox{ is even and } 2 \le j \le n\}
 * \f]
 * and usually \f$n = 30\f$
 */
class LZ07_F4 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	LZ07_F4( unsigned d = 30 );

    //! Destructor
    ~LZ07_F4();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem LZ07_F5 
 *  
 * Multi objective problem with complicated Pareto Set shape.
 * The LZ07_F5 problem is given by:
 * \f[
 *      f_1(x) = x_1 + \frac{2}{|J_1|} \sum_{j \in J_1} (x_j - 0.3 x_1 ( x_1 \cos(4(6 \pi x_1 + \frac{j \pi}{n}))+2) \sin(6 \pi x_1 + \frac{j \pi}{n}))^2
 * \f]
 * \f[
 *    	f_2(x) = 1 - \sqrt{x_1} + \frac{2}{|J_2|} \sum_{j \in J_2} (x_j - 0.3 x_1 (x_1 \cos(4(6 \pi x_1 
 *      + \frac{j \pi}{n}))+2) \cos(6 \pi x_1 + \frac{j \pi}{n}))^2
 * \f]
 * where
 * \f[
 *      x= (x_1, \ldots,x_n) \in [0,1]\times[-1,1]^{n-1}
 * \f]
 * \f[
 *      J_1=\{j|j \mbox{ is odd and } 2 \le j \le n\}
 * \f]
 * and 
 * \f[
 *      J_2=\{j|j \mbox {is even and } 2 \le j \le n\}
 * \f]
 * and usually \f$n = 30\f$
 */
class LZ07_F5 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	LZ07_F5( unsigned d = 30 );

    //! Destructor
    ~LZ07_F5();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem LZ07_F6 
 *  
 * Multi objective problem with complicated Pareto Set shape.
 * The LZ07_F6 problem is given by:
 *
 * \f[
 *      f_1(x) = \cos(0.5 x_1 \pi)\cos(0.5 x_2 \pi) + \frac{2}{|J_1|} \sum_{j \in J_1}(x_j - 2 x_2 \sin(2 \pi x_1 + \frac{j \pi}{n}))^2
 * \f]
 * \f[
 *      f_2(x)= \cos(0.5 x_1 \pi)\sin(0.5 x_2 \pi) + \frac{2}{|J_2|} \sum_{j \in J_2}(x_j - 2 x_2 \sin(2 \pi x_1 + \frac{j \pi}{n}))^2
 * \f]
 * \f[
 *      f_3(x) = \sin(0.5 x_1 \pi) + \frac{2}{|J_3|} \sum_{j \in J_3}(x_j - 2 x_2 \sin(2 \pi x_1 + \frac{j \pi}{n}))^2
 * \f]
 * where
 * \f[
 *      x = (x_1, \ldots,x_n)\in [0,1]^2\times[-2,2]^{n-2}
 * \f]
 * \f[
 *      J_1=\{j |3 \leq j \leq n \mbox{ and } j-1 \mbox{ is a multiplication of } 3 \}
 * \f]
 * \f[
 *      J_2=\{j |3 \leq j \leq n \mbox{ and } j-2 \mbox{ is a multiplication of } 3 \}
 * \f]
 * and 
 * \f[
 *      J_3=\{j |3 \leq j \leq n \mbox{ and }  j  \mbox{ is a multiplication of } 3\} 
 * \f]
 * and usually \f$n = 10\f$
 */
class LZ07_F6 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	LZ07_F6( unsigned d = 10 );

    //! Destructor
    ~LZ07_F6();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem LZ07_F7
 *  
 * Multi objective problem with complicated Pareto Set shape.
 * The LZ07_F7 problem is given by:
 *
 * \f[
 *       f_1(x) = x_1 + \frac{2}{|J_1|}\sum_{j \in J_1} (4 y_j^2 - \cos(8 y_j \pi) + 1.0)     
 * \f]
 * \f[
 *       f_2(x) = 1 - \sqrt{x_1} + \frac{2}{|J_2|}\sum_{j \in J_2} (4 y_j^2 - \cos(8 y_j \pi) + 1.0)
 * \f]
 * where
 * \f[
 * 	    y_j = x_j - x_1^{0.5(1.0 + \frac{3(j-2)}{n-2})}, j=2,\ldots,n
 * \f]
 * \f[
 *      x = (x_1, \ldots,x_n) \in [0,1]^n
 * \f]
 * \f[
 *      J_1=\{j|j \mbox{ is odd and } 2 \le j \le n\}
 * \f]
 * and 
 * \f[
 *      J_2=\{j|j \mbox{ is even and } 2 \le j \le n\}
 * \f]
 * and usually \f$n = 30\f$
 */
class LZ07_F7 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	LZ07_F7( unsigned d = 30 );

    //! Destructor
    ~LZ07_F7();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 * \brief multi-objective problem LZ07_F8
 *  
 * Multi objective problem with complicated Pareto Set shape.
 * The LZ07_F8 problem is given by:
 *
 * \f[
 *       f_1(x) = x_1 + \frac{4}{|J_1|}(2 \sum_{j \in J_1} y_j^2 - \prod_{j \in J_1} \cos(\frac{20 y_j \pi}{\sqrt{j}}) + 1) 
 * \f]
 * \f[
 *       f_2(x) = 1 - \sqrt{x_1} + \frac{4}{|J_2|} (2 \sum_{j \in J_2} y_j^2 - \prod_{j \in J_2} \cos(\frac{20 y_j \pi}{\sqrt{j}}) + 1)
 * \f]
 * where
 * \f[
 * 	     y_j = x_j - x_1^{0.5(1.0 + \frac{3(j-2)}{n-2})}, j=2,\ldots,n
 * \f]
 * \f[
 *      x = (x_1, \ldots,x_n) \in [0,1]^n
 * \f]
 * \f[
 *      J_1=\{j|j \mbox{ is odd and } 2 \le j \le n\}
 * \f]
 * and 
 * \f[
 *      J_2=\{j|j \mbox{ is even and } 2 \le j \le n\}
 * \f]
 * and usually \f$n = 30\f$
 */
class LZ07_F8 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	LZ07_F8( unsigned d = 30 );

    //! Destructor
    ~LZ07_F8();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};    
 

/*!
 * \brief multi-objective problem LZ07_F9
 *   
 * Multi objective problem with complicated Pareto Set shape.
 * The LZ07_F9 problem is given by:LZ07_F9 problem with concave Pareto Front
 *
 * \f[
 *      f_1(x) = x_1 + \frac{2}{|J_1|} \sum_{j \in J_1}(x_j -\sin(6 \pi x_1 + \frac{j \pi}{n}))^2 
 * \f]
 * \f[
 *      f_2(x)= 1 - x_1^2 + \frac{2}{|J_2|} \sum_{j \in J_2} (x_j - \sin(6 \pi x_1 + \frac{j \pi}{n}))^2 
 * \f]
 * where
 * \f[
 *      x= (x_1, \ldots,x_n) \in [0,1]\times[-1,1]^{n-1}
 * \f]
 * \f[
 *      J_1=\{j|j \mbox{ is odd and } 2 \le j \le n\}
 * \f]
 * and 
 * \f[
 *      J_2=\{j|j \mbox{ is even and } 2 \le j \le n\}
 * \f]
 * and usually \f$n = 30\f$
 */
class LZ07_F9 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	LZ07_F9( unsigned d = 30 );

    //! Destructor
    ~LZ07_F9();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


/*!
 *  \brief multi-objective problem ELLIBase
 * 
 * The unrotated base problem of ELLI1 
 *
 * \f[
 *      f_1(x) = \frac{1}{a^2*n}\sum_{i=1}^{n}(a^{2{}\frac{i-1}{n-1}}*x_i^2)
 * \f]
 * \f[
 *      f_2(x) = \frac{1}{a^2*n}\sum_{i=1}^{n}(a^{2{}\frac{i-1}{n-1}}*(x_i-2)^2)
 * \f]
 * where usually \f$n = 10\f$ and \f$ a=1000 \f$
 */
class ELLIBase : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	ELLIBase( unsigned d = 10, double a =1000);

    //! Destructor
    ~ELLIBase();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;

protected:
	double m_a;
};


/*!
 * \brief multi-objective problem ELLI1
 *
 * \f[
 *      f_1(x) = \frac{1}{a^2*n}\sum_{i=1}^{n}(a^{2{}\frac{i-1}{n-1}}*y_i^2)
 * \f]
 * \f[
 *      f_2(x) = \frac{1}{a^2*n}\sum_{i=1}^{n}(a^{2{}\frac{i-1}{n-1}}*(y_i-2)^2)
 * \f]
 * where y is defined by:
 * \f[
 *      y= Ox
 * \f]
 * where O is an arbitrary orthogonal rotation matrix
 * and usually \f$n = 10\f$ and \f$ a=1000 \f$
 */
class ELLI1 : public TransformedObjectiveFunction {
public:
	//! Constructor
	ELLI1( unsigned d = 10, double a=1000 );

	//! Destructor
	~ELLI1();

	unsigned int objectives() const;
	bool ProposeStartingPoint(double*& point) const;

protected:
	//! non-transformed base model
	ELLIBase base;
};



/*!
 *  \brief multi-objective problem ELLI2
 *
 * \f[
 *      f_1(x) = \frac{1}{a^2*n}\sum_{i=1}^{n}(a^{2{}\frac{i-1}{n-1}}*x_i^2)
 * \f]
 * \f[
 *      f_2(x) = \frac{1}{a^2*n}\sum_{i=1}^{n}(a^{2{}\frac{i-1}{n-1}}*(x_i-2)^2)
 * \f]
 * where y and z are defined by:
 * \f[
 *     y= O_1*x
 * \f]
 * \f[
 *     z= O_2*x
 * \f]
 * where \f$ O_1 \f$ and \f$ O_2\f$ are arbitrary orthogonal rotation matrices
 * and usually \f$ n = 10\f$ and \f$ a=1000 \f$
 */
class ELLI2 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	ELLI2( unsigned d = 10, double a =1000);

    //! Destructor
    ~ELLI2();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;

	//! Create a random orthogonal transformation
	void initRandomRotation();

	protected:
	//! transform #in with m_Transformation_1, resulting in #out
	void transform_1 (const double* in, std::vector<double>& out) const;

	//! transform #in with m_Transformation_2, resulting in #out
	void transform_2 (const double* in, std::vector<double>& out) const;

	double m_a;
    //! transformation matrix for the first objective function  
 	Matrix m_Transformation_1;
    //! transformation matrix for the secound objective function
	Matrix m_Transformation_2;
	
};



/*!
 * \brief multi-objective problem CIGTABBase
 *
 * The unrotated base problem of CIGTAB1
 *
 * \f[
 *      f_1(x) = \frac{1}{a^2*n} [x_1^2 + \sum_{i=2}^{n-1}a*x_i^2 + a^2*x_n^2]
 * \f]
 * \f[
 *      f_2(x) = \frac{1}{a^2*n} [(x_1-2)^2+ \sum_{i=2}^{n-1}a*(x_i-2)^2 + a^2* (x_n-2)^2]
 * \f]
 * where usually \f$n = 10\f$ and \f$ a=1000 \f$
 */
class CIGTABBase : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	CIGTABBase( unsigned d = 10, double a =1000);

    //! Destructor
    ~CIGTABBase();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;

	protected:
	double m_a;
};



/*!
 * \brief multi-objective problem CIGTAB1
 *
 * \f[
 *      f_1(x) = \frac{1}{a^2*n} [y_1^2 + \sum_{i=2}^{n-1}a*y_i^2 + a^2*y_n^2]
 * \f]
 * \f[
 *      f_2(x) = \frac{1}{a^2*n} [(y_1-2)^2+\sum_{i=2}^{n-1}a*(y_i-2)^2 + a^2 (y_n-2)^2]
 * \f]
 * where y is defined by:
 * \f[
 *     y= Ox
 * \f]
 * where O is an arbitrary orthogonal rotation matrix
 * and usually \f$n = 10\f$ and \f$ a=1000 \f$
 */
class CIGTAB1 : public TransformedObjectiveFunction {
public:
	//! Constructor
	CIGTAB1( unsigned d = 10, double a=1000 );

	//! Destructor
	~CIGTAB1();

	unsigned int objectives() const;
	bool ProposeStartingPoint(double*& point) const;

protected:
	//! non-transformed base model
	CIGTABBase base;
};


/*!
 * \brief multi-objective problem CIGTAB2
 *
 * \f[
 *      f_1(x) = \frac{1}{a^2*n} [y_1^2 + \sum_{i=2}^{n-1}a*y_i^2 + a^2*y_n^2]
 * \f]
 * \f[
 *      f_2(x) = \frac{1}{a^2*n} [(z_1-2)^2+ \sum_{i=2}^{n-1}a*(z_i-2)^2 + a^2 *(z_n-2)^2]
 * \f]
  * where y and z are defined by:
 * \f[
 *     y= O_1*x
 * \f]
 * \f[
 *     z= O_2*x
 * \f]
 * where  \f$ O_1 \f$ and \f$ O_2 \f$ are arbitrary orthogonal rotation matrices
 * and usually \f$n = 10\f$ and \f$ a=1000 \f$
 */
class CIGTAB2 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	CIGTAB2( unsigned d = 10, double a =1000);

    //! Destructor
    ~CIGTAB2();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;

	//! Create a random orthogonal transformation
	void initRandomRotation();

	protected:
	//! transform #in with m_Transformation_1, resulting in #out
	void transform_1 (const double* in, std::vector<double>& out) const;

	//! transform #in with m_Transformation_2, resulting in #out
	void transform_2 (const double* in, std::vector<double>& out) const;

	double m_a;
    //! transformation matrix for the first objective function  
 	Matrix m_Transformation_1;
    //! transformation matrix for the secound objective function
	Matrix m_Transformation_2;
	
};



/*!
 * \brief multi-objective problem DTLZ1
 *
 * \f[
 *      f_1(x) = 0.5*x_1*x_2 \ldots x_{m-1}*(1+g(X_m))
 * \f]
 * \f[
 *      f_2(x) = 0.5*x_1*x_2 \ldots * x_{m-2}*(1-x_{m-1})*(1+g(X_m))
 * \f]
 * \f$ \ldots \f$
 * \f[
 *		f_{m-1}(x) = 0.5*x_1*(1-x_2)*(1+g(X_m))
 * \f]
 * \f[
 *   	f_m(x) = 0.5*(1-x_1)*(1+g(X_m))
 * \f]
 * where
 * \f[
 *      g(X_m)= 100 *(|X_m|+\sum_{x_i \in X_m}(x_i-0.5)^2-cos(20 \pi (x_i-0.5)))
 * \f]
 * \f[
 *      x= (x_1, \ldots,x_n) \in [0,1]^{n}
 * \f]
 * \f$ m \leq n \f$ and usually \f$n = 30\f$
 */
class DTLZ1 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	DTLZ1( unsigned d = 30, unsigned m=3);

    //! Destructor
    ~DTLZ1();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;

protected:
	unsigned m_objectives; 
};


/*!
 * \brief multi-objective problem DTLZ2
 *
 * \f[
 *      f_1(x) = (1+g(X_m))*\cos(x_1* \pi / 2) \ldots cos(x_{m-1}*\pi /2)
 * \f]
 * \f[
 *      f_2(x) = (1+g(X_m))*\cos(x_1* \pi / 2) \ldots sin(x_{m-1}*\pi /2))
 * \f]
 * \f$ \ldots \f$
 * \f[
 *   	f_m(x) = (1+g(X_m))*\sin(x_1* \pi / 2) 
 * \f]
 * where
 * \f[
 *      g(X_m)= \sum_{x_i \in X_m}(x_i-0.5)^2
 * \f]
 * \f[
 *      x= (x_1, \ldots,x_n) \in [0,1]^{n}
 * \f]
 * \f$ m \leq n \f$ and usually \f$n = 30\f$
 */
class DTLZ2 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	DTLZ2( unsigned d = 30, unsigned m=3);

    //! Destructor
    ~DTLZ2();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;

protected:
	unsigned m_objectives; 
};


/*!
 *  \brief multi-objective problem DTLZ3
 *
 * \f[
 *      f_1(x) = (1+g(X_m))*\cos(x_1* \pi / 2) \ldots cos(x_{m-1}*\pi /2)
 * \f]
 * \f[
 *      f_2(x) = (1+g(X_m))*\cos(x_1* \pi / 2) \ldots sin(x_{m-1}*\pi /2))
 * \f]
 * \f$ \ldots \f$
 * \f[
 *   	f_m(x) = (1+g(X_m))*\sin(x_1* \pi / 2) 
 * \f]
 * where
 * \f[
 *      g(X_m)= 100 *(|X_m|+\sum_{x_i \in X_m}(x_i-0.5)^2-cos(20 \pi (x_i-0.5)))
 * \f]
 * \f[
 *      x= (x_1, \ldots,x_n) \in [0,1]^{n}
 * \f]
 * \f$ m \leq n \f$ and usually \f$n = 30\f$
 */
class DTLZ3 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	DTLZ3( unsigned d = 30, unsigned m=3);

    //! Destructor
    ~DTLZ3();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;

protected:
	unsigned m_objectives; 
};


/*!
 * \brief multi-objective problem DTLZ4
 *
 * \f[
 *      f_1(x) = (1+g(X_m))*\cos(x_1* \pi / 2) \ldots cos(x_{m-1}*\pi /2)
 * \f]
 * \f[
 *      f_2(x) = (1+g(X_m))*\cos(x_1* \pi / 2) \ldots sin(x_{m-1}*\pi /2))
 * \f]
 * \f$ \ldots \f$
 * \f[
 *   	f_m(x) = (1+g(X_m))*\sin(x_1* \pi / 2) 
 * \f]
 * where
 * \f[
 *      g(X_m)= \sum_{x_i \in X_m}(x_i^{\alpha}-0.5)^2
 * \f]
 * \f$ \alpha = 100 \f$
 * \f[
 *      x= (x_1, \ldots,x_n) \in [0,1]^{n}
 * \f]
 * \f$ m \leq n \f$ and usually \f$n = 30\f$
 */
class DTLZ4 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	DTLZ4( unsigned d = 30, unsigned m=3);

    //! Destructor
    ~DTLZ4();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;

protected:
	unsigned m_objectives; 
};


/*!
 * \brief multi-objective problem DTLZ5
 *
 * \f[
 *      f_1(x) = (1+g(X_m))*\cos(\phi_1* \pi / 2) \ldots cos(\phi_{m-1}*\pi /2)
 * \f]
 * \f[
 *      f_2(x) = (1+g(X_m))*\cos(\phi_1* \pi / 2) \ldots sin(\phi_{m-1}*\pi /2))
 * \f]
 * \f$ \ldots \f$
 * \f[
 *   	f_m(x) = (1+g(X_m))*\sin(\phi_1* \pi / 2) 
 * \f]
 * where
 * \f[
 *      g(X_m)= \sum_{x_i \in X_m}(x_i-0.5)^2
 * \f]
 * \f$ \phi_i = \frac{\pi}{4(1+g(X_m))}*(1+2g(X_m)x_i) \f$
 * \f[
 *      x= (x_1, \ldots,x_n) \in [0,1]^{n}
 * \f]
 * \f$ m \leq n \f$ and usually \f$n = 30\f$
 */
class DTLZ5 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	DTLZ5( unsigned d = 30, unsigned m=3);

    //! Destructor
    ~DTLZ5();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;

protected:
	unsigned m_objectives; 
};

/*!
 * \brief multi-objective problem DTLZ6
 *
 * \f[
 *      f_1(x) = (1+g(X_m))*\cos(\phi_1* \pi / 2) \ldots cos(\phi_{m-1}*\pi /2)
 * \f]
 * \f[
 *      f_2(x) = (1+g(X_m))*\cos(\phi_1* \pi / 2) \ldots sin(\phi_{m-1}*\pi /2))
 * \f]
 * \f$ \ldots \f$
 * \f[
 *   	f_m(x) = (1+g(X_m))*\sin(\phi_1* \pi / 2) 
 * \f]
 * where
 * \f[
 *      g(X_m)= \sum_{x_i \in X_m}x_i^{0,1}
 * \f]
 * \f$ \phi_i = \frac{\pi}{4(1+g(X_m))}*(1+2g(X_m)x_i) \f$
 * \f[
 *      x= (x_1, \ldots,x_n) \in [0,1]^{n}
 * \f]
 * \f$ m \leq n \f$ and usually \f$n = 30\f$
 */
class DTLZ6 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	DTLZ6( unsigned d = 30, unsigned m=3);

    //! Destructor
    ~DTLZ6();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;

	protected:
	unsigned m_objectives; 
};


/*!
 * \brief multi-objective problem DTLZ7
 *
 * \f[
 *      f_1(x) = x_1
 * \f]
 * \f[
 *      f_2(x) = x_2
 * \f]
 * \f$ \ldots \f$
 * \f[
 *      f_{m-1}(x) = x_{m-1}
 * \f]
 * \f[
 *   	f_m(x) = (1+g(X_m))*h(f_1,f_2, \ldots ,f_{m-1},g)
 * \f]
 * where
 * \f[
 *      g(X_m)= 1 + \frac{9}{|X_m|}\sum_{x_i \in X_m}x_i
 * \f]
 * \f$  h= m -\sum_{i=1}^{m-1}\frac{f_i}{1+g(X_m)(1+sin(3 \pi f_i))}\f$
 * \f[
 *      x= (x_1, \ldots,x_n) \in [0,1]^{n}
 * \f]
 * \f$ m \leq n \f$ and usually \f$n = 30\f$
 */
class DTLZ7 : public ObjectiveFunctionVS<double> {
public:
    //! Constructor
	DTLZ7( unsigned d = 30, unsigned m=3);

    //! Destructor
    ~DTLZ7();

    unsigned int objectives() const;
    void result(double* const& point, std::vector<double>& value);
    bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;

protected:
	unsigned m_objectives; 
};


//!
//! \brief multi-objective problem Superspheres
//!
//! \f[ f_1(x) = (1+r)\cdot \cos(x_1) \f]
//! \f[ f_2(x) = (1+r)\cdot \sin(x_1) \f]
//! with \f$ d=\frac{1}{n-1}\sum_{i=2}^n x_i \f$
//! and \f$ r=\sin^2(\pi\cdot d) \f$.
//!
class Superspheres : public ObjectiveFunctionVS<double>
{
public:
	//! Constructor
	Superspheres(unsigned int dim);

	//! Destructor
	~Superspheres();


	unsigned int objectives() const;
	void result(double* const& point, std::vector<double>& value);
	bool ProposeStartingPoint(double*& point) const;
	bool utopianFitness(std::vector<double>& value) const;
	bool nadirFitness(std::vector<double>& value) const;
};


#endif
