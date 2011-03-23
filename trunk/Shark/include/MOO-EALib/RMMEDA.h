/*! \file	GenModRM.cpp
	
 * \brief	Evolutionary Aglorithm Generator with Local PCA (RM-MEDA)
 * \brief  "RM-MEDA: A Regularity Model-Based Multiobjective Estimation of Distribution Algorithm". IEEE Transaction on Evolutionary Computation
	
 * \author Aimin ZHOU
 * \author Department of Computer Science,
 * \author University of Essex, 
 * \author Colchester, CO4 3SQ, U.K
 * \author azhou@essex.ac.uk
 *
 * Copyright (c) 2005, 2006, 2007, Aimin ZHOU
 *
 * \date	Nov.29 2005 make great changes: noise, border checking
 * \date	Apr.10 2006 redesign
 * \date	Jul.18 2006 add quadratic models
 * \date	Nov.12 2006 modify to uniform version
 * \date	Jun.26 2006 rename and change Generate()
 * \date	Sep.03 2007 modify the boundary checking procedure, it plays an important role in the algorithm
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
*/


#ifndef AZ_GENERATOR_MODEL_H
#define AZ_GENERATOR_MODEL_H

#include <vector>
//#include "PCA.h"
//#include "Matrix.h"
#include <Array/Array.h>
#include <MOO-EALib/PopulationMOO.h>

//!\brief	az namespace, the top namespace
namespace az
{

//!\brief	mea namespace, the multiobjective evolutionary algorithm namespace
namespace mea
{

//!\brief	gen namespace, offspring generate strategies
namespace gen
{

//!\brief model based generator
namespace mod
{
//!\brief Local PCA based EDA generator
class RM
{
protected:
	double			**pData,	//!< pointer to data
	  mExtension;	//!< Principal Curve(Surface) extension ratio
	unsigned int	mDataSize,	//!< data number
	  mDataDim,	//!< data dimension
	  mTrainSteps,//!< inner train steps
	  mLatentDim,	//!< latent dimension
	  mMaxCluster;//!< maximum cluster number
protected:
	//!\brief	clear data pool
	//!\return	void
	void Clear();
public:
	//!\brief	constructor
	//!\return	void
	RM();
	
	//!\brief	destructor
	//!\return	void
	~RM();

	//!\brief	Generator
	//!\param	latent		latent dimension
	//!\param	cluster		maximum cluster number
	//!\param	trainsteps	Local PCA train steps
	//!\param	extension	extension ratio
	//!\param	sizenew		number of new trial solutions
	//!\param	popnew		offspring population
	//!\param	popref		reference population(current population)
	//!\return	void
	void Generate(	unsigned int			latent, 
			unsigned int			cluster, 
			unsigned int			trainsteps, 
			double					extension, 
			Array<double>&	xlow,
			Array<double>&	xupp,
			unsigned int			sizenew, 
			PopulationMOO& popnew, 
			PopulationMOO& popref);
	void Generate(	unsigned int			latent, 
				unsigned int			cluster, 
				unsigned int			trainsteps, 
				double					extension, 
				std::vector<double>&	xlow,
				std::vector<double>&	xupp,
				unsigned int			sizenew, 
				std::vector< std::vector<double> >& popnew, 
				std::vector< std::vector<double> >& popref);
	  
}; //class RM

} //namespace mod

} //namespace gen

} //namespace mea

} //namespace az


 //* =============================================================
 //* NDSelector - Select some points out from a given population
 //*   modified version of Deb's crowded selection
 //* 
 //* Copyright (c) 2008 Aimin Zhou
 //* Dept. of Computer Science
 //* Univ. of Essex
 //* Colchester, CO4 0DY, U.K
 //* azhou@essex.ac.uk
 //* =============================================================

#include <vector>
#include <MOO-EALib/PopulationMOO.h>
//!\brief	az namespace, the top namespace
namespace az
{

//!\brief	mea namespace, the multiobjective evolutionary algorithm namespace
namespace mea
{

//!\brief	gen namespace, offspring generate strategies
namespace sel
{

//!\brief MIDEA
class NDS
{
protected:
	int 	XDim,	//dimension of X
			FDim,	//dimension of F
			NData,	//number of data
			SData;	//number of selected data
	double 	**pX,	//pointer to X
			**pF;	//pointer to F
	std::vector<int> rankV;	//rank vector
	std::vector<int> id;	//id vector	
protected:
	int		Dominate(int iA, int iB);
	void	SortRank();
	double	FDen(double dis);
	void	SortDensity(int iS, int iE);
public:
	//!\brief	ND-selector
	//!\param	size	number of offspring
	//!\param	of		offspring objective vectors
	//!\param	ox		offspring decision vectors
	//!\param	pf		parent objective vectors
	//!\param	px		parent decision vectors
	void Select(unsigned int size, std::vector< std::vector<double> >& of, std::vector< std::vector<double> >& ox, std::vector< std::vector<double> >& pf, std::vector< std::vector<double> >& px);
	
	//!\brief	ND-selector
	//!\param	size	number of offspring
	//!\param	ids		selected id in original order
	//!\param	pf		parent objective vectors
	//!\param	px		parent decision vectors
	void Select(unsigned int size, std::vector< unsigned int >& ids, std::vector< std::vector<double> >& pf, std::vector< std::vector<double> >& px);
	void Select(unsigned int size, unsigned int numberOfObjectives, PopulationMOO &offspring,PopulationMOO &total);

};//class NDS

} //namespace sel
} //namespace mea
} //namespace az
/*! \file	Matrix.h
	
	\brief	Matrix: contains 2-D data
	
	\author Aimin ZHOU
	\author Department of Computer Science,
	\author University of Essex, 
	\author Colchester, CO4 3SQ, U.K
	\author azhou@essex.ac.uk
	
	\date	Dec.30 2004 create
	\date	Sep.25 2005 rewrite & reorganize structure
	\date	Oct.20 2005 remove bugs in Sub,Det,Inv
	\date	Oct.21 2005 add LU decomposition
	\date	Oct.23 2005 add linear algebra functions(Cholesky,CholeskySolve,SVD,pinv)
*/

#include <list>
#include <iostream>
#include <string>
#include <iomanip>
#include <vector>

//!\brief	az namespace, the top namespace
namespace az
{

//!\brief	alg namespace, contains algorithms
namespace alg
{
	//!\brief error process for DATA namespace
	class error : public std::exception 
	{
	public:
		//!\brief	constructor
		//!\param	msg error message
		//!\return	void
		error(std::string const& msg) throw()
			: msg_(std::string("DATA error: ") + msg)
		{}
		
		//!\brief	destructor
		virtual ~error() throw() {}

		//!\brief	look up the error reason
		//!\return	error message
		virtual const char* what() const throw() { return msg_.c_str(); }
	protected:	
		std::string msg_;	//!< error message
	};

	//! check to ensure the expression is right
	#ifdef _DEBUG
		#define CHECK(cond, str)	if(!(cond))	{ throw error(str); }
	#else
		#define CHECK(cond, str)	{}	//if(!(cond))	{ throw error(str); }
	#endif

	//!\brief real vector
	typedef std::vector<double>	FVECTOR;

	//! index structure based on list
	typedef std::list< unsigned int >	LINDEX;

	//! index structure based on vector
	typedef std::vector< unsigned int > VINDEX;

	//!\brief Matrix class
	class Matrix
	{
	protected:
		unsigned int mRow,	//!< row size 
					 mCol;	//!< column size
		double*		 pData;	//!< pointer to the data		
	public:
		//!\brief	constructor
		//!\param	row row size
		//!\param	col column size
		//!\return	void
		Matrix(unsigned int row = 0, unsigned int col = 0);

		//!\brief	constructor
		//!\brief	mat reference matrix
		//!\return	void
		Matrix(const Matrix& mat);

		//!\brief	destructor
		//!\return	void
		~Matrix();

		//!\brief	reset the matrix size
		//!\param	row row size
		//!\param	col column size
		//!\return	reference of the matrix
		Matrix& Resize(unsigned int row, unsigned int col);

		//!\brief	create an identity matrix
		//!\param	size row and column size
		//!\return	reference of the matrix
		Matrix& Identity(unsigned int size);

		//!\brief	get the row size
		//!\return	row size
		inline unsigned int RowSize() {return mRow;}

		//!\brief	get the column size
		//!\return	column size
		inline unsigned int ColSize() {return mCol;}

		//!\brief	get the pointer to the data
		//!\return	the pointer to the data
		inline double* operator()() {return pData;}
		
		//!\brief	get an element
		//!\param	row row number
		//!\param	col column number
		//!\return	reference to the element
		double& operator()(unsigned int row, unsigned int col);

		//!\brief	reset to another matrix
		//!\param	mat matrix reference
		//!\return	reference of the matrix
		Matrix& operator= (const Matrix& mat);
		
		//!\brief	get a row
		//!\param	row row number
		//!\param	value a vector to store the row
		//!\return	reference to value
		FVECTOR& Row(unsigned int row, FVECTOR& value);

		//!\brief	get a column
		//!\param	col column number
		//!\param	value a vector to store the column
		//!\return	reference to value
		FVECTOR& Column(unsigned int col, FVECTOR& value);

		//!\brief	get a sub-matrix except a row and a column
		//!\param	row row number
		//!\param	col column number
		//!\param	mat a matrix to store the sub-matrix
		//!\return	sub-matrix
		Matrix& Sub(unsigned int row, unsigned int col, Matrix& mat);

		//!\brief	calculate the determinant of a square matrix
		//!\return	the deterministic
		double Det();

		//!\brief	translate the matrix
		//!\return	reference to the matrix
		Matrix& Trans();

		//!\brief	inverse the matrix
		//!\return	reference to the matrix
		Matrix& Inv();
		
		//!\brief	calculate the eigenvalue and egienvectors
		//!\param	eigvalue eigenvalue
		//!\param	eigvector eigenvector
		//!\return	void
		void Eig(FVECTOR& eigvalue, Matrix& eigvector);

		//!\brief	multiply a matrix
		//!\param	mat another matrix
		//!\param	result reuslt matrix
		//!\return	reuslt matrix
		Matrix& Multiply(Matrix& mat, Matrix& result);

		//!\brief	left multiply a vector
		//!\param	vec vector
		//!\param	result reuslt vector
		//!\return	reuslt vector
		FVECTOR& LeftMultiply(FVECTOR& vec, FVECTOR& result);

		//!\brief	right multiply a vector
		//!\param	vec vector
		//!\param	result reuslt vector
		//!\return	reuslt vector
		FVECTOR& RightMultiply(FVECTOR& vec, FVECTOR& result);

		//!\brief	divide a scalar
		//!\param	sca scalar
		//!\return	reference to the matrix
		Matrix& Divide(double sca);

		//!\brief	get the mean of all columns
		//!\param	mean the mean vector
		//!\return	mean vector
		FVECTOR& ColMean(FVECTOR& mean);

		//!\brief	get the mean of all rows
		//!\param	mean the mean vector
		//!\return	mean vector
		FVECTOR& RowMean(FVECTOR& mean);

		//!\brief	get standard variation of all columns
		//!\param	std the std vector
		//!\return	std vector
		FVECTOR& ColStd(FVECTOR& std);

		//!\brief	get standard variation of all rows
		//!\param	std the std vector
		//!\return	std vector
		FVECTOR& RowStd(FVECTOR& std);

		//!\brief	subtract a row vector
		//!\param	value row vector
		//!\return	reference to the matrix
		Matrix& RowSub(FVECTOR& value);

		//!\brief	subtract a column vector
		//!\param	value row vector
		//!\return	reference to the matrix
		Matrix& ColSub(FVECTOR& value);

		//!\brief	read a matrix
		//!\param	is input stream
		//!\param	mat matrix
		//!\return	reference to input stream
		friend std::istream& operator>> (std::istream& is, Matrix& mat);

		//!\brief	write a matrix
		//!\param	os output stream
		//!\param	mat matrix
		//!\return	reference to output stream
		friend std::ostream& operator<< (std::ostream& os, Matrix& mat);

		//!\brief	solve A X = b(Numerical Recipes in C++ pp.50-51)
		//!\param	mat LU docomposition of A
		//!\param	indx input vector that records the row permutation by LUdcmp
		//!\param	b right hand of equation
		//!\return	void
		friend void LUbksb(Matrix& mat, std::vector<unsigned int>& indx, std::vector<double>& b);

		//!\brief	LU decompostion of a rowwise permutation(Numerical Recipes in C++ pp.49-50)
		//!\param	mat input and output matrix
		//!\param	indx output vector that records the row permutation
		//!\param	d +-1 depending on whether the number of row interchanges was even or odd
		//!\return	void
		friend void LUdcmp(Matrix& mat, std::vector<unsigned int>& indx, double& d);
	protected:
		//!\brief Householder reduction of Matrix a to tridiagonal form.
		//!
		//! Algorithm: Martin et al., Num. Math. 11, 181-195, 1968.
		//! Ref: Smith et al., Matrix Eigensystem Routines -- EISPACK Guide
		//! Springer-Verlag, 1976, pp. 489-494.
		//! W H Press et al., Numerical Recipes in C, Cambridge U P,
		//! 1988, pp. 373-374. 
		//!\param	eigenvalue eigenvalue
		//!\param	interm temporal variable
		//!\param	eigenvector eigenvector
		//!\return	void
		void tred2(FVECTOR& eigenvalue, FVECTOR& interm, Matrix& eigenvector);

		//!\brief Tridiagonal QL algorithm -- Implicit 
		//!\param	eigenvalue eigenvalue
		//!\param	interm temporal variable
		//!\param	eigenvector eigenvector
		//!\return	void
		void tqli(FVECTOR& eigenvalue, FVECTOR& interm, Matrix& eigenvector);

		//!\brief	sort the eigenvalue by decreasing order
		//!\param	eigenvalue eigenvalue
		//!\param	eigenvector eigenvector
		//!\return	void
		void Sort(FVECTOR& eigenvalue, Matrix& eigenvector);
	};//class Matrix

	//!\brief linear algebra functions

	//!\brief	cholesky factorization of A: L*L'
	//!\param	L factorization matrix(output)
	//!\param	A a square matrix(input)
	//!\return	success or not
	bool Cholesky(Matrix&L, Matrix&A);

	//!\brief 	Solve a linear system A*X = B, using cholesky factorization of A: L*L'
	//!\param	X a matrix so that L*L'*X = B(output)
	//!\param	A a square matrix(input)
	//!\param	B righthand matrix(input)
	//!\return	success or not
	bool CholeskySolve(Matrix& X, Matrix& A, Matrix& B);

	//!\brief	For an m-by-n matrix A with m >= n, so that A = U*S*V'.
	//!\param	U m-by-n orthogonal matrix(output)
	//!\param	S n-by-n diagonal matrix(output)
	//!\param	V n-by-n orthogonal matrix V(output)
	//!\param	A m-by-n matrix(input)
	//!\param	no
	void SVD(Matrix& U, Matrix&S, Matrix&V, Matrix& A);

	//!\brief	find Pseudo inverse matrix by SVD
	//!\param	inA inverse A(output)
	//!\param	A m-by-n matrix(input)
	//!\param	no
	void pinv(Matrix& inA, Matrix& A);

} //namespace alg

} //namespace az
/*! \file	PCA.h
	
	\brief	Principal Component Analysis(PCA)
	
	\author Aimin ZHOU
	\author Department of Computer Science,
	\author University of Essex, 
	\author Colchester, CO4 3SQ, U.K
	\author azhou@essex.ac.uk
	
	\date	Dec.30 2004  create
	\date	Sep.25 2005 rewrite & reorganize structure
*/

#include <iostream>
#include <string>
//#include "Matrix.h"

//!\brief	az namespace, the top namespace
namespace az
{

//!\brief	alg namespace, contains algorithms
namespace alg
{

	//!\brief PCA class
	//!\warning the data must be stored in a column matrix
	class PCA
	{
	protected:
		alg::Matrix*	pData,		//!< data matrix 
						mCov,			//!< colvariance matrix
						mEigenvector;	//!< eigenvectors
		alg::FVECTOR	mMean,			//!< data mean
						mEigenvalue;	//!< eigenvalue
	public:
		//!\brief	constructor
		//!\return	void
		PCA();

		//!\brief	constructor
		//!\param	data data matrix
		//!\return	void
		PCA(alg::Matrix& data);
		
		//!\brief	destructor
		//!\return	void
		~PCA();

		//!\brief	set data
		//!\param	data data matrix
		//!\return	data matrix reference
		inline alg::Matrix& Data(alg::Matrix& data) {pData=&data;return *pData;}

		//!\brief	get data matrix reference
		//!\return	data matrix reference
		inline alg::Matrix& Data() {return *pData;}

		//!\brief	get data mean vector
		//!\return	mean vector
		inline alg::FVECTOR& Mean() {return mMean;}
		
		//!\brief	get eigenvectors
		//!\return	eigenvector matrix
		inline alg::Matrix& Eigenvector() {return mEigenvector;}

		//!\brief	get eigenvalue vector
		//!\return	eigenvalue vector
		inline alg::FVECTOR& Eigenvalue() {return mEigenvalue;}	

		//!\brief	initialize colvariance matrix to be an identity
		//!\param	dim data dimension
		//!\return	void
		void Initialize(unsigned int dim);

		//!\brief	get the range of the projections in a dimension 
		//!\param	dim data dimension
		//!\param	min minimum projection
		//!\param	max maximum projection
		//!\return	void
		void PrimaryRange(unsigned int dim, double& min, double& max);
		
		//!\brief	calculate the mean, eigenvalue and eigenvectors 
		//!\return	void		
		void Train();

		//!\brief	write results into a stream
		//!\param	os output stream
		//!\param	pca PCA object
		//!\return	output stream
		friend std::ostream& operator<<(std::ostream& os, PCA& pca);

		//!\brief	write results into a stream
		//!\return	void
		void Write( std::ostream& os );

		//!\brief	write results into a file
		//!\param	name filename
		//!\return	void
		void Write( std::string& name );

		//!\brief	write results into a file
		//!\param	name filename
		//!\return	void
		void Write( const char* name );
	};

} //namespace alg

} //namespace az



/*! \file	Model.h
	
	\brief	Estimation Distribution Models
	
	\author Aimin ZHOU
	\author Department of Computer Science,
	\author University of Essex, 
	\author Colchester, CO4 3SQ, U.K
	\author azhou@essex.ac.uk
	
	\date	Nov.18 2006 create
*/#include <fstream>
#include <string>
#include <vector>

//!\brief	az namespace, the top namespace
namespace az
{

//!\brief	alg namespace, contains algorithms
namespace alg
{

//!\brief	calcualte the eigenvalues and eigenvectors of a covariance matrix
//!\param	eva		sorted eigenvalues
//!\param	eve		sorted eigenvectors
//!\param	no		the main no-eigens are stored
//!\param	cov		the covariance matrix
//!\return	void
void Eigen(std::vector<double>& eva, std::vector< std::vector<double> >& eve, unsigned int no, std::vector< std::vector<double> >& cov);

//!\brief Model structure, used to store data
class ModelRM
{
public:
	double			mErrTol;		//!< error tolerance
	unsigned int	mNClu,			//!< number of clusters
					mNX,			//!< number of trainning data
					mDX,			//!< dimension of trainning data
					mDLat,			//!< dimension of latent space
					mMaxIter,		//!< maximal trainning steps
					mIter;			//!< real trainning steps
	std::vector< unsigned int >			mvIndex,						//!< cluster index of each data
										mvNo;							//!< the number of points assigned to each cluster
	std::vector< std::vector<double> >	mvX,							//!< trainning data, each row is a data vector
										mvMean,							//!< center of each cluster
										mvEigenvalue,					//!< eigenvalue of ecah cluster
										mvProMin,						//!< minimal projection in primary dimensions
										mvProMax;						//!< maximal projection in primary dimensions
	std::vector< std::vector< std::vector<double> > >	mvEigenvector;	//!< eigenvector of each cluster

public:
	//!\brief	set parameters
	//!\param	nclu	number of cluster
	//!\param	nx		number of trainning data
	//!\param	dx		dimension of trainning data
	//!\param	dlat	dimension of latent space
	//!\param	maxiter	maximal trainning steps
	//!\param	errtol	error tolerance
	//!\return	void
	void Set(unsigned int nclu, unsigned int nx, unsigned int dx, unsigned int dlat, unsigned int maxiter=100, double errtol=1.0E-5);
	//!\brief	write model into file
	//!\param	file file name
	//!\return	void
	void Write(std::string file);

protected:
	//!\brief	calcualte the mean, eigenvalue and eigenvector of a given set of data
	//!\param	mean	mean value of the data
	//!\param	eva		sorted eigenvalues
	//!\param	eve		sorted eigenvectors
	//!\param	index	index of given set of data
	//!\return	void
	void Eigen(std::vector<double>& mean, std::vector<double>& eva, std::vector< std::vector<double> >& eve, std::vector< unsigned int >& index);
};// class Model

} //namespace alg

} //namespace az

/*! \file	LocalPCA.h
	
	\brief	Local Principal Component Analysis (Local PCA) model
	
	\author Aimin ZHOU
	\author Department of Computer Science,
	\author University of Essex, 
	\author Colchester, CO4 3SQ, U.K
	\author azhou@essex.ac.uk
	
	\date	Jan.05 2005 create
	\date	Mar.30 2005 modify
	\date	Apr.15 2005 add VolumeDim()
	\date	Aug.09 2005 rewrite
	\date	Sep.25 2005 rewrite & reorganize structure
	\date	Nov.12 2006 rewrite
	\date	Nov.18 2006 reorganize
						combine Local PCA and Kmeans together
*/



#include <vector>
//#include "Model.h"

//!\brief	az namespace, the top namespace
namespace az
{

//!\brief	alg namespace, contains algorithms
namespace alg
{
//!\brief Local PCA, partion data into clusters 
class LocalPCA:public ModelRM
{
protected:
	std::vector< std::vector< std::vector<double> > >	mvPI;	//!< matrix PI to each cluster
public:
	//!\brief	train process
	//!\return	void
	void Train();
protected:
	//!\brief	calculate the distance between data m to cluster c
	//!\param	m	datat index
	//!\param	c	cluster index
	//!\return	distance
	double Distance(unsigned int m, unsigned int c);
};//class LocalPCA

} //namespace alg

} //namespace az

#endif //AZ_GENERATOR_MODEL_H


