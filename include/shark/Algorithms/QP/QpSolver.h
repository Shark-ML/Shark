//===========================================================================
/*!
 * 
 *
 * \brief       General and specialized quadratic program classes and a generic solver.
 * 
 * 
 *
 * \author      T. Glasmachers, O.Krause
 * \date        2007-2013
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


#ifndef SHARK_ALGORITHMS_QP_QPSOLVER_H
#define SHARK_ALGORITHMS_QP_QPSOLVER_H

#include <shark/Core/Timer.h>
#include <shark/Algorithms/QP/QuadraticProgram.h>

namespace shark{
	
/// \brief Most gneral problem formualtion, needs to be configured by hand.
template<class MatrixT>
class GeneralQuadraticProblem{
public:
	typedef MatrixT MatrixType;
	typedef typename MatrixType::QpFloatType QpFloatType;

	//Setup only using kernel matrix, labels and regularization parameter
	GeneralQuadraticProblem(MatrixType& quadratic)
	: quadratic(quadratic)
	, linear(quadratic.size(),0)
	, alpha(quadratic.size(),0)
	, diagonal(quadratic.size())
	, permutation(quadratic.size())
	, boxMin(quadratic.size(),0)
	, boxMax(quadratic.size(),0)
	{
		for(std::size_t i = 0; i!= dimensions(); ++i){
			permutation[i] = i;
			diagonal(i) = quadratic.entry(i, i);
		}
	}

	/// \brief constructor which initializes a C-SVM problem with weighted datapoints and different regularizers for every class
	GeneralQuadraticProblem(
		MatrixType& quadratic, 
		Data<unsigned int> const& labels, 
		Data<double> const& weights, 
		RealVector const& regularizers
	): quadratic(quadratic)
	, linear(quadratic.size())
	, alpha(quadratic.size(),0)
	, diagonal(quadratic.size())
	, permutation(quadratic.size())
	, boxMin(quadratic.size())
	, boxMax(quadratic.size())
	{
		SIZE_CHECK(dimensions() == linear.size());
		SIZE_CHECK(dimensions() == quadratic.size());
		SIZE_CHECK(dimensions() == labels.numberOfElements());
		SIZE_CHECK(dimensions() == weights.numberOfElements());
		SIZE_CHECK(regularizers.size() > 0);
		SIZE_CHECK(regularizers.size() <= 2);
		
		double Cn = regularizers[0];
		double Cp = regularizers[0];
		if(regularizers.size() == 2)
			Cp = regularizers[1];

		for(std::size_t i = 0; i!= dimensions(); ++i){
			unsigned int label = labels.element(i);
			double weight = weights.element(i);
			permutation[i] = i;
			diagonal(i) = quadratic.entry(i, i);
			linear(i) = label? 1.0:-1.0;
			boxMin(i) = label? 0.0:-Cn*weight;
			boxMax(i) = label? Cp*weight : 0.0;
		}
	}

	std::size_t dimensions()const{
		return quadratic.size();
	}

	/// exchange two variables via the permutation
	void flipCoordinates(std::size_t i, std::size_t j)
	{
		if (i == j) return;

		// notify the matrix cache
		quadratic.flipColumnsAndRows(i, j);
		std::swap( alpha[i], alpha[j]);
		std::swap( linear[i], linear[j]);
		std::swap( diagonal[i], diagonal[j]);
		std::swap( boxMin[i], boxMin[j]);
		std::swap( boxMax[i], boxMax[j]);
		std::swap( permutation[i], permutation[j]);
	}
	
	/// \brief Scales all box constraints by a constant factor and adapts the solution by scaling it by the same factor.
	void scaleBoxConstraints(double factor){
		alpha *= factor;
		boxMin *=factor;
		boxMax *=factor;
	}


	/// representation of the quadratic part of the objective function
	MatrixType& quadratic;

	///\brief Linear part of the problem
	RealVector linear;

	/// Solution candidate
	RealVector alpha;

	/// diagonal matrix entries
	/// The diagonal array is of fixed size and not subject to shrinking.
	RealVector diagonal;

	/// permutation of the variables alpha, gradient, etc.
	std::vector<std::size_t> permutation;

	RealVector boxMin;

	RealVector boxMax;
};

///\brief Boxed problem for alpha in [lower,upper]^n and equality constraints. 
///
///It is assumed for the initial alpha value that there exists a sum to one constraint and lower <= 1/n <= upper 
template<class MatrixT>
class BoxedSVMProblem{
public:
	typedef MatrixT MatrixType;
	typedef typename MatrixType::QpFloatType QpFloatType;

	//Setup only using kernel matrix, labels and regularization parameter
	BoxedSVMProblem(MatrixType& quadratic, RealVector const& linear, double lower, double upper)
	: quadratic(quadratic)
	, linear(linear)
	, alpha(quadratic.size(),1.0/quadratic.size())
	, diagonal(quadratic.size())
	, permutation(quadratic.size())
	, m_lower(lower)
	, m_upper(upper)
	{
		SIZE_CHECK(dimensions() == linear.size());
		SIZE_CHECK(dimensions() == quadratic.size());
		
		for(std::size_t i = 0; i!= dimensions(); ++i){
			permutation[i] = i;
			diagonal(i) = quadratic.entry(i, i);
		}
	}

	std::size_t dimensions()const{
		return quadratic.size();
	}

	double boxMin(std::size_t i)const{
		return m_lower;
	}
	double boxMax(std::size_t i)const{
		return m_upper;
	}

	/// representation of the quadratic part of the objective function
	MatrixType& quadratic;

	///\brief Linear part of the problem
	RealVector linear;

	/// Solution candidate
	RealVector alpha;

	/// diagonal matrix entries
	/// The diagonal array is of fixed size and not subject to shrinking.
	RealVector diagonal;

	/// exchange two variables via the permutation
	void flipCoordinates(std::size_t i, std::size_t j)
	{
		if (i == j) return;
		
		// notify the matrix cache
		quadratic.flipColumnsAndRows(i, j);
		std::swap( alpha[i], alpha[j]);
		std::swap( linear[i], linear[j]);
		std::swap( diagonal[i], diagonal[j]);
		std::swap( permutation[i], permutation[j]);
	}
	
	/// \brief Scales all box constraints by a constant factor and adapts the solution by scaling it by the same factor.
	void scaleBoxConstraints(double factor){
		m_lower*=factor;
		m_upper*=factor;
		alpha *= factor;
	}

	/// permutation of the variables alpha, gradient, etc.
	std::vector<std::size_t> permutation;
private:
	double m_lower;
	double m_upper;
};


/// \brief Problem formulation for binary C-SVM problems
template<class MatrixT>
class CSVMProblem{
public:
	typedef MatrixT MatrixType;
	typedef typename MatrixType::QpFloatType QpFloatType;

	/// \brief Setup only using kernel matrix, labels and regularization parameter
	CSVMProblem(MatrixType& quadratic, Data<unsigned int> const& labels, double C)
	: quadratic(quadratic)
	, linear(quadratic.size())
	, alpha(quadratic.size(),0)
	, diagonal(quadratic.size())
	, permutation(quadratic.size())
	, positive(quadratic.size())
	, m_Cp(C)
	, m_Cn(C)
	{
		SIZE_CHECK(dimensions() == linear.size());
		SIZE_CHECK(dimensions() == quadratic.size());
		SIZE_CHECK(dimensions() == labels.numberOfElements());

		for(std::size_t i = 0; i!= dimensions(); ++i){
			permutation[i] = i;
			diagonal(i) = quadratic.entry(i, i);
			linear(i) = labels.element(i)? 1.0:-1.0;
			positive[i] = linear(i) > 0;
		}
	}
	///\brief  Setup using kernel matrix, labels and different regularization parameters for positive and negative classes
	CSVMProblem(MatrixType& quadratic, Data<unsigned int> const& labels, RealVector const& regularizers)
	: quadratic(quadratic)
	, linear(quadratic.size())
	, alpha(quadratic.size(),0)
	, diagonal(quadratic.size())
	, permutation(quadratic.size())
	, positive(quadratic.size())
	{
		SIZE_CHECK(dimensions() == linear.size());
		SIZE_CHECK(dimensions() == quadratic.size());
		SIZE_CHECK(dimensions() == labels.numberOfElements());
		
		SIZE_CHECK(regularizers.size() > 0);
		SIZE_CHECK(regularizers.size() <= 2);
		m_Cp = m_Cn = regularizers[0];
		if(regularizers.size() == 2)
			m_Cp = regularizers[1];

			
		for(std::size_t i = 0; i!= dimensions(); ++i){
			permutation[i] = i;
			diagonal(i) = quadratic.entry(i, i);
			linear(i) = labels.element(i)? 1.0:-1.0;
			positive[i] = linear(i) > 0;
		}
	}

	//Setup with changed linear part
	CSVMProblem(MatrixType& quadratic, RealVector linear, Data<unsigned int> const& labels, double C)
	: quadratic(quadratic)
	, linear(linear)
	, alpha(quadratic.size(),0)
	, diagonal(quadratic.size())
	, permutation(quadratic.size())
	, positive(quadratic.size())
	, m_Cp(C)
	, m_Cn(C)
	{
		SIZE_CHECK(dimensions() == quadratic.size());
		SIZE_CHECK(dimensions() == linear.size());
		SIZE_CHECK(dimensions() == labels.numberOfElements());
		
		for(std::size_t i = 0; i!= dimensions(); ++i){
			permutation[i] = i;
			diagonal(i) = quadratic.entry(i, i);
			positive[i] = labels.element(i) ? 1: 0;
		}
	}

	std::size_t dimensions()const{
		return quadratic.size();
	}

	double boxMin(std::size_t i)const{
		return positive[i] ? 0.0 : -m_Cn;
	}
	double boxMax(std::size_t i)const{
		return positive[i] ? m_Cp : 0.0;
	}

	/// representation of the quadratic part of the objective function
	MatrixType& quadratic;

	///\brief Linear part of the problem
	RealVector linear;

	/// Solution candidate
	RealVector alpha;

	/// diagonal matrix entries
	/// The diagonal array is of fixed size and not subject to shrinking.
	RealVector diagonal;

	/// exchange two variables via the permutation
	void flipCoordinates(std::size_t i, std::size_t j)
	{
		if (i == j) return;
		
		// notify the matrix cache
		quadratic.flipColumnsAndRows(i, j);
		std::swap( linear[i], linear[j]);
		std::swap( alpha[i], alpha[j]);
		std::swap( diagonal[i], diagonal[j]);
		std::swap( permutation[i], permutation[j]);
		std::swap( positive[i], positive[j]);
	}
	
	/// \brief Scales all box constraints by a constant factor and adapts the solution by scaling it by the same factor.
	void scaleBoxConstraints(double factor, double variableScalingFactor){
		bool sameFactor = factor == variableScalingFactor;
		double newCp = m_Cp*factor;
		double newCn = m_Cn*factor;
		for(std::size_t i = 0; i != dimensions();  ++i){
			if(sameFactor && alpha(i)== m_Cp)
				alpha(i) = newCp;
			else if(sameFactor && alpha(i) == -m_Cn)
				alpha(i) = -newCn;
			else
			alpha(i) *= variableScalingFactor;
		}
		m_Cp = newCp;
		m_Cn = newCn;
	}

	/// permutation of the variables alpha, gradient, etc.
	std::vector<std::size_t> permutation;
private:
	///\brief whether the label of the point is positive
	std::vector<char> positive;

	///\brief Regularization constant of the positive class
	double m_Cp;
	///\brief Regularization constant of the negative class
	double m_Cn;
};

enum AlphaStatus{
	AlphaFree = 0,
	AlphaLowerBound = 1,
	AlphaUpperBound = 2,
	AlphaDeactivated = 3//also:  AlphaUpperBound and AlphaLowerBound
};

///
/// \brief Quadratic program solver
///
/// todo: new documentation
template<class Problem, class SelectionStrategy = typename Problem::PreferedSelectionStrategy >
class QpSolver
{
public:
	QpSolver(
		Problem& problem
	):m_problem(problem){}

	void solve(
		QpStoppingCondition& stop,
		QpSolutionProperties* prop = NULL
	){
		double start_time = Timer::now();
		unsigned long long iter = 0;
		unsigned long long shrinkCounter = 0;

		SelectionStrategy workingSet;

		// decomposition loop
		for(;;){
			//stop if iterations exceed
			if( iter == stop.maxIterations){
				if (prop != NULL) prop->type = QpMaxIterationsReached;
				break;
			}
			//stop if the maximum running time is exceeded
			if (stop.maxSeconds < 1e100 && (iter+1) % 1000 == 0 ){
				double current_time = Timer::now();
				if (current_time - start_time > stop.maxSeconds){
					if (prop != NULL) prop->type = QpTimeout;
					break;
				}
			}
			// select a working set and check for optimality
			std::size_t i = 0, j = 0;
			if (workingSet(m_problem,i, j) < stop.minAccuracy){
				m_problem.unshrink();
				if(m_problem.checkKKT() < stop.minAccuracy){
					if (prop != NULL) prop->type = QpAccuracyReached;
					break;
				}
				m_problem.shrink(stop.minAccuracy);
				workingSet(m_problem,i,j);
				workingSet.reset();
			}
			
			//update smo with the selected working set
			m_problem.updateSMO(i,j);
			
			//do a shrinking every 1000 iterations. if variables got shrink
			//notify working set selection
			if(shrinkCounter == 0 && m_problem.shrink(stop.minAccuracy)){
				shrinkCounter = std::max<std::size_t>(1000,m_problem.dimensions());
				workingSet.reset();
			}
			iter++;
			shrinkCounter--;
		}

		if (prop != NULL)
		{
			double finish_time = Timer::now();
			
			std::size_t i = 0, j = 0;
			prop->accuracy = workingSet(m_problem,i, j);
			prop->value = m_problem.functionValue();
			prop->iterations = iter;
			prop->seconds = finish_time - start_time;
		}
	}

protected:
	Problem& m_problem;
};

}
#endif
