//===========================================================================
/*!
 *  \brief General and specialized quadratic program classes and a generic solver.
 *
 *  \author  T. Glasmachers, O.Krause
 *  \date    2007-2013
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
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

	/// \brief constructor which initializes a C-SVM problem
	GeneralQuadraticProblem(MatrixType& quadratic, Data<unsigned int> const& labels, double C)
	: quadratic(quadratic)
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

		for(std::size_t i = 0; i!= dimensions(); ++i){
			unsigned int label = labels.element(i);
			permutation[i] = i;
			diagonal(i) = quadratic.entry(i, i);
			linear(i) = label? 1.0:-1.0;
			boxMin(i) = label? 0.0:-C;
			boxMax(i) = label? C : 0.0;
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
		SIZE_CHECK(dimensions() == this->linear.size());
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

template<class Problem>
class BaseShrinkingProblem : private Problem{
public:
	typedef typename Problem::QpFloatType QpFloatType;
	typedef typename Problem::MatrixType MatrixType;
	typedef typename Problem::PreferedSelectionStrategy PreferedSelectionStrategy;
	template<class SVMProblem>
	BaseShrinkingProblem(SVMProblem& problem, bool shrink=true)
	: Problem(problem)
	, m_isUnshrinked(false)
	, m_shrink(shrink)
	, m_shrinkCounter(std::min<std::size_t>(this->dimensions(),1000))
	, m_gradientEdge(problem.linear){}

	using Problem::dimensions;
	using Problem::active;
	using Problem::linear;
	using Problem::quadratic;
	using Problem::boxMin;
	using Problem::boxMax;
	using Problem::alpha;
	using Problem::diagonal;
	using Problem::gradient;
	using Problem::functionValue;
	using Problem::getUnpermutedAlpha;
	using Problem::isUpperBound;
	using Problem::isLowerBound;
	using Problem::deactivateVariable;
	using Problem::activateVariable;

	///\brief Does an update of SMO given a working set with indices i and j.
	void updateSMO(std::size_t i, std::size_t j){
		SIZE_CHECK(i < active());
		SIZE_CHECK(j < active());
		double ai = alpha(i);
		double aj = alpha(j);
		Problem::updateSMO(i,j);//call base class update of alpha values
		
		if(!m_shrink) return;
		
		//update the shrinked variables if they got changed
		if(ai != alpha(i))
			updateGradientEdge(i, ai, alpha(i));
		if(i != j && aj != alpha(j))//take care that we are not in a working-set-size=1 case 
			updateGradientEdge(j, aj, alpha(j));
	}

	bool shrink(double epsilon){
		if(m_shrink){
			//check if shrinking is necessary
			--m_shrinkCounter;
			if(m_shrinkCounter != 0) return false;
			m_shrinkCounter = std::min<std::size_t>(this->dimensions(),1000);
			
			return doShrink(epsilon);
		}
		return false;
	}

	///\brief Unshrink the problem
	void unshrink(){
		if (active() == dimensions()) return;
		m_isUnshrinked = true;
		
		// recompute the gradient of the whole problem.
		// we assume here that all shrinked variables are on the border of the problem.
		// the gradient of the active components is already correct and
		// we store the gradient of the subset of variables which are on the
		// borders of the box for the whole set.
		// Thus we only have to recompute the part of the gradient which is
		// based on variables in the active set which are not on the border.
		for (std::size_t a = active(); a < dimensions(); a++) 
			Problem::m_gradient(a) = m_gradientEdge(a);

		for (std::size_t i = 0; i < active(); i++)
		{
			if (isUpperBound(i) || isLowerBound(i)) continue;//alpha value is already stored in gradientEdge
			
			QpFloatType* q = quadratic().row(i, 0, dimensions());
			for (std::size_t a = active(); a < dimensions(); a++) 
				Problem::m_gradient(a) -= alpha(i) * q[a] ;
		}

		this->m_active = dimensions();
	}

	void setShrinking(bool shrinking){
		m_shrink = shrinking;
		if(!shrinking)
			unshrink();
	}

	//~ void modifyStep(std::size_t i, std::size_t j, double diff){
		//~ SIZE_CHECK(i < dimensions());
		//~ double ai = alpha(i);
		//~ double aj = alpha(j);
		//~ Problem::modifyStep(i,j,diff);
		
		//~ if(!m_shrink) return;
		
		//~ if(ai != alpha(i))
			//~ updateGradientEdge(i, ai, alpha(i));
		//~ if(i != j && aj != alpha(j))//take care that we are not in a working-set-size=1 case 
			//~ updateGradientEdge(j, aj, alpha(j));
	//~ }

protected:
	///\brief Shrink the variable from the problem.
	void shrinkVariable(std::size_t i){
		SIZE_CHECK(i < active());
		Problem::flipCoordinates(i,active()-1);
		std::swap( m_gradientEdge[i], m_gradientEdge[active()-1]);
		--this->m_active;
	}
	
	virtual bool doShrink(double epsilon)=0;
	
	bool m_isUnshrinked;

private:
	///\brief Update the edge-part of the gradient
	void updateGradientEdge(std::size_t i, double oldAlpha, double newAlpha){
		SIZE_CHECK(i < active());
		bool isInsideOld = oldAlpha > boxMin(i) && oldAlpha < boxMax(i);
		bool isInsideNew = newAlpha > boxMin(i) && newAlpha < boxMax(i);
		//check if variable is relevant at all, that means that old and new alpha value are inside
		//or old alpha is 0 and new alpha inside
		if( (oldAlpha == 0 || isInsideOld) && isInsideNew  )
			return;

		//compute change to the gradient
		double diff = 0;
		if(!isInsideOld)//the value was on a border, so remove it's old influeence to the gradient
			diff -=oldAlpha;
		if(!isInsideNew){//variable entered boundary or changed from one boundary to another
			diff  += newAlpha;
		}

		QpFloatType* q = quadratic().row(i, 0, dimensions());
		for(std::size_t a = 0; a != dimensions(); ++a){
			m_gradientEdge(a) -= diff*q[a];
		}
	}
	
	///\brief true if shrinking is to be used.
	bool m_shrink;
	
	///\brief Number of iterations until next shrinking.
	std::size_t m_shrinkCounter;

	///\brief Stores the gradient of the alpha dimeensions which are either 0 or C
	RealVector m_gradientEdge;
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
				if(workingSet(m_problem,i,j) < stop.minAccuracy){
					if (prop != NULL) prop->type = QpAccuracyReached;
					break;
				}
				m_problem.shrink(stop.minAccuracy);
				workingSet.reset();
			}
			
			//update smo with the selected working set
			m_problem.updateSMO(i,j);
			if(m_problem.shrink(stop.minAccuracy))
				workingSet.reset();
			iter++;
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
