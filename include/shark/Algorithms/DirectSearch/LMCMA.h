/*!
 * \brief       -
 *
 * \author      Thomas Voss and Christian Igel
 * \date        April 2014
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_LM_CMA_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_LM_CMA_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <shark/Algorithms/DirectSearch/Individual.h>

#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/PopulationBasedStepSizeAdaptation.h>
#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/ElitistSelection.h>



namespace shark {

namespace detail{

///\brief Approximates a Limited Memory Cholesky Matrix from a stream of samples.
///
/// GIven a set of points \f$ v_i\f$, produces an approximation of the cholesky factor of a matrix:
/// \f[ AA^T=C= (1-\alpha) C^{t-1} + \alpha* x_{j_t} x_{j_t}^T \f]
/// here the \f j_t \f$ are chosen such to have an approximate distance \f$ N_{steps} \f$. It is assumed
/// that the \f$x_i \f$ are correlated and thus a big \f$ N_{steps} \f$ tris to get points which are less 
/// correlated. The matrix keeps a set of vectors and decides at every step which is will discard.
///
/// This is the corrected algorithm as proposed in 
/// Ilya Loshchilov, "A Computationally Efficient Limited Memory CMA-ES for Large Scale Optimization"
///
class IncrementalCholeskyMatrix{
public:
	IncrementalCholeskyMatrix(){}
	void init (double alpha,std::size_t dimensions, std::size_t numVectors, std::size_t Nsteps){
		m_vArr.resize(numVectors,dimensions);
		m_pcArr.resize(numVectors,dimensions);
		m_b.resize(numVectors);
		m_d.resize(numVectors);
		m_l.resize(numVectors);
		m_j.resize(0);//nothing stored at the bginning
		m_Nsteps = Nsteps;
		m_maxStoredVectors = numVectors;
		m_counter = 0;
		m_alpha = alpha;
		
		m_vArr.clear();
		m_pcArr.clear();
		m_b.clear();
		m_d.clear();
		m_l.clear();
	}

	//computes x = Az
	template<class T>
	void prod(RealVector& x, T const& z)const{
		x = z;
		double a = std::sqrt(1-m_alpha);
		for(std::size_t j=0; j != m_j.size(); j++){
			std::size_t jcur = m_j[j];	
			double k = m_b(jcur) *inner_prod(row(m_vArr,jcur),z);
			noalias(x) = a*x+k*row(m_pcArr,jcur);
		}
	}
	
	//computes x= A^{-1}z
	template<class T>
	void inv(RealVector& x, T const& z)const{
		inv(x,z,m_j.size());
	}
	
	void update(RealVector const& newPc){
		std::size_t imin = 0;//the index of the removed point
		if (m_j.size() < m_maxStoredVectors)
		{
			std::size_t index = m_j.size();
			m_j.push_back(index);
			imin = index;
		}
		else
		{
			//find the largest "age"gap between neighbouring points (i.e. the time between insertion)
			//we want to remove the smallest gap as to make the
			//time distances as equal as possible
			std::size_t dmin = m_l[m_j[1]] - m_l[m_j[0]];
			imin = 1;
			for(std::size_t j=2; j != m_j.size(); j++)
			{
				std::size_t dcur = m_l[m_j[j]] - m_l[m_j[j-1]];
				if (dcur < dmin)
				{
					dmin = dcur;
					imin = j;
				}
			}
			//if the gap is bigger than Nsteps, we remove the oldest point to
			//shrink it.
			if (dmin >= m_Nsteps)
				imin = 0;
			//we push all points backwards and append the freed index to the end of the list
			if (imin != m_j.size()-1)
			{
				std::size_t sav = m_j[imin];
				for(std::size_t j = imin; j != m_j.size()-1; j++)
					m_j[j] = m_j[j+1];
				m_j.back() = sav;
			}
		}
		//set the values of the new added index
		int newidx = m_j.back();
		m_l[newidx] = m_counter;
		noalias(row(m_pcArr,newidx)) = newPc;
		++m_counter;
	
		// this procedure recomputes v vectors correctly, in the original LM-CMA-ES they were outdated/corrupted.
		// all vectors v_k,v_{k+1},...,v_m are corrupted where k=j_imin. it also computes the proper v and b/d values for the newest
		// inserted vector
		RealVector v;
		for(std::size_t i = imin; i != m_j.size(); ++i)
		{
			int index = m_j[i];
			inv(v,row(m_pcArr,index),i);
			noalias(row(m_vArr,index)) = v;

			double normv2 = norm_sqr(row(m_vArr,index));
			double c = std::sqrt(1.0-m_alpha);
			double f = std::sqrt(1+m_alpha/(1-m_alpha)*normv2);
			m_b[index] = c/normv2*(f-1);
			m_d[index] = 1/(c*normv2)*(1-1/f);
		}
	}
	
private:
	template<class T>
	void inv(RealVector& x, T const& z,std::size_t k)const{
		x = z;
		double c= 1.0/std::sqrt(1-m_alpha);
		for(std::size_t j=0; j != k; j++){// O(m*n)
			std::size_t jcur = m_j[j];
			double k = m_d(jcur) * inner_prod(row(m_vArr,jcur),x);
			noalias(x) = c*x - k*row(m_vArr,jcur);
		}
	}

	//variables making up A
	RealMatrix m_vArr;
	RealMatrix m_pcArr;
	RealVector m_b;
	RealVector m_d;
	
	//index variables for computation of A
	std::vector<std::size_t> m_j;
	std::vector<std::size_t> m_l;
	std::size_t m_Nsteps;
	std::size_t m_maxStoredVectors;
	std::size_t m_counter;
	
	double m_alpha;
};
}

/// \brief Implements a Limited-Memory-CMA
///
/// This is the algorithm as proposed in 
/// Ilya Loshchilov, "A Computationally Efficient Limited Memory CMA-ES for Large Scale Optimization"
/// with a few corrections regarding the covariance matrix update.
///
/// The algorithm stores a subset of previous evolution path vectors and approximates the covariance
/// matrix based on this. This algorithm only requires O(nm) memory, where n is the dimensionality
/// and n the problem dimensionality. To be more exact, 2*m vectors of size n are stored to calculate
/// the matrix-vector product with the choelsky factor of the covariance matrix in O(mn). 
///
/// The algorithm uses the population based step size adaptation strategy as proposed in
/// the same paper.
class LMCMA: public AbstractSingleObjectiveOptimizer<RealVector >
{
public:
	/// \brief Default c'tor.
	LMCMA(){
		m_features |= REQUIRES_VALUE;
	}
	
	
	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LMCMA-ES"; }

	/// \brief Calculates lambda for the supplied dimensionality n.
	unsigned suggestLambda( unsigned int dimension ) {
		unsigned lambda = unsigned( 4. + ::floor( 3. * ::log( static_cast<double>( dimension ) ) ) ); // eq. (44)
		// heuristic for small search spaces
		lambda = std::max<unsigned int>( 5, std::min( lambda, dimension ) );
		return( lambda );
	}

	/// \brief Calculates mu for the supplied lambda and the recombination strategy.
	double suggestMu( unsigned int lambda) {
		return lambda / 2.; // eq. (44)
	}
	///\brief Configures the algorithm based on the supplied configuration.
	void configure( const PropertyTree & node ){}

	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	
	/// \brief Initializes the algorithm for the supplied objective function.
	void init( ObjectiveFunctionType const& function, SearchPointType const& p) {
		unsigned int lambda = suggestLambda( p.size() );
		unsigned int mu = suggestMu(  lambda );
		init( function,
			p,
			lambda,
			mu,
			1.0/std::sqrt(double(p.size()))
		);
	}

	/// \brief Initializes the algorithm for the supplied objective function.
	void init( 
		ObjectiveFunctionType const& function, 
		SearchPointType const& initialSearchPoint,
		unsigned int lambda, 
		double mu,
		double initialSigma
	) {
		m_numberOfVariables = function.numberOfVariables();
		m_lambda = lambda;
		m_mu = static_cast<unsigned int>(::floor(mu));

		//set initial point
		m_mean = initialSearchPoint;
		m_best.point = initialSearchPoint;
		m_best.value = function(initialSearchPoint);
		
		//init step size adaptation
		m_stepSize.init(initialSigma);
			
		//weighting of the mu-best individuals
		m_weights.resize(m_mu);
		for (unsigned int i = 0; i < m_mu; i++){
			m_weights(i) = ::log(mu + 0.5) - ::log(1. + i);
		}
		m_weights /= sum(m_weights);
		
		// learning rates
		m_muEff = 1. / sum(sqr(m_weights)); // equal to sum(m_weights)^2 / sum(sqr(m_weights))
		double c1 = 1/(10*std::log(m_numberOfVariables+1.0));
		m_cC =1.0/m_lambda;
		
		//init variables for covariance matrix update
		m_evolutionPathC = blas::repeat(0.0,m_numberOfVariables);
		m_A.init(c1,m_numberOfVariables,lambda,lambda);
	}

	/// \brief Executes one iteration of the algorithm.
	void step(ObjectiveFunctionType const& function){

		std::vector< Individual<RealVector, double, RealVector> > offspring( m_lambda );

		PenalizingEvaluator penalizingEvaluator;
		for( unsigned int i = 0; i < offspring.size(); i++ ) {
			createSample(offspring[i].searchPoint(),offspring[i].chromosome());
		}
		penalizingEvaluator( function, offspring.begin(), offspring.end() );

		// Selection and parameter update
		// opposed to normal CMA selection, we don't remove any indidivudals but only order
		// them by rank to allow the use of the population based strategy.
		std::vector< Individual<RealVector, double, RealVector> > parents( lambda() );
		ElitistSelection<FitnessExtractor> selection;
		selection(offspring.begin(),offspring.end(),parents.begin(), parents.end());
		updateStrategyParameters( parents );

		//update the best solution found so far.
		m_best.point= parents[ 0 ].searchPoint();
		m_best.value= parents[ 0 ].unpenalizedFitness();
	}

	/// \brief Accesses the current step size.
	double sigma() const {
		return m_stepSize.stepSize();
	}

	/// \brief Accesses the current population mean.
	RealVector const& mean() const {
		return m_mean;
	}

	/// \brief Accesses the current weighting vector.
	RealVector const& weights() const {
		return m_weights;
	}

	/// \brief Accesses the evolution path for the covariance matrix update.
	RealVector const& evolutionPath() const {
		return m_evolutionPathC;
	}
	
	/// \brief Returns the size of the parent population \f$\mu\f$.
	unsigned int mu() const {
		return m_mu;
	}
	
	/// \brief Returns a mutabl rference to the size of the parent population \f$\mu\f$.
	unsigned int& mu(){
		return m_mu;
	}
	
	/// \brief Returns a immutable reference to the size of the offspring population \f$\mu\f$.
	unsigned int lambda()const{
		return m_lambda;
	}

	/// \brief Returns a mutable reference to the size of the offspring population \f$\mu\f$.
	unsigned int & lambda(){
		return m_lambda;
	}

private:
	/// \brief Updates the strategy parameters based on the supplied offspring population.
	void updateStrategyParameters( std::vector<Individual<RealVector, double, RealVector> > const& offspring ) {
		//line 8, creation of the new mean (but not updating the mean of the distribution
		RealVector m( m_numberOfVariables, 0. );
		for( unsigned int j = 0; j < mu(); j++ ){
			noalias(m) += m_weights( j ) * offspring[j].searchPoint();
		}
		
		//update evolution path, line 9
		noalias(m_evolutionPathC) = (1. - m_cC ) * m_evolutionPathC + std::sqrt( m_cC * (2. - m_cC) * m_muEff ) * (m - m_mean) / sigma();
		
		//update mean now, as oldmean is not needed any more (line 8 continued)
		m_mean = m;
		
		//corrected version of lines 10-14- the covariance matrix adaptation
		//we replace one vector that makes up the approximation of A by the newly updated evolution path
		m_A.update(m_evolutionPathC);
		
		//update the step size using the population success rule, line 15-18
		m_stepSize.update(offspring);

	}
	
	/// \brief Creates a vector-sample pair x=Az, where z is a gaussian random vector.
	void createSample(RealVector& x,RealVector& z)const{
		x.resize(m_numberOfVariables);
		z.resize(m_numberOfVariables);
		for(std::size_t i = 0; i != m_numberOfVariables; ++i){
			z(i) = Rng::gauss(0,1);
		}
		m_A.prod(x,z);
		noalias(x) = sigma()*x +m_mean;
	}
	
	unsigned int m_numberOfVariables; ///< Stores the dimensionality of the search space.
	unsigned int m_mu; ///< The size of the parent population.
	unsigned int m_lambda; ///< The size of the offspring population, needs to be larger than mu.

	double m_cC;///< learning rate of the evolution path
	

	detail::IncrementalCholeskyMatrix m_A;
	PopulationBasedStepSizeAdaptation m_stepSize;///< step size adaptation for the step size sigma()
	
	RealVector m_mean; ///< current mean of the distribution
	RealVector m_weights;///< weighting for the mu best individuals
	double m_muEff;///< effective sample size for the weighted samples

	RealVector m_evolutionPathC;///< 
	
	
};

}

#endif
