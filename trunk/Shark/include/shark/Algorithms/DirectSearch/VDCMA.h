/*!
 * \brief       Implements the VD-CMA-ES Algorithm
 *
 * \author     Oswin Krause
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

#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_VD_CMA_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_VD_CMA_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <shark/Algorithms/DirectSearch/Individual.h>

#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/ElitistSelection.h>


/// \brief Implements the VD-CMA-ES Algorithm
///
/// The VD-CMA-ES implements a restricted form of the CMA-ES where the covariance matrix is restriced to be (D+vv^T)
/// where D is a diagonal matrix and v a single vector. Therefore this variant is capable of large-scale optimisation
///
/// For more reference, see the paper
/// Akimoto, Y., A. Auger, and N. Hansen (2014). Comparison-Based Natural Gradient Optimization in High Dimension. 
/// To appear in Genetic and Evolutionary Computation Conference (GECCO 2014), Proceedings, ACM
///
/// The implementation differs from the paper to be closer to the reference implementation and to have better numerical
/// accuracy.
namespace shark {
class VDCMA : public AbstractSingleObjectiveOptimizer<RealVector >
{
private:
	double chi( unsigned int n ) {
		return( std::sqrt( static_cast<double>( n ) )*(1. - 1./(4.*n) + 1./(21.*n*n)) );
	}
public:

	/// \brief Default c'tor.
	VDCMA():m_initialSigma(0.0){
		m_features |= REQUIRES_VALUE;
	}
	
	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "VDCMA-ES"; }

	/// \brief Calculates lambda for the supplied dimensionality n.
	unsigned suggestLambda( unsigned int dimension ) {
		return unsigned( 4. + ::floor( 3. * ::log( static_cast<double>( dimension ) ) ) ); // eq. (44)
	}

	/// \brief Calculates mu for the supplied lambda and the recombination strategy.
	double suggestMu( unsigned int lambda) {
		return lambda / 2.; // eq. (44)
	}

	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	
	void init( ObjectiveFunctionType const& function, SearchPointType const& p) {
		unsigned int lambda = suggestLambda( p.size() );
		unsigned int mu = suggestMu(  lambda );
		double sigma = m_initialSigma;
		if(m_initialSigma == 0) sigma = 1.0/std::sqrt(double(p.size()));
		
		init( function,
			p,
			lambda,
			mu,
			sigma
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
		m_sigma = initialSigma;

		m_mean = blas::repeat(0.0,m_numberOfVariables);
		m_vn.resize(m_numberOfVariables);
		for(std::size_t i = 0; i != m_numberOfVariables;++i){
			m_vn(i) = Rng::uni(0,1.0/m_numberOfVariables);
		}
		m_normv = norm_2(m_vn);
		m_vn /= m_normv;
		
		m_D = blas::repeat(1.0,m_numberOfVariables);
		m_evolutionPathC = blas::repeat(0.0,m_numberOfVariables);
		m_evolutionPathSigma = blas::repeat(0.0,m_numberOfVariables);
		
		//set initial point
		m_mean = initialSearchPoint;
		m_best.point = initialSearchPoint;
		m_best.value = function(initialSearchPoint);
		
		m_counter = 0;//first iteration
			
		//weighting of the mu-best individuals
		m_weights.resize(m_mu);
		for (unsigned int i = 0; i < m_mu; i++){
			m_weights(i) = ::log(mu + 0.5) - ::log(1. + i);
		}
		m_weights /= sum(m_weights);
		
		// constants based on (4) and Step 3 in the algorithm
		m_muEff = 1. / sum(sqr(m_weights)); // equal to sum(m_weights)^2 / sum(sqr(m_weights))
		m_cSigma = 0.5/(1+std::sqrt(m_numberOfVariables/m_muEff));
		m_dSigma = 1. + 2. * std::max(0., std::sqrt((m_muEff-1.)/(m_numberOfVariables+1)) - 1.) + m_cSigma;

		m_cC = (4. + m_muEff / m_numberOfVariables) / (m_numberOfVariables + 4. +  2 * m_muEff / m_numberOfVariables);
		double correction = (m_numberOfVariables - 5.0)/6.0;
		m_c1 = correction*2 / (sqr(m_numberOfVariables + 1.3) + m_muEff);
		m_cMu = std::min(1. - m_c1, correction* 2 * (m_muEff - 2. + 1./m_muEff) / (sqr(m_numberOfVariables + 2) + m_muEff));
	}

	/// \brief Executes one iteration of the algorithm.
	void step(ObjectiveFunctionType const& function){

		std::vector< Individual<RealVector, double, RealVector> > offspring( m_lambda );

		PenalizingEvaluator penalizingEvaluator;
		for( unsigned int i = 0; i < offspring.size(); i++ ) {
			createSample(offspring[i].searchPoint(),offspring[i].chromosome());
		}
		penalizingEvaluator( function, offspring.begin(), offspring.end() );

		// Selection
		std::vector< Individual<RealVector, double, RealVector> > parents( m_mu );
		ElitistSelection<FitnessExtractor> selection;
		selection(offspring.begin(),offspring.end(),parents.begin(), parents.end());
		// Strategy parameter update
		m_counter++; // increase generation counter
		updateStrategyParameters( parents );

		m_best.point= parents[ 0 ].searchPoint();
		m_best.value= parents[ 0 ].unpenalizedFitness();
	}

	/// \brief Accesses the current step size.
	double sigma() const {
		return m_sigma;
	}

	/// \brief Accesses the current step size.
	void setSigma(double sigma) {
		m_sigma = sigma;
	}
	
	/// \brief set the initial step size of the algorithm. 
	///
	/// Sets the initial sigma at init to a given value. If this is 0, which it is
	/// by default, the default initialisation will be sigma= 1/sqrt(N) where N 
	/// is the number of variables to optimize.
	///
	/// this method is the prefered one instead of init()
	void setInitialSigma(double initialSigma){
		m_initialSigma = initialSigma;
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

	/// \brief Accesses the evolution path for the step size update.
	RealVector const& evolutionPathSigma() const {
		return m_evolutionPathSigma;
	}
	
	///\brief Returns the size of the parent population \f$\mu\f$.
	unsigned int mu() const {
		return m_mu;
	}
	
	///\brief Returns a mutabl reference to the size of the parent population \f$\mu\f$.
	unsigned int& mu(){
		return m_mu;
	}
	
	///\brief Returns a immutable reference to the size of the offspring population \f$\mu\f$.
	unsigned int lambda()const{
		return m_lambda;
	}

	///\brief Returns a mutable reference to the size of the offspring population \f$\mu\f$.
	unsigned int & lambda(){
		return m_lambda;
	}

private:
	/// \brief Updates the strategy parameters based on the supplied offspring population.
	///
	/// The chromosome stores the y-vector that is the step from the mean in D=1, sigma=1 space.
	void updateStrategyParameters( std::vector<Individual<RealVector, double, RealVector> >& offspring ) {
		RealVector m( m_numberOfVariables, 0. );
		RealVector z( m_numberOfVariables, 0. );
		
		for( unsigned int j = 0; j < offspring.size(); j++ ){
			noalias(m) += m_weights( j ) * offspring[j].searchPoint();
			noalias(z) += m_weights( j ) * offspring[j].chromosome();
		}
		//compute z from y= (1+(sqrt(1+||v||^2)-1)v_n v_n^T)z
		//therefore z= (1+(1/sqrt(1+||v||^2)-1)v_n v_n^T)y
		double b=(1/std::sqrt(1+sqr(m_normv))-1);
		noalias(z)+= b*inner_prod(z,m_vn)*m_vn;
		
		//update paths
		noalias(m_evolutionPathSigma) = (1. - m_cSigma)*m_evolutionPathSigma + std::sqrt( m_cSigma * (2. - m_cSigma) * m_muEff ) * z;
		// compute h_sigma
		double hSigLHS = norm_2( m_evolutionPathSigma ) / std::sqrt(1. - pow((1 - m_cSigma), 2.*(m_counter+1)));
		double hSigRHS = (1.4 + 2 / (m_numberOfVariables+1.)) * chi( m_numberOfVariables );
		double hSig = 0;
		if(hSigLHS < hSigRHS) hSig = 1.;
		noalias(m_evolutionPathC) = (1. - m_cC ) * m_evolutionPathC + hSig * std::sqrt( m_cC * (2. - m_cC) * m_muEff ) * (m - m_mean) / m_sigma;
		
		
		
		//we split the computation of s and t in the paper in two parts
		//we compute the first two steps and then compute the weighted mean over samples and
		//evolution path. afterwards we compute the rest using the mean result
		//the paper describes this as first computing S and T for all samples and compute the weighted
		//mean of that, but the reference implementation does it the other way to prevent numerical instabilities
		RealVector meanS(m_numberOfVariables,0.0);
		RealVector meanT(m_numberOfVariables,0.0);
		for(std::size_t j = 0; j != mu(); ++j){
			computeSAndTFirst(offspring[j].chromosome(),meanS,meanT,m_cMu*m_weights(j));
		}
		computeSAndTFirst(m_evolutionPathC/m_D,meanS,meanT,hSig*m_c1);
		
		//compute the remaining mean S and T steps
		computeSAndTSecond(meanS,meanT);
		
		//compute update to v and d
		noalias(m_D) += m_D*meanS;
		noalias(m_vn) = m_vn*m_normv+meanT/m_normv;//result is v and not vn
		//store the new v separately as vn and its norm
		m_normv = norm_2(m_vn);
		m_vn /= m_normv;
		
		//update step length
		m_sigma *= std::exp( (m_cSigma / m_dSigma) * (norm_2(m_evolutionPathSigma)/ chi( m_numberOfVariables ) - 1.) ); // eq. (39)
		
		//update mean
		m_mean = m;
	}
	
	//samples a point and stores additionally y=(x-m_mean)/(sigma*D)
	//as this is required for calculation later
	void createSample(RealVector& x,RealVector& y)const{
		x.resize(m_numberOfVariables);
		y.resize(m_numberOfVariables);
		for(std::size_t i = 0; i != m_numberOfVariables; ++i){
			y(i) = Rng::gauss(0,1);
		}
		double a = std::sqrt(1+sqr(m_normv))-1;
		a *= inner_prod(y,m_vn);
		noalias(y) +=a*m_vn;
		noalias(x) = m_mean+ m_sigma*m_D*y;
	}
	
	///\brief computes the sample wise first two steps of S and T of theorem 3.6 in the paper
	///
	/// S and T arguments accordingly
	void computeSAndTFirst(RealVector const& y, RealVector& s,RealVector& t, double weight )const{
		if(weight == 0) return;//nothing to do
		double yvn = inner_prod(y,m_vn);
		double normv2 = sqr(m_normv);
		double gammav = 1+normv2;
		//step 1
		noalias(s) += weight*(sqr(y) - (normv2/gammav*yvn)*(y*m_vn)-blas::repeat(1.0,m_numberOfVariables));
		//step 2
		noalias(t) += weight*(yvn*y - 0.5*(sqr(yvn)+gammav)*m_vn);
	}
		
	///\brief computes the last three steps of S and T of theorem 3.6 in the paper
	void computeSAndTSecond(RealVector& s,RealVector& t)const{
		RealVector vn2 = m_vn*m_vn;
		double normv2 = sqr(m_normv);
		double gammav = 1+normv2;
		//alpha of 3.5
		double alpha = sqr(normv2)+(2*gammav - std::sqrt(gammav))/max(vn2);
		alpha=std::sqrt(alpha);
		alpha /= 2+normv2;
		alpha = std::min(alpha,1.0);
		//constants (b,A) of 3.4
		double b=-(1-sqr(alpha))*sqr(normv2)/gammav+2*sqr(alpha);
		RealVector A= blas::repeat(2.0,m_numberOfVariables)-(b+2*sqr(alpha))*vn2;
		RealVector invAvn2= vn2/A;
		
		//step 3
		noalias(s) -= alpha/gammav*((2+normv2)*(m_vn*t)-normv2*inner_prod(m_vn,t)*vn2);
		//step 4
		noalias(s) = s/A -b*inner_prod(s,invAvn2)/(1+b*inner_prod(vn2,invAvn2))*invAvn2;
		//step 5
		noalias(t) -= alpha*((2+normv2)*(m_vn*s)-inner_prod(s,vn2)*m_vn);
	}
	
	unsigned int m_numberOfVariables; ///< Stores the dimensionality of the search space.
	unsigned int m_mu; ///< The size of the parent population.
	unsigned int m_lambda; ///< The size of the offspring population, needs to be larger than mu.

	double m_initialSigma;///0 by default which indicates initial sigma = 1/sqrt(N)
	double m_sigma;
	double m_cC; 
	double m_c1; 
	double m_cMu; 
	double m_cSigma;
	double m_dSigma;
	double m_muEff;

	RealVector m_mean;
	RealVector m_weights;

	RealVector m_evolutionPathC;
	RealVector m_evolutionPathSigma;
	
	///\brief normalised vector v 
	RealVector m_vn;
	///\brief norm of the vector v, therefore  v=m_vn*m_normv
	double m_normv;
	
	RealVector m_D;

	unsigned m_counter; ///< counter for generations
	
	
	
	
};

}

#endif
