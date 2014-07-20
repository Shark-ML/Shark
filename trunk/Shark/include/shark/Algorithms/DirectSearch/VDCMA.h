/*!
 * \brief       -
 *
 * \author      Thomas Voss and Christian Igel
 * \date        April 2014
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

#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_VD_CMA_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_VD_CMA_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <shark/Algorithms/DirectSearch/Individual.h>

#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/ElitistSelection.h>


namespace shark {
class VDCMA : public AbstractSingleObjectiveOptimizer<RealVector >
{
private:
	double chi( unsigned int n ) {
		return( std::sqrt( static_cast<double>( n ) )*(1. - 1./(4.*n) + 1./(21.*n*n)) );
	}
public:

	/// \brief Default c'tor.
	VDCMA(){
		m_features |= REQUIRES_VALUE;
	}
	
	

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "VDCMA-ES"; }

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
	/// \brief Configures the algorithm based on the supplied configuration.
	void configure( const PropertyTree & node ){}

	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	
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
		m_sigma = initialSigma;

		m_mean = blas::repeat(0.0,m_numberOfVariables);
		m_v.resize(m_numberOfVariables);
		for(std::size_t i = 0; i != m_numberOfVariables;++i){
			m_v(i) = Rng::uni(0,1.0/m_numberOfVariables);
		}
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
			m_weights(i) = ::log(mu + 0.5) - ::log(1. + i); // eq. (45)
		}
		m_weights /= sum(m_weights); // eq. (45)
		
		// constants based on (4) and Step 3 in the algorithm
		m_muEff = 1. / sum(sqr(m_weights)); // equal to sum(m_weights)^2 / sum(sqr(m_weights))
		m_cSigma = 0.5/(std::sqrt(m_numberOfVariables/m_muEff));
		m_dSigma = 1. + 2. * std::max(0., std::sqrt((m_muEff-1.)/(m_numberOfVariables+1)) - 1.) + m_cSigma;

		m_cC = (4. + m_muEff / m_numberOfVariables) / (m_numberOfVariables + 4. +  2 * m_muEff / m_numberOfVariables);
		double correction = (m_numberOfVariables - 5.0)/6.0;
		m_c1 = correction*2 / (sqr(m_numberOfVariables + 1.3) + m_muEff); // eq. (48)
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
	
	///\brief Returns a mutabl rference to the size of the parent population \f$\mu\f$.
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
	void updateStrategyParameters( std::vector<Individual<RealVector, double, RealVector> > const& offspring ) {
		RealVector m( m_numberOfVariables, 0. );
		RealVector z( m_numberOfVariables, 0. );
		
		for( unsigned int j = 0; j < offspring.size(); j++ ){
			noalias(m) += m_weights( j ) * offspring[j].searchPoint();
			noalias(z) += m_weights( j ) * offspring[j].chromosome();
		}

		// compute h_sigma
		double hSigLHS = norm_2( m_evolutionPathSigma ) / std::sqrt(1. - pow((1 - m_cSigma), 2.*(m_counter+1)));
		double hSigRHS = (1.4 + 2 / (m_numberOfVariables+1.)) * chi( m_numberOfVariables );
		double hSig = 0;
		if(hSigLHS < hSigRHS) hSig = 1.;
		
		//update paths
		m_evolutionPathC = (1. - m_cC ) * m_evolutionPathC + hSig * std::sqrt( m_cC * (2. - m_cC) * m_muEff ) * (m - m_mean) / m_sigma; // eq. (42)
		m_evolutionPathSigma = (1. - m_cSigma)*m_evolutionPathSigma + std::sqrt( m_cSigma * (2. - m_cSigma) * m_muEff ) * z; // eq. (40)
		
		//compute individualwise update of D and v
		RealVector Dnew = m_D;
		RealVector vnew = m_v;
		double normv = norm_2(m_v);
		RealVector s(m_numberOfVariables);
		RealVector t(m_numberOfVariables);
		//rank mu update
		for(std::size_t j = 0; j != mu(); ++j){
			computeSAndT(offspring[j].searchPoint(),s,t);
			
			//compute update to v and d
			double weight = m_cMu*m_weights(j) ;
			noalias(vnew) +=  (weight/normv) * t;
			noalias(Dnew) += weight * m_D*s;			
		}
		//rank 1 update similarly
		computeSAndT(m_mean+m_sigma*m_evolutionPathC,s,t);
		double weight = (1-hSig)*m_c1;
		noalias(vnew) +=  (weight/normv) * t;
		noalias(Dnew) += weight * m_D*s;		
		//~ std::cout<<s<<t<<std::endl;
		m_D = Dnew;
		m_v = vnew;
		
		//update step length
		m_sigma *= std::exp( (m_cSigma / m_dSigma) * (norm_2(m_evolutionPathSigma)/ chi( m_numberOfVariables ) - 1.) ); // eq. (39)
		
		//update mean
		m_mean = m;
	}
	
	void createSample(RealVector& x,RealVector& z)const{
		x.resize(m_numberOfVariables);
		z.resize(m_numberOfVariables);
		for(std::size_t i = 0; i != m_numberOfVariables; ++i){
			z(i) = Rng::gauss(0,1);
		}
		double normv2=norm_sqr(m_v);
		double a = std::sqrt(1+normv2)-1;
		a *= inner_prod(z,m_v)/normv2;
		noalias(x) = m_mean+ m_sigma*m_D*(z+a*m_v);
	}
	/// \brief Compute s and t as in theorem 3.6
	void computeSAndT(RealVector const& x, RealVector& s,RealVector& t)const{
		RealVector y = ((x - m_mean) / m_sigma)/m_D;
		double normv2 = norm_sqr(m_v);
		RealVector vbar = m_v/sqrt(normv2);
		RealVector vbarbar = vbar*vbar;
		double gammav = 1+normv2;
		//alpha from 3.5
		double alpha = sqr(normv2)+(2*gammav -std::sqrt(gammav))/max(abs(vbar));
		alpha /= 2+normv2;
		alpha = std::min(alpha,1.0);
		//constants of 3.4
		double b=-(1-sqr(alpha))*sqr(normv2)/gammav+2*sqr(alpha);
		RealVector A= blas::repeat(2.0,m_numberOfVariables)-(b+2*alpha)*sqr(vbar);
		
		double yvbar = inner_prod(y,vbar);
		//step 1
		noalias(s) = sqr(y) - (normv2*yvbar/gammav)*(y*vbar)-blas::repeat(1.0,m_numberOfVariables);
		//step 2 
		noalias(t) = yvbar*y - 0.5*(sqr(yvbar)+gammav)*vbar;
		//step 3
		noalias(s) -= alpha/gammav*((2+normv2)*(vbar*t)-normv2*inner_prod(vbar,t)*vbarbar);
		//step 4
		noalias(s) = s/A -b/(1+b*inner_prod(vbarbar,vbarbar/A))*inner_prod(s,vbarbar/A)*(vbarbar/A);
		//step 5
		noalias(t) -= alpha*((2+normv2)*(vbar*s)-inner_prod(s,vbarbar)*vbar);
	}
	
	unsigned int m_numberOfVariables; ///< Stores the dimensionality of the search space.
	unsigned int m_mu; ///< The size of the parent population.
	unsigned int m_lambda; ///< The size of the offspring population, needs to be larger than mu.

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
	
	RealVector m_v;
	RealVector m_D;

	unsigned m_counter; ///< counter for generations
	
	
	
	
};

}

#endif
