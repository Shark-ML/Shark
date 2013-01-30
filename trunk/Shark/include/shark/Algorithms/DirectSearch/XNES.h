#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_XNES_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_XNES_H

#include <shark/Core/ResultSets.h>
#include <shark/Algorithms/DirectSearch/TypedIndividual.h>

#include <shark/LinAlg/fastOperations.h>
#include <shark/LinAlg/solveSystem.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

#include <boost/foreach.hpp>

namespace shark {

    /** \cond */
    namespace detail {
	
	template<typename MATRIX> 
	    MATRIX expm_pad(const MATRIX &H, const int p = 6) {
	    typedef typename MATRIX::value_type value_type;
	    typedef typename MATRIX::size_type size_type;
	    typedef double real_value_type;	// Correct me. Need to modify.
	    assert(H.size1() == H.size2());	
	    const size_type n = H.size1();
	    const boost::numeric::ublas::identity_matrix<value_type> I(n);
	    boost::numeric::ublas::matrix<value_type> U(n,n),H2(n,n),P(n,n),Q(n,n);
	    real_value_type norm = 0.0;

	    // Calcuate Pade coefficients  (1-based instead of 0-based as in the c vector)
	    boost::numeric::ublas::vector<real_value_type> c(p+2);
	    c(1)=1;  
	    for(size_type i = 1; i <= p; ++i) 
		c(i+1) = c(i) * ((p + 1.0 - i)/(i * (2.0 * p + 1 - i)));
	    // Calcuate the infinty norm of H, which is defined as the largest row sum of a matrix
	    for(size_type i=0; i<n; ++i) {
		real_value_type temp = 0.0;
		for(size_type j=0;j<n;j++)
		    temp += std::abs<real_value_type>(H(i,j)); // Correct me, if H is complex, can I use that abs?
		norm = std::max<real_value_type>(norm, temp);
	    }

	    if (norm == 0.0)  {
		throw( SHARKEXCEPTION( "Error! Null input in the routine EXPM_PAD.\n" ) );
	    }
		
	    // Scaling, seek s such that || H*2^(-s) || < 1/2, and set scale = 2^(-s)
	    int s = 0;
	    real_value_type scale = 1.0;
	    if(norm > 0.5) {
		s = std::max<int>(0, static_cast<int>((log(norm) / log(2.0) + 2.0)));
		scale /= static_cast<real_value_type>(std::pow(2.0, s));
		U.assign(scale * H); // Here U is used as temp value due to that H is const
	    }
	    // Horner evaluation of the irreducible fraction, see the following ref above.
	    // Initialise P (numerator) and Q (denominator) 
	    H2.assign( prod(U, U) );
	    Q.assign( c(p+1)*I );
	    P.assign( c(p)*I );
	    size_type odd = 1;
	    for( size_type k = p - 1; k > 0; --k) {
		if( odd == 1) {
		    Q = ( prod(Q, H2) + c(k) * I ); 
		} else {
		    P = ( prod(P, H2) + c(k) * I );
		}
		odd = 1 - odd;
	    }
	    if( odd == 1) {
			Q = ( prod(Q, U) );	
			Q -= P ;
	    } else {
			P = (prod(P, U));
			Q -= P;
	    }
	    // In origine expokit package, they use lapack ZGESV to obtain inverse matrix,
	    // and in that ZGESV routine, it uses LU decomposition for obtaing inverse matrix.
	    // Since in ublas, there is no matrix inversion template, I simply use the build-in
	    // LU decompostion package in ublas, and back substitute by myself.
	    //
	    //////////////// Implement Matrix Inversion ///////////////////////
//	    boost::numeric::ublas::permutation_matrix<size_type> pm(n); 
//	    int res = lu_factorize(Q, pm);
//	    if( res != 0 ) {
//		throw( SHARKEXCEPTION( "Error in the matrix inversion in template expm_pad.\n" ) );
//	    }
//	    H2 = I;  // H2 is not needed anymore, so it is temporary used as identity matrix for substituting.
//	    boost::numeric::ublas::lu_substitute(Q, pm, H2); 
//	    if( odd == 1)
//		U.assign( -(I + 2.0 * prod(H2, P)));
//	    else
//		U.assign( I + 2.0 * prod(H2, P));

		//OK: found the above code and figured, that it is most likely to be the solution of the System QU=P
		//followed by U= 2*U+I U= -(2*U+I)
		solveSystem(Q,U,P);
		if( odd == 1)
			noalias(U)= -(I + 2.0 * U);
	    else
			noalias(U)= I + 2.0 * U;
		
	    // Squaring 
	    for(size_t i = 0; i < s; ++i) {
			//U = (prod(U,U));
			//OK:using prod is likely to be extremely devastating for the performance of the algorithm
			//using H2 as tmeporary storage
			fast_prod(U,U,H2);
			swap(U,H2);
	    }
	    return U;
	}
	
	struct XNES {

	    struct FitnessComparator {
		template<typename T>
		bool operator()( const T & a, const T & b ) const {
		    static shark::FitnessTraits<T> ft;
		    return( ft( a, shark::tag::PenalizedFitness() )[0] < ft( b, shark::tag::PenalizedFitness() )[0] );
		}
	    };

	    struct Chromosome {
		RealVector m_z;
	    };

	    typedef TypedIndividual<RealVector, Chromosome> Individual;
	    typedef std::vector< Individual > Population;

	    Population m_pop;
	    RealVector m_util;

	    Individual m_bestIndividual;
	    RealVector m_cog;
	    RealMatrix m_A;

	    unsigned int m_mu;

	    double m_etaCog;
	    double m_etaA;

	XNES() : m_etaA( 0. ) {
	    init();
	}
	    
	    void init( unsigned int mu = 100, double etaCog = 1.0 ) {
		m_mu = mu;
		m_etaCog = etaCog;
		m_etaA = 0.;
	    }
	    
	    template<typename Function>
	    void init( const Function & f ) {
		const unsigned int n = f.numberOfVariables();

		RealVector initialPoints[3];
		f.proposeStartingPoint( initialPoints[0] );
		f.proposeStartingPoint( initialPoints[1] );
		f.proposeStartingPoint( initialPoints[2] );
		double d[3];
		d[0] = blas::norm_2( initialPoints[1] - initialPoints[0] );
		d[1] = blas::norm_2( initialPoints[2] - initialPoints[0] );
		d[2] = blas::norm_2( initialPoints[2] - initialPoints[1] );
		std::sort( d, d+3 );

		const blas::identity_matrix<double> I( n, n );
		m_A = d[1] * I;

		m_mu = static_cast<unsigned int>( 4. + ::floor(3.*::log(  static_cast<double>( n ) ) ) );
		m_util.resize( m_mu );
		m_pop.resize( m_mu );
		double sum = 0.;
		for( unsigned int i = 0; i < m_mu; i++ ) {
		    double u = std::max( ::log(m_mu/2.0 + 1.0) - ::log(i + 1.0), 0.0);
		    m_util( i ) = u;
		    sum += u;
		}
		m_util /= sum;
		m_util -= RealVector( m_util.size(), -1.0 / m_mu );

		m_etaCog = 1.0;
		m_etaA = 0.6 * (3.0 + ::log(static_cast<double>( n ))) / (n * ::sqrt( static_cast<double>( n ) ) );
				
		f.proposeStartingPoint( m_cog );
		*m_bestIndividual = m_cog;
		m_bestIndividual.fitness( shark::tag::UnpenalizedFitness() )[0] = f.eval( m_cog );
		m_bestIndividual.fitness( shark::tag::PenalizedFitness() )[0] = m_bestIndividual.fitness( shark::tag::UnpenalizedFitness() )[0];				
	    }

	    template<typename Function>
	    ResultSet< 
	        typename Function::SearchPointType, 
		typename Function::ResultType 
	    > step( const Function & f ) {

		unsigned int n = f.numberOfVariables();
		const RealMatrix I = blas::identity_matrix<double>( n, n );
				
		BOOST_FOREACH( Individual & ind, m_pop ) {
		    // ind.get<0>().m_z = shark::MultiVariateNormalDistribution::standard( n );
		    ind.get< 0 >().m_z.resize( n, true );
		    for( std::size_t i = 0; i < n; i++ )
			ind.get< 0 >().m_z( i ) = shark::Rng::gauss();
		    
		    *ind = m_cog + blas::prod( m_A, ind.get<0>().m_z );
		    ind.fitness( shark::tag::PenalizedFitness() )[0] = f.eval( *ind );
		    ind.fitness( shark::tag::UnpenalizedFitness() ) = ind.fitness( shark::tag::PenalizedFitness() );
		}

		std::sort( m_pop.begin(), m_pop.end(), FitnessComparator() );

		if( m_pop[0].fitness( shark::tag::PenalizedFitness() )[0] < m_bestIndividual.fitness( shark::tag::PenalizedFitness() )[0] ) {
		    m_bestIndividual = m_pop[0];
		}

		RealVector gDelta( n, 0. );
		RealMatrix gM = blas::zero_matrix<double>( n, n );

		for( unsigned int i = 0; i < m_pop.size(); i++ ) {
		    double u = m_util( i );
		    gDelta += u * m_pop[i].get<0>().m_z;
		    gM += u * ( blas::outer_prod( m_pop[i].get<0>().m_z, m_pop[i].get<0>().m_z ) - I );
		}

		std::cout << "A: " << m_A << std::endl;

		m_cog += blas::prod( m_A, m_etaCog * gDelta );
		RealMatrix M( m_A );
		m_A = blas::prod( m_A, shark::detail::expm_pad( RealMatrix( 0.5 * m_etaA * gM ) ) );

		static RealMatrix S( n, n ), T( n , n ); 
		static RealVector lambda( n );
		shark::svd( m_A, S, T, lambda );

		std::cout << "eta_mu, gDelta: " << m_etaCog << ", " << gDelta << std::endl;
		std::cout << "gM: " << gM << std::endl;
		std::cout << "u: " << m_util << std::endl;
		std::cout << "mu: " << m_cog << std::endl;

		return( shark::makeResultSet( *m_bestIndividual, m_bestIndividual.fitness( shark::tag::UnpenalizedFitness() )[0] ) );
	    }

	};
    }
    
    /** \endcond */

}

#endif
