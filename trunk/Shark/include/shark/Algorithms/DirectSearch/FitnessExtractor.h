/*!
 * 
 * \file        FitnessExtractor.h
 *
 * \brief       Explicit traits for extracting fitness values from arbitrary types
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010-2011
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_FITNESS_EXTRACTOR_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_FITNESS_EXTRACTOR_H

#include <shark/Algorithms/DirectSearch/EA.h>
#include <shark/LinAlg/Base.h>

#include <vector>

namespace shark {

    /**
     * \brief Functor that returns its argument without conversion
     */
    struct IdentityFitnessExtractor {
		
	/**
	 * \brief Returns the argument without conversion
	 * \tparam Member Type of the element
	 * \param [in] member The member to return
	 */
	template<typename Member>
	const Member & operator()( const Member & member ) const {
	    return( member );
	}

	/**
	 * \brief Serializes/Deserializes the state of the extractor to the supplied archive.
	 * \tparam Archive Archive type, needs to be a model of a boost::serialization archive.
	 * \param [in,out] archive Archive to store to/load from.
	 * \param [in] version Currently unused.
	 */
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
	    (void) archive;
	    (void) version;
	}
    };

    /**
     * \brief Default fitness extractor
     */
    struct FitnessExtractor {

	/**
	 * \brief Calls the function fitness on the supplied element
	 * \tparam Member Element type; needs to provide a function fitness
	 * \param [in] member The element to extract fitness values from
	 */
	template<typename Member>
	inline const RealVector & operator()( const Member & member ) const {
	    return( member.fitness( tag::PenalizedFitness() ) );
	}

	/**
	 * \brief Serializes/Deserializes the state of the extractor to the supplied archive.
	 * \tparam Archive Archive type, needs to be a model of a boost::serialization archive.
	 * \param [in,out] archive Archive to store to/load from.
	 * \param [in] version Currently unused.
	 */
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
	    (void) archive;
	    (void) version;
	}

    };

    namespace soo {
	/**
	 * \brief Default fitness extractor
	 */
	struct FitnessExtractor {

	    /**
	     * \brief Calls the function fitness on the supplied element
	     * \tparam Member Element type; needs to provide a function fitness
	     * \param [in] member The element to extract fitness values from
	     */
	    template<typename Member>
	    inline const double & operator()( const Member & member ) const {
		return( member.fitness( tag::PenalizedFitness() )( 0 ) );
	    }

	    /**
	     * \brief Serializes/Deserializes the state of the extractor to the supplied archive.
	     * \tparam Archive Archive type, needs to be a model of a boost::serialization archive.
	     * \param [in,out] archive Archive to store to/load from.
	     * \param [in] version Currently unused.
	     */
	    template<typename Archive>
	    void serialize( Archive & archive, const unsigned int version ) {
		(void) archive;
		(void) version;
	    }

	};
    }
	
    /**
     * \brief Casting fitness extractor
     * \tparam CastTo Target type for static_cast
     */
    template<typename CastTo>
	struct CastingFitnessExtractor {

	    typedef RealVector fitness_type;

	    /**
	     * \brief Calls the function fitness on the supplied element
	     * \tparam Member Element type; needs to provide a function fitness
	     * \param [in] member The element to extract fitness values from
	     */
	    template<typename Member>
	    const fitness_type & operator()( const Member & member ) const {
		return( static_cast<const CastTo &>( member).fitness( tag::PenalizedFitness() ) );
	    }

	    /**
	     * \brief Serializes/Deserializes the state of the extractor to the supplied archive.
	     * \tparam Archive Archive type, needs to be a model of a boost::serialization archive.
	     * \param [in,out] archive Archive to store to/load from.
	     * \param [in] version Currently unused.
	     */
	    template<typename Archive>
	    void serialize( Archive & archive, const unsigned int version ) {
		(void) archive;
		(void) version;
	    }
	};
}

#endif
