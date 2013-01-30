//===========================================================================
/*!
*  \file AbstractMultiObjectiveFunction.h
*
*  \brief AbstractMultiObjectiveFunction
*
*  \author T.Voss, T. Glasmachers, O.Krause
*  \date 2010-2011
*
*  \par Copyright (c) 1998-2007:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_ABSTRACTMULTIOBJECTIVEFUNCTION_H
#define SHARK_OBJECTIVEFUNCTIONS_ABSTRACTMULTIOBJECTIVEFUNCTION_H


#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>

#include <shark/LinAlg/Base.h>

namespace shark {

	/**
	\brief Super class of all vector-valued objective functions for optimization and learning.

	\tparam SearchSpaceType The search space the function is defined upon.
	\tparam ResultT The objective space the function is defined upon.
	*/
	template <typename SearchSpaceType>
	class AbstractMultiObjectiveFunction : public AbstractObjectiveFunction<SearchSpaceType, RealVector> {
	public:
		typedef AbstractMultiObjectiveFunction< SearchSpaceType > this_type;
		typedef Factory< this_type, std::string > factory_type;
		typedef TypeErasedAbstractFactory< this_type, factory_type > abstract_factory_type;
		typedef FactoryRegisterer< factory_type > factory_registerer_type;	

	public:
		/**
		* \brief Constructs an instance with the supplied number of objectives.
		* \param [in] noObjectives The initial number of objectives.
		*/
		AbstractMultiObjectiveFunction( std::size_t noObjectives = 1 ) : m_noObjectives( noObjectives ) {
		}

		/** 
		* \brief Virtual destructor
		*/
		virtual ~AbstractMultiObjectiveFunction() {}

		/**
		* \brief Accesses the number this function provides.
		* \returns The number of objectives.
		*/
		virtual std::size_t noObjectives() const {
			return( m_noObjectives );
		}		

		/**
		* \brief Accesses the number of objectives this function provides.
		* \returns A mutable instance to the number of objectives, allows for l-value semantics.
		*/
		virtual std::size_t & noObjectives() {
			return( m_noObjectives );
		}

		/**
		* \brief Adjusts the number of objectives.
		* \param [in] noObjectives The new number of objectives.
		*/
		virtual void setNoObjectives( unsigned int noObjectives ) {
			m_noObjectives = noObjectives;
		}

	protected:
		std::size_t m_noObjectives; ///< Holds the number of objectives, default value: 1.
	};

}

#include <shark/Core/SearchSpaces/VectorSpace.h>

namespace shark {
	namespace moo {
		/** \brief Defines the default factory type for real-valued multi-objective optimization problems. */
		typedef Factory< AbstractMultiObjectiveFunction< VectorSpace< double > >, std::string > RealValuedObjectiveFunctionFactory;
	}
} 

/**
* \brief Convenience macro for registering multi-objective functions with a factory at compile-time.
*/
#define ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( Function, Factory ) \
	namespace Function ## _detail {\
		typedef TypeErasedAbstractFactory< Function, Factory > abstract_factory_type;\
		typedef FactoryRegisterer< Factory > factory_registerer_type;\
		static factory_registerer_type FACTORY_REGISTERER = factory_registerer_type( #Function, new abstract_factory_type() );\
	}\

#endif 
