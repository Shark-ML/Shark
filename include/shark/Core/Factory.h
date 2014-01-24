/*!
 * 
 * \file        Factory.h
 *
 * \brief       Implements the factory pattern.
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010
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
#ifndef SHARK_CORE_FACTORY_H
#define SHARK_CORE_FACTORY_H

#include <map>
#include <iostream>

namespace shark {

	/**
	* \brief Implements the factory pattern.
	* \tparam ClassType The type to be constructed by the factory.
	* \tparam TagType The type identifying a type.
	*/
	template<typename ClassType, typename TagType>
	class Factory {
	public:
		typedef ClassType class_type;
		typedef TagType tag_type;

		/**
		* \brief Explicit interface for a factory.
		*/
		struct AbstractFactory {

			/**
			* \brief Default virtual d'tor.
			*/
			virtual ~AbstractFactory() {}

			/**
			* \brief Instantiates an object of type class_type. Needs to
			* be implemented by implementations of the interface.
			*/
			virtual class_type * create() = 0;
		};

		/**
		* \brief The map type.
		*/
		typedef std::map< tag_type, AbstractFactory* > map_type;

		/**
		* \brief Forwarded typedef for const iterator.
		*/
		typedef typename map_type::const_iterator const_iterator;

		/**
		* \brief Accesses the global instance of the factory.
		*/
		static Factory & instance() {
			static Factory factory;
			return( factory );
		}

		/**
		* \brief Queries the factory for the supplied tag.
		* \param [in] tag Tag identifying the type.
		* \returns An instance of abstract factory or NULL.
		*/
		class_type * operator[]( const tag_type & tag ) {
			typename map_type::iterator it = m_factoryMap.find( tag );

			if( it == m_factoryMap.end() )
				return( NULL );

			return( it->second->create() );
		}

		/**
		* \brief Registers the supplied AbstractFactory instance for the given tag.
		* \param [in] tag The tag to associate the AbstractFactory instance with.
		* \param [in] aFactory The AbstractFactory of the instance.
		*/
		void registerType( const tag_type & tag, AbstractFactory * aFactory ) {
			if( aFactory == NULL )
				return;

			m_factoryMap[ tag ] = aFactory;
		}

		/**
		* \brief Unregisters the supplied AbstractFactory instance for the given tag.
		* \param [in] tag The tag to remove from the factory.
		*/
		void unregisterType( const tag_type & tag ) {
			typename map_type::iterator it = m_factoryMap.find( tag );

			if( it != m_factoryMap.end() )
				m_factoryMap.erase( it );
		}

		/**
		* \brief Outputs the factory to the supplied stream.
		*/
		template<typename Stream>
		void print( Stream & s ) const {

			for( const_iterator it = m_factoryMap.begin(); it != m_factoryMap.end(); ++it )
				s << it->first << ": " << it->second << std::endl;
		}

		/**
		* \brief Returns an iterator pointing to the first valid element of the factory.
		*/
		const_iterator begin() const {
			return( m_factoryMap.begin() );
		}

		/**
		* \brief Returns an iterator pointing to the first invalid element of the factory.
		*/
		const_iterator end() const {
			return( m_factoryMap.end() );
		}

	protected:

		Factory() {}		

		// Empty
		Factory( const Factory< tag_type, class_type > & rhs );
		Factory< ClassType, TagType > & operator=( const Factory< ClassType, TagType > & rhs );
		bool operator==( const Factory< ClassType, TagType > & rhs ) const;

		map_type m_factoryMap;
	};

	/**
	* \brief Helper structure to allow for automatic registration of types with the
	* factory modelled by the template parameter.
	*
	* \tparam FactoryType The factory to register the type with.
	*/
	template<typename FactoryType>
	struct FactoryRegisterer {

		/**
		* \brief C'tor.
		*/
		FactoryRegisterer( const typename FactoryType::tag_type & tag, typename FactoryType::AbstractFactory * factory ) {
			FactoryType::instance().registerType( tag, factory );
		}
	};

	/**
	* \brief Type erase to ease implementing factories for custom types.
	*
	* \code
	*	class MyClass {
	*   ...
	*   };
	*
	*   typedef TypeErasedAbstractFactory< MyClass, FactoryType > MyClassFactory;
	* \endcode
	*/
	template<typename T, typename FactoryType>
	struct TypeErasedAbstractFactory : public FactoryType::AbstractFactory {

		typename FactoryType::class_type * create() {
			return( new T() );
		}

	};
}

#endif
