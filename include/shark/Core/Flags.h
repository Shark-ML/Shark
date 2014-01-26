//===========================================================================
/*!
 * 
 *
 * \brief       Flexible and extensible mechanisms for holding flags.
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
//===========================================================================
#ifndef SHARK_CORE_FLAGS_H
#define SHARK_CORE_FLAGS_H

#include <shark/Core/Exception.h>
#include <shark/Core/ISerializable.h>

namespace shark {


///
/// \brief Flexible and extensible mechanisms for holding flags.
///
/// \par
/// The world's most airbrushed integer ever...
///
/// \par
/// This class encapsulates a flexible mechanism for holding flags.
/// Its templatization makes it possible to base it on any base
/// type, while unsigned int will be the most common choice. This
/// mechanism makes it possible, in principle, to support as many
/// flags as needed. Furthermore, classes may extend the flags
/// defined by their superclass.
///
template<typename Flag>
class TypedFlags : public ISerializable {
public:
	TypedFlags() : m_flags( 0 ) { }
	TypedFlags(TypedFlags const& other) : m_flags(other.m_flags) { }

        virtual ~TypedFlags() {}

	inline TypedFlags<Flag> & operator = ( TypedFlags<Flag> const& rhs ) {
		m_flags = rhs.m_flags;
		return( *this );
	}

	inline void set( Flag f ) {
		m_flags |= f;
	}

	inline void setAll() {
		m_flags |= ~0;
	}

	inline void reset() {
		m_flags = 0;
	}

	inline void reset( Flag f ) {
		m_flags &= ~f;
	}

	inline bool test( Flag f ) const {
		return ( m_flags & f) == (unsigned int)f;
	}

	inline bool operator&( Flag f ) const {
		return ( m_flags & f) == (unsigned int)f;
	}

	inline TypedFlags<Flag> & operator|=( Flag f ) {
		m_flags |= f;
		return( *this );
	}
	
	inline TypedFlags<Flag> & operator|=(const TypedFlags<Flag>& flags ) {
		m_flags |= flags.m_flags;
		return( *this );
	}

	inline TypedFlags<Flag> operator|( Flag f ) const {
		TypedFlags<Flag> copy( *this );
		copy |= f;
		return( copy );
	}
	inline TypedFlags<Flag> operator|(const TypedFlags<Flag>& flags ) const {
		TypedFlags<Flag> copy( *this );
		copy |= flags;
		return copy;
	}

	virtual void read( InArchive & archive )
	{ archive & m_flags; }

	virtual void write( OutArchive & archive ) const
	{ archive & m_flags; }

protected:
	unsigned int m_flags;
};


///
/// \brief Exception indicating the attempt to use a feature which is not supported
///
template<class Feature>
class TypedFeatureNotAvailableException : public Exception {
public:
	TypedFeatureNotAvailableException( Feature feature, const std::string & file = std::string(), unsigned int line = 0 )
	: Exception( "Feature not available", file, line ),
	m_feature( feature ) {}
	TypedFeatureNotAvailableException( const std::string & message, Feature feature, const std::string & file = std::string(), unsigned int line = 0 )
	: Exception( message, file, line ),
	m_feature( feature ) {}

	Feature feature() const {
		return m_feature ;
	}
protected:
	Feature m_feature;
};

}

namespace boost {
namespace serialization {

template< typename T >
struct tracking_level< shark::TypedFlags<T> > {
    typedef mpl::integral_c_tag tag;
    BOOST_STATIC_CONSTANT( int, value = track_always );
};

}
}

#define SHARK_FEATURE_INTERFACE \
typedef TypedFlags<Feature> Features;\
protected:\
Features m_features;\
public:\
const Features & features() const {\
	return( m_features );\
}\
virtual void updateFeatures(){}\
typedef TypedFeatureNotAvailableException<Feature> FeatureNotAvailableException

/// Throws an Exception when called.
/// This macro should be used in default implementations of the interface.
/// This define also checks first whether the feature is set to true inside the class.
/// If this is the case then we have encountered a programming mistake - so we assert instead.
#define SHARK_FEATURE_EXCEPTION(FEATURE) \
{assert(!(this->features()&FEATURE));\
throw FeatureNotAvailableException("Class does not support Feature " #FEATURE, FEATURE,__FILE__, __LINE__);}
/// Same as SHARK_FEATURE_EXCEPTION, but used when called from a derived class.
/// Assumes that a typedef "base_type" for the Baseclass exists
#define SHARK_FEATURE_EXCEPTION_DERIVED(FEATURE) \
{assert(!(this->features()&base_type::FEATURE));\
throw typename base_type::FeatureNotAvailableException("Class does not support Feature " #FEATURE, base_type::FEATURE,__FILE__, __LINE__);}

/// Checks whether the feature is available, if not, it throws an exception.
#define SHARK_FEATURE_CHECK(FEATURE)\
if(!(this->features()&base_type::FEATURE)){SHARK_FEATURE_EXCEPTION_DERIVED(FEATURE);}
#endif // SHARK_CORE_FLAGS_H
