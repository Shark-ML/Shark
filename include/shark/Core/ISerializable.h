/*!
 * 
 *
 * \brief       ISerializable interface.
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <https://shark-ml.github.io/Shark/>
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
#ifndef SHARK_CORE_ISERIALIZABLE_H
#define SHARK_CORE_ISERIALIZABLE_H

#include <boost/version.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/tracking.hpp>

#if (BOOST_VERSION >= 105600) && (BOOST_VERSION <= 105800) 
	#define USE_SERIALIZATION_WORKAROUND
#endif
	
#ifdef USE_SERIALIZATION_WORKAROUND
	#include <boost/archive/text_iarchive.hpp>
	#include <boost/archive/text_oarchive.hpp>
#else
	#include <boost/archive/polymorphic_iarchive.hpp>
	#include <boost/archive/polymorphic_oarchive.hpp>
	#include <boost/archive/polymorphic_text_iarchive.hpp>
	#include <boost/archive/polymorphic_text_oarchive.hpp>
#endif
namespace shark {
	


#ifdef USE_SERIALIZATION_WORKAROUND
	//broken workaround for broken boost version
	/**
	* \brief Type of an archive to read from.
	*/
	typedef boost::archive::text_iarchive InArchive;
	typedef boost::archive::text_iarchive TextInArchive;
	

	/**
	* \brief Type of an archive to write to.
	*/
	typedef boost::archive::text_oarchive OutArchive;
	typedef boost::archive::text_oarchive TextOutArchive;
#else	
	/**
	* \brief Type of an archive to read from.
	*/
	typedef boost::archive::polymorphic_iarchive InArchive;
	typedef boost::archive::polymorphic_text_iarchive TextInArchive;

	/**
	* \brief Type of an archive to write to.
	*/
	typedef boost::archive::polymorphic_oarchive OutArchive;
	typedef boost::archive::polymorphic_text_oarchive TextOutArchive;
#endif
    /**
     * \brief Abstracts serializing functionality.
     * 
     * In order to integrate alien serialization libraries
     * with the components based on this interface, the classes
     * boost::archive::polymorphic_iarchive and boost::archive::polymorphic_oarchive
     * need to be implemented in terms of alien serialization library.
     */
    class ISerializable {
    public:
	/**
	 * \brief Virtual d'tor.
	 */ 
	virtual ~ISerializable() {}

	/**
	 * \brief Read the component from the supplied archive.
	 * \param [in,out] archive The archive to read from.
	 */
	virtual void read( InArchive & archive )
	{ }

	/**
	 * \brief Write the component to the supplied archive.
	 * \param [in,out] archive The archive to write to.
	 */
	virtual void write( OutArchive & archive ) const
	{ }

	/**
	 * \brief Versioned loading of components, calls read(...).
	 */
	void load(InArchive & archive,unsigned int version)
	{
	    (void) version;
	    read(archive);
	}

	/**
	 * \brief Versioned storing of components, calls write(...).
	 */
	void save(OutArchive & archive,unsigned int version)const
	{
	    (void) version;
	    write(archive);
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER();
    };
}

#include <vector>

namespace boost {
namespace serialization {

template< typename T >
struct tracking_level< std::vector<T> > {
    typedef mpl::integral_c_tag tag;
    BOOST_STATIC_CONSTANT( int, value = track_always );
};

}
}

#endif // SHARK_CORE_ISERIALIZABLE_H
