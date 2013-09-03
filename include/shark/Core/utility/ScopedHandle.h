//===========================================================================
/*!
 *  \brief A scoped_ptr like container for C type handles
 *
 *
 *  \par
 *  This class provides RAII handle management to Shark
 *
 *
 *  \author  B. Li
 *  \date    2012
 *
 *  \par Copyright (c) 2010:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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

#ifndef SHARK_CORE_UTILITY_SCOPED_HANDLE_H
#define SHARK_CORE_UTILITY_SCOPED_HANDLE_H

#include "shark/Core/Exception.h"

#include <boost/assert.hpp>
#include <boost/bind/arg.hpp>
#include <boost/format.hpp>
#include <boost/function.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/noncopyable.hpp>

namespace shark {

/// A handle container which usually taking C style handle such as file Id
/// @tparam T The type of handle the container holds
template <typename T>
class ScopedHandle : private boost::noncopyable
{
public:

	/// The typedef of deleter for type a T
	typedef boost::function<void (const T&) > DeleterType;

	/// Type of verifier which should return true for valid handle, and false otherwise
	typedef boost::function<bool (const T&) > VerifierType;

	/// Constructor
	/// @param handle
	///     The handle container will hold
	/// @param deleter
	///     The deleter used for freeing a handle which should return true for valid handles, false otherwise
	/// @param handleDescription
	///     A description of handle for easy debugging in case of validation failure
	/// @param verifier
	///     Used for verifying passed in a handle. The default value is boost::lambda::_1 >=0
	ScopedHandle(
		const T& handle,
		const DeleterType& deleter,
		const std::string& handleDescription = "",
		const VerifierType& verifier = boost::lambda::_1 >= 0)
	:
		m_handle(handle),
		m_isValidHandle(verifier ? verifier(m_handle) : true), // null verifier means valid handle
		m_deleter(deleter)
	{
		BOOST_ASSERT(deleter);
		if (!m_isValidHandle)
			throw SHARKEXCEPTION((boost::format("%s (FAILED)") % handleDescription).str());
	}

	~ScopedHandle()
	{
		if (m_isValidHandle)
			m_deleter(m_handle);
	}

	/// The only way to access handle externally
	const T& operator*() const { return m_handle; }

private:

	const T m_handle;
	const bool m_isValidHandle;
	const DeleterType m_deleter;
};

} // namespace shark {

#endif // SHARK_CORE_SCOPED_HANDLE_H
