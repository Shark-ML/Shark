/*!
 * 
 *
 * \brief      Defines SHARK_COMPILE_DLL
 * 
 * Windows requires the explicit export of symbols inside a DLL otherwise DLL symbols are not available.
 * SHARK_COMPILE_DLL is a macro that expands to nothing if SHARK_USE_DYNLIB is not defined or the compiler is not MSVC.
 * This is the case on linux systems or if no DLL is build. Otherwise, it expands to __declspec(dllexport) or __declspec(dllimport)
 * based on whether SHARK_COMPILE_DLL is defined.
 *
 * \author      Oswin Krause
 * \date        2015
 *
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
#ifndef SHARK_CORE_DLLSUPPORT_H
#define SHARK_CORE_DLLSUPPORT_H

#include <boost/config.hpp>

#if !( defined SHARK_USE_DYNLIB && defined BOOST_MSVC) 
#define SHARK_EXPORT_SYMBOL
#elif !(defined SHARK_COMPILE_DLL)
#define SHARK_EXPORT_SYMBOL __declspec(dllimport)
#else
#define SHARK_EXPORT_SYMBOL __declspec(dllexport)
#endif

#endif
