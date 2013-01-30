//===========================================================================
/*!
 * \file SignalTrap.h
 *
 * \brief Class that models a singleton.
 *
 *  The function is described in
 *
 * H. Li and Q. Zhang. 
 * Multiobjective Optimization Problems with Complicated Pareto Sets, MOEA/D and NSGA-II, 
 * IEEE Trans on Evolutionary Computation, 2(12):284-302, April 2009. 
 *
 * <BR><HR>
 * This file is part of Shark. This library is free software;
 * you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software
 * Foundation; either version 3, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 */
//===========================================================================
#ifndef SHARK_CORE_SINGLETON_H
#define SHARK_CORE_SINGLETON_H

#include <boost/scoped_ptr.hpp>
#include <boost/thread.hpp>

namespace shark {
  /**
   * \brief Models a singleton, i.e., enforces a single instance of a class.
   * 
   *
   */
	template<typename T>
	    class Singleton : /** \cond */ private boost::noncopyable /** \endcond */ {
	public:
		static T & instance() {
			boost::call_once(init, flag);
			return *t;
		}

		static void init() { // never throws 
			t.reset(new T());
		}

	protected:
		~Singleton() {}
		Singleton() {}

	private:
		static boost::scoped_ptr<T> t;
		static boost::once_flag flag;
	};
}

template<typename T> boost::scoped_ptr<T> shark::Singleton<T>::t(0);
template<typename T> boost::once_flag shark::Singleton<T>::flag = BOOST_ONCE_INIT; 

#endif 
