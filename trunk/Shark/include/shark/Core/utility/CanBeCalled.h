/*!
 * 
 *
 * \brief       Template class checking whether for a functor F and Argument U, F(U) can be called.
 * 
 * \par Implementation is based on
 * http://www.boost.org/doc/libs/1_52_0/doc/html/proto/appendices.html
 * 
 * 
 *
 * \author      O. Krause
 * \date        2013
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

#include <boost/mpl/bool.hpp>

 namespace shark{

	///\brief detects whether Functor(Argument) can be called.
	///
	///Suppose you have a general Functor. It might take several arguments,
	///some more suited for your computation than others. In this case one may
	///want to check, which input argument may work.
	///The exact implementation and inner workings are outlined in
	///http://www.boost.org/doc/libs/1_52_0/doc/html/proto/appendices.html
	///the trick is to generate a functor F2 from our supplied functor, such that
	/// F2(arg) is always a valid expression, but at the same time F2(arg) and F(arg)
	/// have clearly distinguishable return types. There are a few tricks needed for the
	/// case that F(arg) has return type void.
	template<class Functor, class Argument>
	struct CanBeCalled{
	private:
		struct dont_care
		{
			dont_care(...);
		};
		struct private_type
		{
		    private_type const &operator,(int) const;
		};

		typedef char yes_type;      // sizeof(yes_type) == 1
		typedef char (&no_type)[2]; // sizeof(no_type)  == 2

		template<typename T>
		static no_type is_private_type(T const &);

		static yes_type is_private_type(private_type const &);
		
		template<typename Fun>
		struct funwrap : Fun
		{
		    funwrap();
		    typedef private_type const &(*pointer_to_function)(dont_care);
		    operator pointer_to_function() const;
		};
		
		static funwrap<Functor> & m_fun;
		static Argument & m_arg;
	public:
		static bool const value = (
			sizeof(no_type) == sizeof(is_private_type( (m_fun(m_arg), 0) ))
		);

		typedef boost::mpl::bool_<value> type;
	};
}