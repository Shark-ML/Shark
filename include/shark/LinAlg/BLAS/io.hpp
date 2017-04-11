/*!
 * \brief       Input and output of vectors and matrices
 * 
 * \author      O. Krause
 * \date        2013
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
#ifndef REMORA_IO_HPP
#define REMORA_IO_HPP

// Only forward definition required to define stream operations
#include <iosfwd>
#include <sstream>
#include "expression_types.hpp"


namespace remora{

    /** \brief output stream operator for vector expressions
     *
     * Any vector expressions can be written to a standard output stream
     * as defined in the C++ standard library. For example:
     * \code
     * vector<float> v1(3),v2(3);
     * for(size_t i=0; i<3; i++)
     * {
     *       v1(i) = i+0.2;
     *       v2(i) = i+0.3;
     * }
     * cout << v1+v2 << endl;
     * \endcode
     * will display the some of the 2 vectors like this:
     * \code
     * [3](0.5,2.5,4.5)
     * \endcode
     *
     * \param os is a standard basic output stream
     * \param vec is a vector expression
     * \return a reference to the resulting output stream
     */
    template<class E, class T, class VE>
    //  This function seems to be big. So we do not let the compiler inline it.
    std::basic_ostream<E, T> &operator << (std::basic_ostream<E, T> &os,
                                           const vector_expression<VE, cpu_tag> &vec) {
	auto&& v = eval_block(vec);
        typedef typename VE::size_type size_type;
        size_type size = v.size ();
        std::basic_ostringstream<E, T, std::allocator<E> > s;
        s.flags (os.flags ());
        s.imbue (os.getloc ());
        s.precision (os.precision ());
        s << '[' << size << "](";
        if (size > 0)
            s << v(0);
        for (size_type i = 1; i < size; ++ i)
            s << ',' << v(i);
        s << ')';
        return os << s.str ().c_str ();
    }

    /** \brief output stream operator for matrix expressions
     *
     * it outputs the content of a \f$ (M \times N) \f$ matrix to a standard output
     * stream using the following format:
     * \c [ (rows),)(columns)](((m00),(m01),...,(m0N)),...,((mM0),(mM1),...,(mMN)))
     *
     * For example:
     * \code
     * matrix<float> m(3,3) = scalar_matrix<float>(3,3,1.0) - diagonal_matrix<float>(3,3,1.0);
     * cout << m << endl;
     * \encode
     * will display
     * \code
     * [3,3]((0,1,1),(1,0,1),(1,1,0))
     * \endcode
     * This output is made for storing and retrieving matrices in a simple way but you can
     * easily recognize the following:
     * \f[ \left( \begin{array}{ccc} 1 & 1 & 1\\ 1 & 1 & 1\\ 1 & 1 & 1 \end{array} \right) - \left( \begin{array}{ccc} 1 & 0 & 0\\ 0 & 1 & 0\\ 0 & 0 & 1 \end{array} \right) = \left( \begin{array}{ccc} 0 & 1 & 1\\ 1 & 0 & 1\\ 1 & 1 & 0 \end{array} \right) \f]
     *
     * \param os is a standard basic output stream
     * \param mat is a matrix expression
     * \return a reference to the resulting output stream
     */
    template<class E, class T, class ME>
    //  This function seems to be big. So we do not let the compiler inline it.
    std::basic_ostream<E, T> &operator << (std::basic_ostream<E, T> &os,
                                           const matrix_expression<ME, cpu_tag> &mat) {
        auto&& m=eval_block(mat);
	typedef typename ME::size_type size_type;
        size_type size1 = m.size1 ();
        size_type size2 = m.size2 ();
        std::basic_ostringstream<E, T, std::allocator<E> > s;
        s.flags (os.flags ());
        s.imbue (os.getloc ());
        s.precision (os.precision ());
        s << '[' << size1 << ',' << size2 << "](";
        if (size1 > 0) {
            s << '(' ;
            if (size2 > 0)
                s << m(0, 0);
            for (size_type j = 1; j < size2; ++ j)
                s << ',' << m(0, j);
            s << ')';
        }
        for (size_type i = 1; i < size1; ++ i) {
            s << ",(" ;
            if (size2 > 0)
                s << m(i, 0);
            for (size_type j = 1; j < size2; ++ j)
                s << ',' << m(i, j);
            s << ')';
        }
        s << ')';
        return os << s.str().c_str ();
    }
}

#endif
