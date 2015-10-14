#ifndef SHARK_LINALG_BLAS_IO_HPP
#define SHARK_LINALG_BLAS_IO_HPP

// Only forward definition required to define stream operations
#include <iosfwd>
#include <sstream>
#include "expression_types.hpp"


namespace shark{ namespace blas{

    /** \brief output stream operator for vector expressions
     *
     * Any vector expressions can be written to a standard output stream
     * as defined in the C++ standard library. For exaboost::mple:
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
     * \param v is a vector expression
     * \return a reference to the resulting output stream
     */
    template<class E, class T, class VE>
    //  This function seems to be big. So we do not let the compiler inline it.
    std::basic_ostream<E, T> &operator << (std::basic_ostream<E, T> &os,
                                           const vector_expression<VE> &v) {
        typedef typename VE::size_type size_type;
        size_type size = v ().size ();
        std::basic_ostringstream<E, T, std::allocator<E> > s;
        s.flags (os.flags ());
        s.imbue (os.getloc ());
        s.precision (os.precision ());
        s << '[' << size << "](";
        if (size > 0)
            s << v () (0);
        for (size_type i = 1; i < size; ++ i)
            s << ',' << v () (i);
        s << ')';
        return os << s.str ().c_str ();
    }

    /** \brief output stream operator for matrix expressions
     *
     * it outputs the content of a \f$ (M \times N) \f$ matrix to a standard output
     * stream using the following format:
     * \c [ (rows),)(columns)](((m00),(m01),...,(m0N)),...,((mM0),(mM1),...,(mMN)))
     *
     * For exaboost::mple:
     * \code
     * matrix<float> m(3,3) = scalar_matrix<float>(3,3,1.0) - diagonal_matrix<float>(3,3,1.0);
     * cout << m << endl;
     * \encode
     * will display
     * \code
     * [3,3]((0,1,1),(1,0,1),(1,1,0))
     * \endcode
     * This output is made for storing and retrieving matrices in a siboost::mple way but you can
     * easily recognize the following:
     * \f[ \left( \begin{array}{ccc} 1 & 1 & 1\\ 1 & 1 & 1\\ 1 & 1 & 1 \end{array} \right) - \left( \begin{array}{ccc} 1 & 0 & 0\\ 0 & 1 & 0\\ 0 & 0 & 1 \end{array} \right) = \left( \begin{array}{ccc} 0 & 1 & 1\\ 1 & 0 & 1\\ 1 & 1 & 0 \end{array} \right) \f]
     *
     * \param os is a standard basic output stream
     * \param m is a matrix expression
     * \return a reference to the resulting output stream
     */
    template<class E, class T, class ME>
    //  This function seems to be big. So we do not let the compiler inline it.
    std::basic_ostream<E, T> &operator << (std::basic_ostream<E, T> &os,
                                           const matrix_expression<ME> &m) {
        typedef typename ME::size_type size_type;
        size_type size1 = m ().size1 ();
        size_type size2 = m ().size2 ();
        std::basic_ostringstream<E, T, std::allocator<E> > s;
        s.flags (os.flags ());
        s.imbue (os.getloc ());
        s.precision (os.precision ());
        s << '[' << size1 << ',' << size2 << "](";
        if (size1 > 0) {
            s << '(' ;
            if (size2 > 0)
                s << m () (0, 0);
            for (size_type j = 1; j < size2; ++ j)
                s << ',' << m () (0, j);
            s << ')';
        }
        for (size_type i = 1; i < size1; ++ i) {
            s << ",(" ;
            if (size2 > 0)
                s << m () (i, 0);
            for (size_type j = 1; j < size2; ++ j)
                s << ',' << m () (i, j);
            s << ')';
        }
        s << ')';
        return os << s.str ().c_str ();
    }
}}

#endif
