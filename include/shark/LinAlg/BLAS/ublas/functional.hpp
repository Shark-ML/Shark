//
//  Copyright (c) 2000-2009
//  Joerg Walter, Mathias Koch, Gunter Winkler
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  GeNeSys mbH & Co. KG in producing this work.
//

#ifndef _BOOST_UBLAS_FUNCTIONAL_
#define _BOOST_UBLAS_FUNCTIONAL_

#include <functional>
#include <boost/math/constants/constants.hpp>

#include <shark/LinAlg/BLAS/ublas/traits.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/definitions.hpp>



namespace shark{ namespace blas{

namespace detail{
	template<class T>
	T maxExpInput(){
		return boost::math::constants::ln_two<T>()*std::numeric_limits<T>::max_exponent;
	}
	/// Minimum value for exp(x) allowed so that it is not 0.
	template<class T>
	T minExpInput(){
		return boost::math::constants::ln_two<T>()*std::numeric_limits<T>::min_exponent;
	}
}

// Scalar functors
template<class T>
struct scalar_identity{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const{
		return x;
	}
};

template<class T>
struct scalar_negate{
	typedef T argument_type;
	typedef T result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const{
		return -x;
	}
};

template<class T>
struct scalar_divide{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;
	
	scalar_divide(T divisor):m_divisor(divisor){}
	result_type operator()(argument_type x)const{
		return x/m_divisor;
	}
private:
	T m_divisor;
};

template<class T>
struct scalar_multiply1{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;
	
	scalar_multiply1(T factor):m_factor(factor){}
	result_type operator()(argument_type x)const{
		return m_factor * x;
	}
private:
	T m_factor;
};

template<class T>
struct scalar_multiply2{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;
	
	scalar_multiply2(T factor):m_factor(factor){}
	result_type operator()(argument_type x)const{
		return x * m_factor;
	}
private:
	T m_factor;
};

template<class T, class U>
struct scalar_pow{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;
	
	scalar_pow(U exponent):m_exponent(exponent){}
	result_type operator()(argument_type x)const{
		using std::pow;
		return pow(x,m_exponent);
	}
private:
	T m_exponent;
};

template<class T>
struct scalar_conj{
	typedef T argument_type;
	typedef T result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const{
		return type_traits<T>::conj(x);
	}
};

template<class T>
struct scalar_real{
	typedef T argument_type;
	typedef typename type_traits<T>::real_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const{
		return type_traits<T>::real(x);
	}
};

template<class T>
struct scalar_abs{
	typedef T argument_type;
	typedef typename type_traits<T>::real_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const{
		using std::abs;
		return abs(x);
	}
};

template<class T>
struct scalar_imag{
	typedef T argument_type;
	typedef typename type_traits<T>::real_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const{
		return type_traits<T>::imag(x);
	}
};

template<class T>
struct scalar_exp{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = false;

	result_type operator()(argument_type x)const{
		using std::exp;
		return exp(x);
	}
};

template<class T>
struct scalar_soft_plus{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = false;

	result_type operator()(argument_type x)const{
		using std::exp;
		using std::log;
		if(x < detail::minExpInput<argument_type>()) {
			return x;
		}
		if(x > detail::maxExpInput<argument_type>()) {
			return argument_type();
		}
		return log(argument_type(1.0)+exp(x));
	}
};

template<class T>
struct scalar_sigmoid{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = false;

	result_type operator()(argument_type x)const{
		using std::exp;
		if(x < detail::minExpInput<argument_type>()) {
			return 0;
		}
		if(x > detail::maxExpInput<argument_type>()) {
			return 1;
		}
		return 1.0/(1.0+exp(-x));
	}
};

template<class T>
struct scalar_log{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = false;

	result_type operator()(argument_type x)const{
		using std::log;
		return log(x);
	}
};

template<class T>
struct scalar_tanh{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const{
		using std::tanh;
		return tanh(x);
	}
};

template<class T>
struct scalar_sqr{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const{
		return x*x;
	}
};

template<class T>
struct scalar_sqrt{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const{
		using std::sqrt;
		return sqrt(x);
	}
};

template<class T1,class T2>
struct scalar_binary_plus{
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	typedef typename promote_traits<T1, T2>::promote_type result_type;
	
	result_type operator()(argument1_type x, argument2_type y)const{
		return x+y;
	}
};
template<class T1,class T2>
struct scalar_binary_minus{
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	typedef typename promote_traits<T1, T2>::promote_type result_type;
	
	result_type operator()(argument1_type x, argument2_type y)const{
		return x-y;
	}
};

template<class T1,class T2>
struct scalar_binary_multiply{
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	typedef typename promote_traits<T1, T2>::promote_type result_type;
	
	result_type operator()(argument1_type x, argument2_type y)const{
		return x*y;
	}
};

template<class T1,class T2>
struct scalar_binary_divide{
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	typedef typename promote_traits<T1, T2>::promote_type result_type;
	
	result_type operator()(argument1_type x, argument2_type y)const{
		return x/y;
	}
};
template<class T1,class T2>
struct scalar_binary_safe_divide{
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	typedef typename promote_traits<T1, T2>::promote_type result_type;
	
	scalar_binary_safe_divide( result_type defaultValue):m_defaultValue(defaultValue){}
	result_type operator()(argument1_type x, argument2_type y)const{
		return y == T2()? m_defaultValue : x/y;
	}
private:
	result_type m_defaultValue;
};


    template<class T1, class T2>
    struct scalar_binary_assign_functor {
        // ISSUE Remove reference to avoid reference to reference problems
        typedef typename type_traits<typename boost::remove_reference<T1>::type>::reference argument1_type;
        typedef typename type_traits<T2>::const_reference argument2_type;
    };

    struct assign_tag {};
    struct computed_assign_tag {};

    template<class T1, class T2>
    struct scalar_assign:
        public scalar_binary_assign_functor<T1, T2> {
        typedef typename scalar_binary_assign_functor<T1, T2>::argument1_type argument1_type;
        typedef typename scalar_binary_assign_functor<T1, T2>::argument2_type argument2_type;
#if BOOST_WORKAROUND( __IBMCPP__, <=600 )
        static const bool computed ;
#else
        static const bool computed = false ;
#endif

        static BOOST_UBLAS_INLINE
        void apply (argument1_type t1, argument2_type t2) {
            t1 = t2;
        }

        template<class U1, class U2>
        struct rebind {
            typedef scalar_assign<U1, U2> other;
        };
    };

#if BOOST_WORKAROUND( __IBMCPP__, <=600 )
    template<class T1, class T2>
    const bool scalar_assign<T1,T2>::computed = false;
#endif

    template<class T1, class T2>
    struct scalar_plus_assign:
        public scalar_binary_assign_functor<T1, T2> {
        typedef typename scalar_binary_assign_functor<T1, T2>::argument1_type argument1_type;
        typedef typename scalar_binary_assign_functor<T1, T2>::argument2_type argument2_type;
#if BOOST_WORKAROUND( __IBMCPP__, <=600 )
        static const bool computed ;
#else
        static const bool computed = true ;
#endif

        static BOOST_UBLAS_INLINE
        void apply (argument1_type t1, argument2_type t2) {
            t1 += t2;
        }

        template<class U1, class U2>
        struct rebind {
            typedef scalar_plus_assign<U1, U2> other;
        };
    };

#if BOOST_WORKAROUND( __IBMCPP__, <=600 )
    template<class T1, class T2>
    const bool scalar_plus_assign<T1,T2>::computed = true;
#endif

    template<class T1, class T2>
    struct scalar_minus_assign:
        public scalar_binary_assign_functor<T1, T2> {
        typedef typename scalar_binary_assign_functor<T1, T2>::argument1_type argument1_type;
        typedef typename scalar_binary_assign_functor<T1, T2>::argument2_type argument2_type;
#if BOOST_WORKAROUND( __IBMCPP__, <=600 )
        static const bool computed ;
#else
        static const bool computed = true ;
#endif

        static BOOST_UBLAS_INLINE
        void apply (argument1_type t1, argument2_type t2) {
            t1 -= t2;
        }

        template<class U1, class U2>
        struct rebind {
            typedef scalar_minus_assign<U1, U2> other;
        };
    };

#if BOOST_WORKAROUND( __IBMCPP__, <=600 )
    template<class T1, class T2>
    const bool scalar_minus_assign<T1,T2>::computed = true;
#endif

    template<class T1, class T2>
    struct scalar_multiplies_assign:
        public scalar_binary_assign_functor<T1, T2> {
        typedef typename scalar_binary_assign_functor<T1, T2>::argument1_type argument1_type;
        typedef typename scalar_binary_assign_functor<T1, T2>::argument2_type argument2_type;
        static const bool computed = true;

        static BOOST_UBLAS_INLINE
        void apply (argument1_type t1, argument2_type t2) {
            t1 *= t2;
        }

        template<class U1, class U2>
        struct rebind {
            typedef scalar_multiplies_assign<U1, U2> other;
        };
    };
    template<class T1, class T2>
    struct scalar_divides_assign:
        public scalar_binary_assign_functor<T1, T2> {
        typedef typename scalar_binary_assign_functor<T1, T2>::argument1_type argument1_type;
        typedef typename scalar_binary_assign_functor<T1, T2>::argument2_type argument2_type;
        static const bool computed ;

        static BOOST_UBLAS_INLINE
        void apply (argument1_type t1, argument2_type t2) {
            t1 /= t2;
        }

        template<class U1, class U2>
        struct rebind {
            typedef scalar_divides_assign<U1, U2> other;
        };
    };
    template<class T1, class T2>
    const bool scalar_divides_assign<T1,T2>::computed = true;

    template<class T1, class T2>
    struct scalar_binary_swap_functor {
        typedef typename type_traits<typename boost::remove_reference<T1>::type>::reference argument1_type;
        typedef typename type_traits<typename boost::remove_reference<T2>::type>::reference argument2_type;
    };

    template<class T1, class T2>
    struct scalar_swap:
        public scalar_binary_swap_functor<T1, T2> {
        typedef typename scalar_binary_swap_functor<T1, T2>::argument1_type argument1_type;
        typedef typename scalar_binary_swap_functor<T1, T2>::argument2_type argument2_type;

        static BOOST_UBLAS_INLINE
        void apply (argument1_type t1, argument2_type t2) {
            std::swap (t1, t2);
        }

        template<class U1, class U2>
        struct rebind {
            typedef scalar_swap<U1, U2> other;
        };
    };

    // Vector functors

    // Unary returning scalar
    template<class V>
    struct vector_scalar_unary_functor {
        typedef typename V::value_type value_type;
        typedef typename V::value_type result_type;
    };

    template<class V>
    struct vector_sum: 
        public vector_scalar_unary_functor<V> {
        typedef typename vector_scalar_unary_functor<V>::value_type value_type;
        typedef typename vector_scalar_unary_functor<V>::result_type result_type;

        template<class E>
        static BOOST_UBLAS_INLINE
        result_type apply (const vector_expression<E> &e) { 
            result_type t = result_type (0);
            typedef typename E::size_type vector_size_type;
            vector_size_type size (e ().size ());
            for (vector_size_type i = 0; i < size; ++ i)
                t += e () (i);
            return t;
        }
        // Dense case
        template<class D, class I>
        static BOOST_UBLAS_INLINE
        result_type apply (D size, I it) { 
            result_type t = result_type (0);
            while (-- size >= 0)
                t += *it, ++ it;
            return t; 
        }
        // Sparse case
        template<class I>
        static BOOST_UBLAS_INLINE
        result_type apply (I it, const I &it_end) {
            result_type t = result_type (0);
            while (it != it_end) 
                t += *it, ++ it;
            return t; 
        }
    };

    // Unary returning real scalar 
    template<class V>
    struct vector_scalar_real_unary_functor {
        typedef typename V::value_type value_type;
        typedef typename type_traits<value_type>::real_type real_type;
        typedef real_type result_type;
    };

    template<class V>
    struct vector_norm_1:
        public vector_scalar_real_unary_functor<V> {
        typedef typename vector_scalar_real_unary_functor<V>::value_type value_type;
        typedef typename vector_scalar_real_unary_functor<V>::real_type real_type;
        typedef typename vector_scalar_real_unary_functor<V>::result_type result_type;

        template<class E>
        static BOOST_UBLAS_INLINE
        result_type apply (const vector_expression<E> &e) {
            real_type t = real_type ();
            typedef typename E::size_type vector_size_type;
            vector_size_type size (e ().size ());
            for (vector_size_type i = 0; i < size; ++ i) {
                real_type u (type_traits<value_type>::type_abs (e () (i)));
                t += u;
            }
            return t;
        }
        // Dense case
        template<class D, class I>
        static BOOST_UBLAS_INLINE
        result_type apply (D size, I it) {
            real_type t = real_type ();
            while (-- size >= 0) {
                real_type u (type_traits<value_type>::norm_1 (*it));
                t += u;
                ++ it;
            }
            return t;
        }
        // Sparse case
        template<class I>
        static BOOST_UBLAS_INLINE
        result_type apply (I it, const I &it_end) {
            real_type t = real_type ();
            while (it != it_end) {
                real_type u (type_traits<value_type>::norm_1 (*it));
                t += u;
                ++ it;
            }
            return t;
        }
    };
    template<class V>
    struct vector_norm_2:
        public vector_scalar_real_unary_functor<V> {
        typedef typename vector_scalar_real_unary_functor<V>::value_type value_type;
        typedef typename vector_scalar_real_unary_functor<V>::real_type real_type;
        typedef typename vector_scalar_real_unary_functor<V>::result_type result_type;

        template<class E>
        static BOOST_UBLAS_INLINE
        result_type apply (const vector_expression<E> &e) {
            real_type t = real_type ();
            typedef typename E::size_type vector_size_type;
            vector_size_type size (e ().size ());
            for (vector_size_type i = 0; i < size; ++ i) {
                real_type u (type_traits<value_type>::norm_2 (e () (i)));
                t +=  u * u;
            }
            return type_traits<real_type>::type_sqrt (t);
        }
        // Dense case
        template<class D, class I>
        static BOOST_UBLAS_INLINE
        result_type apply (D size, I it) {
            real_type t = real_type ();
            while (-- size >= 0) {
                real_type u (type_traits<value_type>::norm_2 (*it));
                t +=  u * u;
                ++ it;
            }
            return type_traits<real_type>::type_sqrt (t);
        }
        // Sparse case
        template<class I>
        static BOOST_UBLAS_INLINE
        result_type apply (I it, const I &it_end) {
            real_type t = real_type ();
            while (it != it_end) {
                real_type u (type_traits<value_type>::norm_2 (*it));
                t +=  u * u;
                ++ it;
            }
            return type_traits<real_type>::type_sqrt (t);
        }
    };
    template<class V>
    struct vector_norm_inf:
        public vector_scalar_real_unary_functor<V> {
        typedef typename vector_scalar_real_unary_functor<V>::value_type value_type;
        typedef typename vector_scalar_real_unary_functor<V>::real_type real_type;
        typedef typename vector_scalar_real_unary_functor<V>::result_type result_type;

        template<class E>
        static BOOST_UBLAS_INLINE
        result_type apply (const vector_expression<E> &e) {
            real_type t = real_type ();
            typedef typename E::size_type vector_size_type;
            vector_size_type size (e ().size ());
            for (vector_size_type i = 0; i < size; ++ i) {
                real_type u (type_traits<value_type>::norm_inf (e () (i)));
                if (u > t)
                    t = u;
            }
            return t;
        }
        // Dense case
        template<class D, class I>
        static BOOST_UBLAS_INLINE
        result_type apply (D size, I it) {
            real_type t = real_type ();
            while (-- size >= 0) {
                real_type u (type_traits<value_type>::norm_inf (*it));
                if (u > t)
                    t = u;
                ++ it;
            }
            return t;
        }
        // Sparse case
        template<class I>
        static BOOST_UBLAS_INLINE
        result_type apply (I it, const I &it_end) { 
            real_type t = real_type ();
            while (it != it_end) {
                real_type u (type_traits<value_type>::norm_inf (*it));
                if (u > t) 
                    t = u;
                ++ it;
            }
            return t; 
        }
    };

    // Unary returning index
    template<class V>
    struct vector_scalar_index_unary_functor {
        typedef typename V::value_type value_type;
        typedef typename type_traits<value_type>::real_type real_type;
        typedef typename V::size_type result_type;
    };

    template<class V>
    struct vector_index_norm_inf:
        public vector_scalar_index_unary_functor<V> {
        typedef typename vector_scalar_index_unary_functor<V>::value_type value_type;
        typedef typename vector_scalar_index_unary_functor<V>::real_type real_type;
        typedef typename vector_scalar_index_unary_functor<V>::result_type result_type;

        template<class E>
        static BOOST_UBLAS_INLINE
        result_type apply (const vector_expression<E> &e) {
            // ISSUE For CBLAS compatibility return 0 index in empty case
            result_type i_norm_inf (0);
            real_type t = real_type ();
            typedef typename E::size_type vector_size_type;
            vector_size_type size (e ().size ());
            for (vector_size_type i = 0; i < size; ++ i) {
                real_type u (type_traits<value_type>::norm_inf (e () (i)));
                if (u > t) {
                    i_norm_inf = i;
                    t = u;
                }
            }
            return i_norm_inf;
        }
        // Dense case
        template<class D, class I>
        static BOOST_UBLAS_INLINE
        result_type apply (D size, I it) {
            // ISSUE For CBLAS compatibility return 0 index in empty case
            result_type i_norm_inf (0);
            real_type t = real_type ();
            while (-- size >= 0) {
                real_type u (type_traits<value_type>::norm_inf (*it));
                if (u > t) {
                    i_norm_inf = it.index ();
                    t = u;
                }
                ++ it;
            }
            return i_norm_inf;
        }
        // Sparse case
        template<class I>
        static BOOST_UBLAS_INLINE
        result_type apply (I it, const I &it_end) {
            // ISSUE For CBLAS compatibility return 0 index in empty case
            result_type i_norm_inf (0);
            real_type t = real_type ();
            while (it != it_end) {
                real_type u (type_traits<value_type>::norm_inf (*it));
                if (u > t) {
                    i_norm_inf = it.index ();
                    t = u;
                }
                ++ it;
            }
            return i_norm_inf;
        }
    };

    // Binary returning scalar
    template<class V1, class V2, class TV>
    struct vector_scalar_binary_functor {
        typedef TV value_type;
        typedef TV result_type;
    };

    template<class V1, class V2, class TV>
    struct vector_inner_prod:
        public vector_scalar_binary_functor<V1, V2, TV> {
        typedef typename vector_scalar_binary_functor<V1, V2, TV>::value_type value_type;
        typedef typename vector_scalar_binary_functor<V1, V2, TV>::result_type result_type;

        template<class E1, class E2>
        static BOOST_UBLAS_INLINE
        result_type apply (const vector_expression<E1> &e1,
                           const vector_expression<E2> &e2) {
            typedef typename E1::size_type vector_size_type;
            vector_size_type size (BOOST_UBLAS_SAME (e1 ().size (), e2 ().size ()));
            result_type t = result_type (0);
            for (vector_size_type i = 0; i < size; ++ i)
                t += e1 () (i) * e2 () (i);
            return t;
        }
        // Dense case
        template<class D, class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (D size, I1 it1, I2 it2) {
            result_type t = result_type (0);
            while (-- size >= 0)
                t += *it1 * *it2, ++ it1, ++ it2;
            return t;
        }
        // Packed case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end) {
            result_type t = result_type (0);
            typedef typename I1::difference_type vector_difference_type;
            vector_difference_type it1_size (it1_end - it1);
            vector_difference_type it2_size (it2_end - it2);
            vector_difference_type diff (0);
            if (it1_size > 0 && it2_size > 0)
                diff = it2.index () - it1.index ();
            if (diff != 0) {
                vector_difference_type size = (std::min) (diff, it1_size);
                if (size > 0) {
                    it1 += size;
                    it1_size -= size;
                    diff -= size;
                }
                size = (std::min) (- diff, it2_size);
                if (size > 0) {
                    it2 += size;
                    it2_size -= size;
                    diff += size;
                }
            }
            vector_difference_type size ((std::min) (it1_size, it2_size));
            while (-- size >= 0)
                t += *it1 * *it2, ++ it1, ++ it2;
            return t;
        }
        // Sparse case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end, sparse_bidirectional_iterator_tag) {
            result_type t = result_type (0);
            if (it1 != it1_end && it2 != it2_end) {
                while (true) {
                    if (it1.index () == it2.index ()) {
                        t += *it1 * *it2, ++ it1, ++ it2;
                        if (it1 == it1_end || it2 == it2_end)
                            break;
                    } else if (it1.index () < it2.index ()) {
                        increment (it1, it1_end, it2.index () - it1.index ());
                        if (it1 == it1_end)
                            break;
                    } else if (it1.index () > it2.index ()) {
                        increment (it2, it2_end, it1.index () - it2.index ());
                        if (it2 == it2_end)
                            break;
                    }
                }
            }
            return t;
        }
    };

    // Matrix functors

    // Binary returning vector
    template<class M1, class M2, class TV>
    struct matrix_vector_binary_functor {
        typedef typename M1::size_type size_type;
        typedef typename M1::difference_type difference_type;
        typedef TV value_type;
        typedef TV result_type;
    };

    template<class M1, class M2, class TV>
    struct matrix_vector_prod1:
        public matrix_vector_binary_functor<M1, M2, TV> {
        typedef typename matrix_vector_binary_functor<M1, M2, TV>::size_type size_type;
        typedef typename matrix_vector_binary_functor<M1, M2, TV>::difference_type difference_type;
        typedef typename matrix_vector_binary_functor<M1, M2, TV>::value_type value_type;
        typedef typename matrix_vector_binary_functor<M1, M2, TV>::result_type result_type;

        template<class E1, class E2>
        static BOOST_UBLAS_INLINE
        result_type apply (const matrix_expression<E1> &e1,
                           const vector_expression<E2> &e2,
                           size_type i) {
            size_type size = BOOST_UBLAS_SAME (e1 ().size2 (), e2 ().size ());
            result_type t = result_type (0);
            for (size_type j = 0; j < size; ++ j)
                t += e1 () (i, j) * e2 () (j);
            return t;
        }
        // Dense case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (difference_type size, I1 it1, I2 it2) {
            result_type t = result_type (0);
            while (-- size >= 0)
                t += *it1 * *it2, ++ it1, ++ it2;
            return t;
        }
        // Packed case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end) {
            result_type t = result_type (0);
            difference_type it1_size (it1_end - it1);
            difference_type it2_size (it2_end - it2);
            difference_type diff (0);
            if (it1_size > 0 && it2_size > 0)
                diff = it2.index () - it1.index2 ();
            if (diff != 0) {
                difference_type size = (std::min) (diff, it1_size);
                if (size > 0) {
                    it1 += size;
                    it1_size -= size;
                    diff -= size;
                }
                size = (std::min) (- diff, it2_size);
                if (size > 0) {
                    it2 += size;
                    it2_size -= size;
                    diff += size;
                }
            }
            difference_type size ((std::min) (it1_size, it2_size));
            while (-- size >= 0)
                t += *it1 * *it2, ++ it1, ++ it2;
            return t;
        }
        // Sparse case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end,
                           sparse_bidirectional_iterator_tag, sparse_bidirectional_iterator_tag) {
            result_type t = result_type (0);
            if (it1 != it1_end && it2 != it2_end) {
                size_type it1_index = it1.index2 (), it2_index = it2.index ();
                while (true) {
                    difference_type compare = it1_index - it2_index;
                    if (compare == 0) {
                        t += *it1 * *it2, ++ it1, ++ it2;
                        if (it1 != it1_end && it2 != it2_end) {
                            it1_index = it1.index2 ();
                            it2_index = it2.index ();
                        } else
                            break;
                    } else if (compare < 0) {
                        increment (it1, it1_end, - compare);
                        if (it1 != it1_end)
                            it1_index = it1.index2 ();
                        else
                            break;
                    } else if (compare > 0) {
                        increment (it2, it2_end, compare);
                        if (it2 != it2_end)
                            it2_index = it2.index ();
                        else
                            break;
                    }
                }
            }
            return t;
        }
        // Sparse packed case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &it1_end, I2 it2, const I2 &/* it2_end */,
                           sparse_bidirectional_iterator_tag, packed_random_access_iterator_tag) {
            result_type t = result_type (0);
            while (it1 != it1_end) {
                t += *it1 * it2 () (it1.index2 ());
                ++ it1;
            }
            return t;
        }
        // Packed sparse case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &/* it1_end */, I2 it2, const I2 &it2_end,
                           packed_random_access_iterator_tag, sparse_bidirectional_iterator_tag) {
            result_type t = result_type (0);
            while (it2 != it2_end) {
                t += it1 () (it1.index1 (), it2.index ()) * *it2;
                ++ it2;
            }
            return t;
        }
        // Another dispatcher
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end,
                           sparse_bidirectional_iterator_tag) {
            typedef typename I1::iterator_category iterator1_category;
            typedef typename I2::iterator_category iterator2_category;
            return apply (it1, it1_end, it2, it2_end, iterator1_category (), iterator2_category ());
        }
    };

    template<class M1, class M2, class TV>
    struct matrix_vector_prod2:
        public matrix_vector_binary_functor<M1, M2, TV> {
        typedef typename matrix_vector_binary_functor<M1, M2, TV>::size_type size_type;
        typedef typename matrix_vector_binary_functor<M1, M2, TV>::difference_type difference_type;
        typedef typename matrix_vector_binary_functor<M1, M2, TV>::value_type value_type;
        typedef typename matrix_vector_binary_functor<M1, M2, TV>::result_type result_type;

        template<class E1, class E2>
        static BOOST_UBLAS_INLINE
        result_type apply (const vector_expression<E1> &e1,
                           const matrix_expression<E2> &e2,
                           size_type i) {
            size_type size = BOOST_UBLAS_SAME (e1 ().size (), e2 ().size1 ());
            result_type t = result_type (0);
            for (size_type j = 0; j < size; ++ j)
                t += e1 () (j) * e2 () (j, i);
            return t;
        }
        // Dense case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (difference_type size, I1 it1, I2 it2) {
            result_type t = result_type (0);
            while (-- size >= 0)
                t += *it1 * *it2, ++ it1, ++ it2;
            return t;
        }
        // Packed case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end) {
            result_type t = result_type (0);
            difference_type it1_size (it1_end - it1);
            difference_type it2_size (it2_end - it2);
            difference_type diff (0);
            if (it1_size > 0 && it2_size > 0)
                diff = it2.index1 () - it1.index ();
            if (diff != 0) {
                difference_type size = (std::min) (diff, it1_size);
                if (size > 0) {
                    it1 += size;
                    it1_size -= size;
                    diff -= size;
                }
                size = (std::min) (- diff, it2_size);
                if (size > 0) {
                    it2 += size;
                    it2_size -= size;
                    diff += size;
                }
            }
            difference_type size ((std::min) (it1_size, it2_size));
            while (-- size >= 0)
                t += *it1 * *it2, ++ it1, ++ it2;
            return t;
        }
        // Sparse case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end,
                           sparse_bidirectional_iterator_tag, sparse_bidirectional_iterator_tag) {
            result_type t = result_type (0);
            if (it1 != it1_end && it2 != it2_end) {
                size_type it1_index = it1.index (), it2_index = it2.index1 ();
                while (true) {
                    difference_type compare = it1_index - it2_index;
                    if (compare == 0) {
                        t += *it1 * *it2, ++ it1, ++ it2;
                        if (it1 != it1_end && it2 != it2_end) {
                            it1_index = it1.index ();
                            it2_index = it2.index1 ();
                        } else
                            break;
                    } else if (compare < 0) {
                        increment (it1, it1_end, - compare);
                        if (it1 != it1_end)
                            it1_index = it1.index ();
                        else
                            break;
                    } else if (compare > 0) {
                        increment (it2, it2_end, compare);
                        if (it2 != it2_end)
                            it2_index = it2.index1 ();
                        else
                            break;
                    }
                }
            }
            return t;
        }
        // Packed sparse case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &/* it1_end */, I2 it2, const I2 &it2_end,
                           packed_random_access_iterator_tag, sparse_bidirectional_iterator_tag) {
            result_type t = result_type (0);
            while (it2 != it2_end) {
                t += it1 () (it2.index1 ()) * *it2;
                ++ it2;
            }
            return t;
        }
        // Sparse packed case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &it1_end, I2 it2, const I2 &/* it2_end */,
                           sparse_bidirectional_iterator_tag, packed_random_access_iterator_tag) {
            result_type t = result_type (0);
            while (it1 != it1_end) {
                t += *it1 * it2 () (it1.index (), it2.index2 ());
                ++ it1;
            }
            return t;
        }
        // Another dispatcher
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end,
                           sparse_bidirectional_iterator_tag) {
            typedef typename I1::iterator_category iterator1_category;
            typedef typename I2::iterator_category iterator2_category;
            return apply (it1, it1_end, it2, it2_end, iterator1_category (), iterator2_category ());
        }
    };

    // Binary returning matrix
    template<class M1, class M2, class TV>
    struct matrix_matrix_binary_functor {
        typedef typename M1::size_type size_type;
        typedef typename M1::difference_type difference_type;
        typedef TV value_type;
        typedef TV result_type;
    };

    template<class M1, class M2, class TV>
    struct matrix_matrix_prod:
        public matrix_matrix_binary_functor<M1, M2, TV> {
        typedef typename matrix_matrix_binary_functor<M1, M2, TV>::size_type size_type;
        typedef typename matrix_matrix_binary_functor<M1, M2, TV>::difference_type difference_type;
        typedef typename matrix_matrix_binary_functor<M1, M2, TV>::value_type value_type;
        typedef typename matrix_matrix_binary_functor<M1, M2, TV>::result_type result_type;

        template<class E1, class E2>
        static BOOST_UBLAS_INLINE
        result_type apply (const matrix_expression<E1> &e1,
                           const matrix_expression<E2> &e2,
                           size_type i, size_type j) {
            size_type size = BOOST_UBLAS_SAME (e1 ().size2 (), e2 ().size1 ());
            result_type t = result_type (0);
            for (size_type k = 0; k < size; ++ k)
                t += e1 () (i, k) * e2 () (k, j);
            return t;
        }
        // Dense case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (difference_type size, I1 it1, I2 it2) {
            result_type t = result_type (0);
            while (-- size >= 0)
                t += *it1 * *it2, ++ it1, ++ it2;
            return t;
        }
        // Packed case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end, packed_random_access_iterator_tag) {
            result_type t = result_type (0);
            difference_type it1_size (it1_end - it1);
            difference_type it2_size (it2_end - it2);
            difference_type diff (0);
            if (it1_size > 0 && it2_size > 0)
                diff = it2.index1 () - it1.index2 ();
            if (diff != 0) {
                difference_type size = (std::min) (diff, it1_size);
                if (size > 0) {
                    it1 += size;
                    it1_size -= size;
                    diff -= size;
                }
                size = (std::min) (- diff, it2_size);
                if (size > 0) {
                    it2 += size;
                    it2_size -= size;
                    diff += size;
                }
            }
            difference_type size ((std::min) (it1_size, it2_size));
            while (-- size >= 0)
                t += *it1 * *it2, ++ it1, ++ it2;
            return t;
        }
        // Sparse case
        template<class I1, class I2>
        static BOOST_UBLAS_INLINE
        result_type apply (I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end, sparse_bidirectional_iterator_tag) {
            result_type t = result_type (0);
            if (it1 != it1_end && it2 != it2_end) {
                size_type it1_index = it1.index2 (), it2_index = it2.index1 ();
                while (true) {
                    difference_type compare = difference_type (it1_index - it2_index);
                    if (compare == 0) {
                        t += *it1 * *it2, ++ it1, ++ it2;
                        if (it1 != it1_end && it2 != it2_end) {
                            it1_index = it1.index2 ();
                            it2_index = it2.index1 ();
                        } else
                            break;
                    } else if (compare < 0) {
                        increment (it1, it1_end, - compare);
                        if (it1 != it1_end)
                            it1_index = it1.index2 ();
                        else
                            break;
                    } else if (compare > 0) {
                        increment (it2, it2_end, compare);
                        if (it2 != it2_end)
                            it2_index = it2.index1 ();
                        else
                            break;
                    }
                }
            }
            return t;
        }
    };

    // Unary returning scalar norm
    template<class M>
    struct matrix_scalar_real_unary_functor {
        typedef typename M::value_type value_type;
        typedef typename type_traits<value_type>::real_type real_type;
        typedef real_type result_type;
    };

    template<class M>
    struct matrix_norm_1:
        public matrix_scalar_real_unary_functor<M> {
        typedef typename matrix_scalar_real_unary_functor<M>::value_type value_type;
        typedef typename matrix_scalar_real_unary_functor<M>::real_type real_type;
        typedef typename matrix_scalar_real_unary_functor<M>::result_type result_type;

        template<class E>
        static BOOST_UBLAS_INLINE
        result_type apply (const matrix_expression<E> &e) {
            real_type t = real_type ();
            typedef typename E::size_type matrix_size_type;
            matrix_size_type size2 (e ().size2 ());
            for (matrix_size_type j = 0; j < size2; ++ j) {
                real_type u = real_type ();
                matrix_size_type size1 (e ().size1 ());
                for (matrix_size_type i = 0; i < size1; ++ i) {
                    real_type v (type_traits<value_type>::norm_1 (e () (i, j)));
                    u += v;
                }
                if (u > t)
                    t = u;
            }
            return t; 
        }
    };

    template<class M>
    struct matrix_norm_frobenius:
        public matrix_scalar_real_unary_functor<M> {
        typedef typename matrix_scalar_real_unary_functor<M>::value_type value_type;
        typedef typename matrix_scalar_real_unary_functor<M>::real_type real_type;
        typedef typename matrix_scalar_real_unary_functor<M>::result_type result_type;

        template<class E>
        static BOOST_UBLAS_INLINE
        result_type apply (const matrix_expression<E> &e) { 
            real_type t = real_type ();
            typedef typename E::size_type matrix_size_type;
            matrix_size_type size1 (e ().size1 ());
            for (matrix_size_type i = 0; i < size1; ++ i) {
                matrix_size_type size2 (e ().size2 ());
                for (matrix_size_type j = 0; j < size2; ++ j) {
                    real_type u (type_traits<value_type>::norm_2 (e () (i, j)));
                    t +=  u * u;
                }
            }
            return type_traits<real_type>::type_sqrt (t); 
        }
    };

    template<class M>
    struct matrix_norm_inf: 
        public matrix_scalar_real_unary_functor<M> {
        typedef typename matrix_scalar_real_unary_functor<M>::value_type value_type;
        typedef typename matrix_scalar_real_unary_functor<M>::real_type real_type;
        typedef typename matrix_scalar_real_unary_functor<M>::result_type result_type;

        template<class E>
        static BOOST_UBLAS_INLINE
        result_type apply (const matrix_expression<E> &e) {
            real_type t = real_type ();
            typedef typename E::size_type matrix_size_type;
            matrix_size_type size1 (e ().size1 ());
            for (matrix_size_type i = 0; i < size1; ++ i) {
                real_type u = real_type ();
                matrix_size_type size2 (e ().size2 ());
                for (matrix_size_type j = 0; j < size2; ++ j) {
                    real_type v (type_traits<value_type>::norm_inf (e () (i, j)));
                    u += v;
                }
                if (u > t) 
                    t = u;  
            }
            return t; 
        }
    };

    // forward declaration
    template <class Z, class D> struct basic_column_major;

    // This functor defines storage layout and it's properties
    // matrix (i,j) -> storage [i * size_i + j]
    template <class Z, class D>
    struct basic_row_major {
        typedef Z size_type;
        typedef D difference_type;
        typedef row_major_tag orientation_category;
        typedef basic_column_major<Z,D> transposed_layout;

        static
        BOOST_UBLAS_INLINE
        size_type storage_size (size_type size_i, size_type size_j) {
            // Guard against size_type overflow
            BOOST_UBLAS_CHECK (size_j == 0 || size_i <= (std::numeric_limits<size_type>::max) () / size_j, bad_size ());
            return size_i * size_j;
        }

        // Indexing conversion to storage element
        static
        BOOST_UBLAS_INLINE
        size_type element (size_type i, size_type size_i, size_type j, size_type size_j) {
            BOOST_UBLAS_CHECK (i < size_i, bad_index ());
            BOOST_UBLAS_CHECK (j < size_j, bad_index ());
            detail::ignore_unused_variable_warning(size_i);
            // Guard against size_type overflow
            BOOST_UBLAS_CHECK (i <= ((std::numeric_limits<size_type>::max) () - j) / size_j, bad_index ());
            return i * size_j + j;
        }
        static
        BOOST_UBLAS_INLINE
        size_type address (size_type i, size_type size_i, size_type j, size_type size_j) {
            BOOST_UBLAS_CHECK (i <= size_i, bad_index ());
            BOOST_UBLAS_CHECK (j <= size_j, bad_index ());
            // Guard against size_type overflow - address may be size_j past end of storage
            BOOST_UBLAS_CHECK (size_j == 0 || i <= ((std::numeric_limits<size_type>::max) () - j) / size_j, bad_index ());
            detail::ignore_unused_variable_warning(size_i);
            return i * size_j + j;
        }

        // Storage element to index conversion
        static
        BOOST_UBLAS_INLINE
        difference_type distance_i (difference_type k, size_type /* size_i */, size_type size_j) {
            return size_j != 0 ? k / size_j : 0;
        }
        static
        BOOST_UBLAS_INLINE
        difference_type distance_j (difference_type k, size_type /* size_i */, size_type /* size_j */) {
            return k;
        }
        static
        BOOST_UBLAS_INLINE
        size_type index_i (difference_type k, size_type /* size_i */, size_type size_j) {
            return size_j != 0 ? k / size_j : 0;
        }
        static
        BOOST_UBLAS_INLINE
        size_type index_j (difference_type k, size_type /* size_i */, size_type size_j) {
            return size_j != 0 ? k % size_j : 0;
        }
        static
        BOOST_UBLAS_INLINE
        bool fast_i () {
            return false;
        }
        static
        BOOST_UBLAS_INLINE
        bool fast_j () {
            return true;
        }

        // Iterating storage elements
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void increment_i (I &it, size_type /* size_i */, size_type size_j) {
            it += size_j;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void increment_i (I &it, difference_type n, size_type /* size_i */, size_type size_j) {
            it += n * size_j;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void decrement_i (I &it, size_type /* size_i */, size_type size_j) {
            it -= size_j;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void decrement_i (I &it, difference_type n, size_type /* size_i */, size_type size_j) {
            it -= n * size_j;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void increment_j (I &it, size_type /* size_i */, size_type /* size_j */) {
            ++ it;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void increment_j (I &it, difference_type n, size_type /* size_i */, size_type /* size_j */) {
            it += n;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void decrement_j (I &it, size_type /* size_i */, size_type /* size_j */) {
            -- it;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void decrement_j (I &it, difference_type n, size_type /* size_i */, size_type /* size_j */) {
            it -= n;
        }

        // Triangular access
        static
        BOOST_UBLAS_INLINE
        size_type triangular_size (size_type size_i, size_type size_j) {
            size_type size = (std::max) (size_i, size_j);
            // Guard against size_type overflow - siboost::mplified
            BOOST_UBLAS_CHECK (size == 0 || size / 2 < (std::numeric_limits<size_type>::max) () / size /* +1/2 */, bad_size ());
            return ((size + 1) * size) / 2;
        }
        static
        BOOST_UBLAS_INLINE
        size_type lower_element (size_type i, size_type size_i, size_type j, size_type size_j) {
            BOOST_UBLAS_CHECK (i < size_i, bad_index ());
            BOOST_UBLAS_CHECK (j < size_j, bad_index ());
            BOOST_UBLAS_CHECK (i >= j, bad_index ());
            detail::ignore_unused_variable_warning(size_i);
            detail::ignore_unused_variable_warning(size_j);
            // FIXME size_type overflow
            // sigma_i (i + 1) = (i + 1) * i / 2
            // i = 0 1 2 3, sigma = 0 1 3 6
            return ((i + 1) * i) / 2 + j;
        }
        static
        BOOST_UBLAS_INLINE
        size_type upper_element (size_type i, size_type size_i, size_type j, size_type size_j) {
            BOOST_UBLAS_CHECK (i < size_i, bad_index ());
            BOOST_UBLAS_CHECK (j < size_j, bad_index ());
            BOOST_UBLAS_CHECK (i <= j, bad_index ());
            // FIXME size_type overflow
            // sigma_i (size - i) = size * i - i * (i - 1) / 2
            // i = 0 1 2 3, sigma = 0 4 7 9
            return (i * (2 * (std::max) (size_i, size_j) - i + 1)) / 2 + j - i;
        }

        // Major and minor indices
        static
        BOOST_UBLAS_INLINE
        size_type index_M (size_type index1, size_type /* index2 */) {
            return index1;
        }
        static
        BOOST_UBLAS_INLINE
        size_type index_m (size_type /* index1 */, size_type index2) {
            return index2;
        }
        static
        BOOST_UBLAS_INLINE
        size_type size_M (size_type size_i, size_type /* size_j */) {
            return size_i;
        }
        static
        BOOST_UBLAS_INLINE
        size_type size_m (size_type /* size_i */, size_type size_j) {
            return size_j;
        }
    };

    // This functor defines storage layout and it's properties
    // matrix (i,j) -> storage [i + j * size_i]
    template <class Z, class D>
    struct basic_column_major {
        typedef Z size_type;
        typedef D difference_type;
        typedef column_major_tag orientation_category;
        typedef basic_row_major<Z,D> transposed_layout;

        static
        BOOST_UBLAS_INLINE
        size_type storage_size (size_type size_i, size_type size_j) {
            // Guard against size_type overflow
            BOOST_UBLAS_CHECK (size_i == 0 || size_j <= (std::numeric_limits<size_type>::max) () / size_i, bad_size ());
            return size_i * size_j;
        }

        // Indexing conversion to storage element
        static
        BOOST_UBLAS_INLINE
        size_type element (size_type i, size_type size_i, size_type j, size_type size_j) {
            BOOST_UBLAS_CHECK (i < size_i, bad_index ());
            BOOST_UBLAS_CHECK (j < size_j, bad_index ());
            detail::ignore_unused_variable_warning(size_j);
            // Guard against size_type overflow
            BOOST_UBLAS_CHECK (j <= ((std::numeric_limits<size_type>::max) () - i) / size_i, bad_index ());
            return i + j * size_i;
        }
        static
        BOOST_UBLAS_INLINE
        size_type address (size_type i, size_type size_i, size_type j, size_type size_j) {
            BOOST_UBLAS_CHECK (i <= size_i, bad_index ());
            BOOST_UBLAS_CHECK (j <= size_j, bad_index ());
            detail::ignore_unused_variable_warning(size_j);
            // Guard against size_type overflow - address may be size_i past end of storage
            BOOST_UBLAS_CHECK (size_i == 0 || j <= ((std::numeric_limits<size_type>::max) () - i) / size_i, bad_index ());
            return i + j * size_i;
        }

        // Storage element to index conversion
        static
        BOOST_UBLAS_INLINE
        difference_type distance_i (difference_type k, size_type /* size_i */, size_type /* size_j */) {
            return k;
        }
        static
        BOOST_UBLAS_INLINE
        difference_type distance_j (difference_type k, size_type size_i, size_type /* size_j */) {
            return size_i != 0 ? k / size_i : 0;
        }
        static
        BOOST_UBLAS_INLINE
        size_type index_i (difference_type k, size_type size_i, size_type /* size_j */) {
            return size_i != 0 ? k % size_i : 0;
        }
        static
        BOOST_UBLAS_INLINE
        size_type index_j (difference_type k, size_type size_i, size_type /* size_j */) {
            return size_i != 0 ? k / size_i : 0;
        }
        static
        BOOST_UBLAS_INLINE
        bool fast_i () {
            return true;
        }
        static
        BOOST_UBLAS_INLINE
        bool fast_j () {
            return false;
        }

        // Iterating
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void increment_i (I &it, size_type /* size_i */, size_type /* size_j */) {
            ++ it;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void increment_i (I &it, difference_type n, size_type /* size_i */, size_type /* size_j */) {
            it += n;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void decrement_i (I &it, size_type /* size_i */, size_type /* size_j */) {
            -- it;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void decrement_i (I &it, difference_type n, size_type /* size_i */, size_type /* size_j */) {
            it -= n;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void increment_j (I &it, size_type size_i, size_type /* size_j */) {
            it += size_i;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void increment_j (I &it, difference_type n, size_type size_i, size_type /* size_j */) {
            it += n * size_i;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void decrement_j (I &it, size_type size_i, size_type /* size_j */) {
            it -= size_i;
        }
        template<class I>
        static
        BOOST_UBLAS_INLINE
        void decrement_j (I &it, difference_type n, size_type size_i, size_type /* size_j */) {
            it -= n* size_i;
        }

        // Triangular access
        static
        BOOST_UBLAS_INLINE
        size_type triangular_size (size_type size_i, size_type size_j) {
            size_type size = (std::max) (size_i, size_j);
            // Guard against size_type overflow - siboost::mplified
            BOOST_UBLAS_CHECK (size == 0 || size / 2 < (std::numeric_limits<size_type>::max) () / size /* +1/2 */, bad_size ());
            return ((size + 1) * size) / 2;
        }
        static
        BOOST_UBLAS_INLINE
        size_type lower_element (size_type i, size_type size_i, size_type j, size_type size_j) {
            BOOST_UBLAS_CHECK (i < size_i, bad_index ());
            BOOST_UBLAS_CHECK (j < size_j, bad_index ());
            BOOST_UBLAS_CHECK (i >= j, bad_index ());
            // FIXME size_type overflow
            // sigma_j (size - j) = size * j - j * (j - 1) / 2
            // j = 0 1 2 3, sigma = 0 4 7 9
            return i - j + (j * (2 * (std::max) (size_i, size_j) - j + 1)) / 2;
        }
        static
        BOOST_UBLAS_INLINE
        size_type upper_element (size_type i, size_type size_i, size_type j, size_type size_j) {
            BOOST_UBLAS_CHECK (i < size_i, bad_index ());
            BOOST_UBLAS_CHECK (j < size_j, bad_index ());
            BOOST_UBLAS_CHECK (i <= j, bad_index ());
            // FIXME size_type overflow
            // sigma_j (j + 1) = (j + 1) * j / 2
            // j = 0 1 2 3, sigma = 0 1 3 6
            return i + ((j + 1) * j) / 2;
        }

        // Major and minor indices
        static
        BOOST_UBLAS_INLINE
        size_type index_M (size_type /* index1 */, size_type index2) {
            return index2;
        }
        static
        BOOST_UBLAS_INLINE
        size_type index_m (size_type index1, size_type /* index2 */) {
            return index1;
        }
        static
        BOOST_UBLAS_INLINE
        size_type size_M (size_type /* size_i */, size_type size_j) {
            return size_j;
        }
        static
        BOOST_UBLAS_INLINE
        size_type size_m (size_type size_i, size_type /* size_j */) {
            return size_i;
        }
    };


    template <class Z>
    struct basic_full {
        typedef Z size_type;

        template<class L>
        static
        BOOST_UBLAS_INLINE
        size_type packed_size (L, size_type size_i, size_type size_j) {
            return L::storage_size (size_i, size_j);
        }

        static
        BOOST_UBLAS_INLINE
        bool zero (size_type /* i */, size_type /* j */) {
            return false;
        }
        static
        BOOST_UBLAS_INLINE
        bool one (size_type /* i */, size_type /* j */) {
            return false;
        }
        static
        BOOST_UBLAS_INLINE
        bool other (size_type /* i */, size_type /* j */) {
            return true;
        }
        // FIXME: this should not be used at all
        static
        BOOST_UBLAS_INLINE
        size_type restrict1 (size_type i, size_type /* j */) {
            return i;
        }
        static
        BOOST_UBLAS_INLINE
        size_type restrict2 (size_type /* i */, size_type j) {
            return j;
        }
        static
        BOOST_UBLAS_INLINE
        size_type mutable_restrict1 (size_type i, size_type /* j */) {
            return i;
        }
        static
        BOOST_UBLAS_INLINE
        size_type mutable_restrict2 (size_type /* i */, size_type j) {
            return j;
        }
    };

    namespace detail {
        template < class L >
        struct transposed_structure {
            typedef typename L::size_type size_type;

            template<class LAYOUT>
            static
            BOOST_UBLAS_INLINE
            size_type packed_size (LAYOUT l, size_type size_i, size_type size_j) {
                return L::packed_size(l, size_j, size_i);
            }

            static
            BOOST_UBLAS_INLINE
            bool zero (size_type i, size_type j) {
                return L::zero(j, i);
            }
            static
            BOOST_UBLAS_INLINE
            bool one (size_type i, size_type j) {
                return L::one(j, i);
            }
            static
            BOOST_UBLAS_INLINE
            bool other (size_type i, size_type j) {
                return L::other(j, i);
            }
            template<class LAYOUT>
            static
            BOOST_UBLAS_INLINE
            size_type element (LAYOUT /* l */, size_type i, size_type size_i, size_type j, size_type size_j) {
                return L::element(typename LAYOUT::transposed_layout(), j, size_j, i, size_i);
            }

            static
            BOOST_UBLAS_INLINE
            size_type restrict1 (size_type i, size_type j, size_type size1, size_type size2) {
                return L::restrict2(j, i, size2, size1);
            }
            static
            BOOST_UBLAS_INLINE
            size_type restrict2 (size_type i, size_type j, size_type size1, size_type size2) {
                return L::restrict1(j, i, size2, size1);
            }
            static
            BOOST_UBLAS_INLINE
            size_type mutable_restrict1 (size_type i, size_type j, size_type size1, size_type size2) {
                return L::mutable_restrict2(j, i, size2, size1);
            }
            static
            BOOST_UBLAS_INLINE
            size_type mutable_restrict2 (size_type i, size_type j, size_type size1, size_type size2) {
                return L::mutable_restrict1(j, i, size2, size1);
            }

            static
            BOOST_UBLAS_INLINE
            size_type global_restrict1 (size_type index1, size_type size1, size_type index2, size_type size2) {
                return L::global_restrict2(index2, size2, index1, size1);
            }
            static
            BOOST_UBLAS_INLINE
            size_type global_restrict2 (size_type index1, size_type size1, size_type index2, size_type size2) {
                return L::global_restrict1(index2, size2, index1, size1);
            }
            static
            BOOST_UBLAS_INLINE
            size_type global_mutable_restrict1 (size_type index1, size_type size1, size_type index2, size_type size2) {
                return L::global_mutable_restrict2(index2, size2, index1, size1);
            }
            static
            BOOST_UBLAS_INLINE
            size_type global_mutable_restrict2 (size_type index1, size_type size1, size_type index2, size_type size2) {
                return L::global_mutable_restrict1(index2, size2, index1, size1);
            }
        };
    }

    template <class Z>
    struct basic_lower {
        typedef Z size_type;
        typedef lower_tag triangular_type;

        template<class L>
        static
        BOOST_UBLAS_INLINE
        size_type packed_size (L, size_type size_i, size_type size_j) {
            return L::triangular_size (size_i, size_j);
        }

        static
        BOOST_UBLAS_INLINE
        bool zero (size_type i, size_type j) {
            return j > i;
        }
        static
        BOOST_UBLAS_INLINE
        bool one (size_type /* i */, size_type /* j */) {
            return false;
        }
        static
        BOOST_UBLAS_INLINE
        bool other (size_type i, size_type j) {
            return j <= i;
        }
        template<class L>
        static
        BOOST_UBLAS_INLINE
        size_type element (L, size_type i, size_type size_i, size_type j, size_type size_j) {
            return L::lower_element (i, size_i, j, size_j);
        }

        // return nearest valid index in column j
        static
        BOOST_UBLAS_INLINE
        size_type restrict1 (size_type i, size_type j, size_type size1, size_type /* size2 */) {
            return (std::max)(j, (std::min) (size1, i));
        }
        // return nearest valid index in row i
        static
        BOOST_UBLAS_INLINE
        size_type restrict2 (size_type i, size_type j, size_type /* size1 */, size_type /* size2 */) {
            return (std::max)(size_type(0), (std::min) (i+1, j));
        }
        // return nearest valid mutable index in column j
        static
        BOOST_UBLAS_INLINE
        size_type mutable_restrict1 (size_type i, size_type j, size_type size1, size_type /* size2 */) {
            return (std::max)(j, (std::min) (size1, i));
        }
        // return nearest valid mutable index in row i
        static
        BOOST_UBLAS_INLINE
        size_type mutable_restrict2 (size_type i, size_type j, size_type /* size1 */, size_type /* size2 */) {
            return (std::max)(size_type(0), (std::min) (i+1, j));
        }

        // return an index between the first and (1+last) filled row
        static
        BOOST_UBLAS_INLINE
        size_type global_restrict1 (size_type index1, size_type size1, size_type /* index2 */, size_type /* size2 */) {
            return (std::max)(size_type(0), (std::min)(size1, index1) );
        }
        // return an index between the first and (1+last) filled column
        static
        BOOST_UBLAS_INLINE
        size_type global_restrict2 (size_type /* index1 */, size_type /* size1 */, size_type index2, size_type size2) {
            return (std::max)(size_type(0), (std::min)(size2, index2) );
        }

        // return an index between the first and (1+last) filled mutable row
        static
        BOOST_UBLAS_INLINE
        size_type global_mutable_restrict1 (size_type index1, size_type size1, size_type /* index2 */, size_type /* size2 */) {
            return (std::max)(size_type(0), (std::min)(size1, index1) );
        }
        // return an index between the first and (1+last) filled mutable column
        static
        BOOST_UBLAS_INLINE
        size_type global_mutable_restrict2 (size_type /* index1 */, size_type /* size1 */, size_type index2, size_type size2) {
            return (std::max)(size_type(0), (std::min)(size2, index2) );
        }
    };

    // the first row only contains a single 1. Thus it is not stored.
    template <class Z>
    struct basic_unit_lower : public basic_lower<Z> {
        typedef Z size_type;
        typedef unit_lower_tag triangular_type;

        template<class L>
        static
        BOOST_UBLAS_INLINE
        size_type packed_size (L, size_type size_i, size_type size_j) {
            // Zero size strict triangles are bad at this point
            BOOST_UBLAS_CHECK (size_i != 0 && size_j != 0, bad_index ());
            return L::triangular_size (size_i - 1, size_j - 1);
        }

        static
        BOOST_UBLAS_INLINE
        bool one (size_type i, size_type j) {
            return j == i;
        }
        static
        BOOST_UBLAS_INLINE
        bool other (size_type i, size_type j) {
            return j < i;
        }
        template<class L>
        static
        BOOST_UBLAS_INLINE
        size_type element (L, size_type i, size_type size_i, size_type j, size_type size_j) {
            // Zero size strict triangles are bad at this point
            BOOST_UBLAS_CHECK (size_i != 0 && size_j != 0 && i != 0, bad_index ());
            return L::lower_element (i-1, size_i - 1, j, size_j - 1);
        }

        static
        BOOST_UBLAS_INLINE
        size_type mutable_restrict1 (size_type i, size_type j, size_type size1, size_type /* size2 */) {
            return (std::max)(j+1, (std::min) (size1, i));
        }
        static
        BOOST_UBLAS_INLINE
        size_type mutable_restrict2 (size_type i, size_type j, size_type /* size1 */, size_type /* size2 */) {
            return (std::max)(size_type(0), (std::min) (i, j));
        }

        // return an index between the first and (1+last) filled mutable row
        static
        BOOST_UBLAS_INLINE
        size_type global_mutable_restrict1 (size_type index1, size_type size1, size_type /* index2 */, size_type /* size2 */) {
            return (std::max)(size_type(1), (std::min)(size1, index1) );
        }
        // return an index between the first and (1+last) filled mutable column
        static
        BOOST_UBLAS_INLINE
        size_type global_mutable_restrict2 (size_type /* index1 */, size_type /* size1 */, size_type index2, size_type size2) {
            BOOST_UBLAS_CHECK( size2 >= 1 , external_logic() );
            return (std::max)(size_type(0), (std::min)(size2-1, index2) );
        }
    };

    // the first row only contains no element. Thus it is not stored.
    template <class Z>
    struct basic_strict_lower : public basic_unit_lower<Z> {
        typedef Z size_type;
        typedef strict_lower_tag triangular_type;

        template<class L>
        static
        BOOST_UBLAS_INLINE
        size_type packed_size (L, size_type size_i, size_type size_j) {
            // Zero size strict triangles are bad at this point
            BOOST_UBLAS_CHECK (size_i != 0 && size_j != 0, bad_index ());
            return L::triangular_size (size_i - 1, size_j - 1);
        }

        static
        BOOST_UBLAS_INLINE
        bool zero (size_type i, size_type j) {
            return j >= i;
        }
        static
        BOOST_UBLAS_INLINE
        bool one (size_type /*i*/, size_type /*j*/) {
            return false;
        }
        static
        BOOST_UBLAS_INLINE
        bool other (size_type i, size_type j) {
            return j < i;
        }
        template<class L>
        static
        BOOST_UBLAS_INLINE
        size_type element (L, size_type i, size_type size_i, size_type j, size_type size_j) {
            // Zero size strict triangles are bad at this point
            BOOST_UBLAS_CHECK (size_i != 0 && size_j != 0 && i != 0, bad_index ());
            return L::lower_element (i-1, size_i - 1, j, size_j - 1);
        }

        static
        BOOST_UBLAS_INLINE
        size_type restrict1 (size_type i, size_type j, size_type size1, size_type size2) {
            return basic_unit_lower<Z>::mutable_restrict1(i, j, size1, size2);
        }
        static
        BOOST_UBLAS_INLINE
        size_type restrict2 (size_type i, size_type j, size_type size1, size_type size2) {
            return basic_unit_lower<Z>::mutable_restrict2(i, j, size1, size2);
        }

        // return an index between the first and (1+last) filled row
        static
        BOOST_UBLAS_INLINE
        size_type global_restrict1 (size_type index1, size_type size1, size_type index2, size_type size2) {
            return basic_unit_lower<Z>::global_mutable_restrict1(index1, size1, index2, size2);
        }
        // return an index between the first and (1+last) filled column
        static
        BOOST_UBLAS_INLINE
        size_type global_restrict2 (size_type index1, size_type size1, size_type index2, size_type size2) {
            return basic_unit_lower<Z>::global_mutable_restrict2(index1, size1, index2, size2);
        }
    };


    template <class Z>
    struct basic_upper : public detail::transposed_structure<basic_lower<Z> >
    { 
        typedef upper_tag triangular_type;
    };

    template <class Z>
    struct basic_unit_upper : public detail::transposed_structure<basic_unit_lower<Z> >
    { 
        typedef unit_upper_tag triangular_type;
    };

    template <class Z>
    struct basic_strict_upper : public detail::transposed_structure<basic_strict_lower<Z> >
    { 
        typedef strict_upper_tag triangular_type;
    };


}}

#endif
