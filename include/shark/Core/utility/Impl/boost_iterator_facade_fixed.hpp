// (C) Copyright David Abrahams 2002.
// (C) Copyright Jeremy Siek    2002.
// (C) Copyright Thomas Witt    2002.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

///OK:
///yeah... boost iterators are broken. so we include here a fixed version.

#ifndef BOOST_ITERATOR_FACADE_23022003THW_HPP_FIXED
#define BOOST_ITERATOR_FACADE_23022003THW_HPP_FIXED

#if BOOST_VERSION < 105700
#define SHARK_USE_ITERATOR_WORKAROUND
#endif

#ifdef SHARK_USE_ITERATOR_WORKAROUND

#define SHARK_ITERATOR_FACADE boost::iterator_facade_fixed
#define SHARK_ITERATOR_CORE_ACCESS boost::iterator_core_access_fixed

#include <boost/iterator/iterator_facade.hpp>

#include <boost/iterator/detail/config_def.hpp> // this goes last

namespace boost
{
  // This forward declaration is required for the friend declaration
  // in iterator_core_access_fixed
  template <class I, class V, class TC, class R, class D> class iterator_facade_fixed;

    // Macros which describe the declarations of binary operators
# ifdef BOOST_NO_STRICT_ITERATOR_INTEROPERABILITY
#  define BOOST_ITERATOR_FACADE_INTEROP_HEAD(prefix, op, result_type)       \
    template <                                                              \
        class Derived1, class V1, class TC1, class Reference1, class Difference1 \
      , class Derived2, class V2, class TC2, class Reference2, class Difference2 \
    >                                                                       \
    prefix typename mpl::apply2<result_type,Derived1,Derived2>::type \
    operator op(                                                            \
        iterator_facade_fixed<Derived1, V1, TC1, Reference1, Difference1> const& lhs   \
      , iterator_facade_fixed<Derived2, V2, TC2, Reference2, Difference2> const& rhs)
# else 
#  define BOOST_ITERATOR_FACADE_INTEROP_HEAD(prefix, op, result_type)   \
    template <                                                          \
        class Derived1, class V1, class TC1, class Reference1, class Difference1 \
      , class Derived2, class V2, class TC2, class Reference2, class Difference2 \
    >                                                                   \
    prefix typename boost::detail::enable_if_interoperable<             \
        Derived1, Derived2                                              \
      , typename mpl::apply2<result_type,Derived1,Derived2>::type       \
    >::type                                                             \
    operator op(                                                        \
        iterator_facade_fixed<Derived1, V1, TC1, Reference1, Difference1> const& lhs   \
      , iterator_facade_fixed<Derived2, V2, TC2, Reference2, Difference2> const& rhs)
# endif 

#  define BOOST_ITERATOR_FACADE_PLUS_HEAD(prefix,args)              \
    template <class Derived, class V, class TC, class R, class D>   \
    prefix Derived operator+ args
  class iterator_core_access_fixed
  {
# if defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS)                  
      // Tasteless as this may seem, making all members public allows member templates
      // to work in the absence of member template friends.
   public:
# else
      
      template <class I, class V, class TC, class R, class D> friend class iterator_facade_fixed;

#  define BOOST_ITERATOR_FACADE_RELATION(op)                                \
      BOOST_ITERATOR_FACADE_INTEROP_HEAD(friend,op, boost::detail::always_bool2);

      BOOST_ITERATOR_FACADE_RELATION(==)
      BOOST_ITERATOR_FACADE_RELATION(!=)

      BOOST_ITERATOR_FACADE_RELATION(<)
      BOOST_ITERATOR_FACADE_RELATION(>)
      BOOST_ITERATOR_FACADE_RELATION(<=)
      BOOST_ITERATOR_FACADE_RELATION(>=)
#  undef BOOST_ITERATOR_FACADE_RELATION

      BOOST_ITERATOR_FACADE_INTEROP_HEAD(
          friend, -, boost::detail::choose_difference_type)
      ;

      BOOST_ITERATOR_FACADE_PLUS_HEAD(
          friend inline
          , (iterator_facade_fixed<Derived, V, TC, R, D> const&
           , typename Derived::difference_type)
      )
      ;

      BOOST_ITERATOR_FACADE_PLUS_HEAD(
          friend inline
        , (typename Derived::difference_type
           , iterator_facade_fixed<Derived, V, TC, R, D> const&)
      )
      ;

# endif

      template <class Facade>
      static typename Facade::reference dereference(Facade const& f)
      {
          return f.dereference();
      }

      template <class Facade>
      static void increment(Facade& f)
      {
          f.increment();
      }

      template <class Facade>
      static void decrement(Facade& f)
      {
          f.decrement();
      }

      template <class Facade1, class Facade2>
      static bool equal(Facade1 const& f1, Facade2 const& f2, mpl::true_)
      {
          return f1.equal(f2);
      }

      template <class Facade1, class Facade2>
      static bool equal(Facade1 const& f1, Facade2 const& f2, mpl::false_)
      {
          return f2.equal(f1);
      }

      template <class Facade>
      static void advance(Facade& f, typename Facade::difference_type n)
      {
          f.advance(n);
      }

      template <class Facade1, class Facade2>
      static typename Facade1::difference_type distance_from(
          Facade1 const& f1, Facade2 const& f2, mpl::true_)
      {
          return -f1.distance_to(f2);
      }

      template <class Facade1, class Facade2>
      static typename Facade2::difference_type distance_from(
          Facade1 const& f1, Facade2 const& f2, mpl::false_)
      {
          return f2.distance_to(f1);
      }

      //
      // Curiously Recurring Template interface.
      //
      template <class I, class V, class TC, class R, class D>
      static I& derived(iterator_facade_fixed<I,V,TC,R,D>& facade)
      {
          return *static_cast<I*>(&facade);
      }

      template <class I, class V, class TC, class R, class D>
      static I const& derived(iterator_facade_fixed<I,V,TC,R,D> const& facade)
      {
          return *static_cast<I const*>(&facade);
      }

   private:
      // objects of this class are useless
      iterator_core_access_fixed(); //undefined
  };

  
  namespace detail{

	///HERE THE FIX IS APPLIED!!!
	
    // operator->() needs special support for input iterators to strictly meet the
    // standard's requirements. If *i is not a reference type, we must still
    // produce a lvalue to which a pointer can be formed. We do that by
    // returning an instantiation of this special proxy class template.
    template <class Reference>
    struct operator_arrow_proxy_fixed
    {
        operator_arrow_proxy_fixed(Reference ref) : m_ref(ref) {}
        Reference* operator->() const { return &m_ref; }
        // This function is needed for MWCW and BCC, which won't call operator->
        // again automatically per 13.3.1.2 para 8
        operator Reference*() const { return &m_ref; }
        mutable Reference m_ref;
    };
	template <class T>
    struct operator_arrow_proxy_fixed<T&>
    {
        operator_arrow_proxy_fixed(T& ref) : m_ref(ref) {}
        T* operator->() const { return &m_ref; }
        // This function is needed for MWCW and BCC, which won't call operator->
        // again automatically per 13.3.1.2 para 8
        operator T*() const { return &m_ref; }
        T& m_ref;
    };
    // A metafunction that gets the result type for operator->.  Also
    // has a static function make() which builds the result from a
    // Reference
    template <class ValueType, class Reference, class Pointer>
    struct operator_arrow_result_fixed
    {
        // CWPro8.3 won't accept "operator_arrow_result::type", and we
        // need that type below, so metafunction forwarding would be a
        // losing proposition here.
//        typedef typename mpl::if_<
//            is_reference<Reference>
//          , Pointer
//          , operator_arrow_proxy<Reference>
//        >::type type;
		typedef operator_arrow_proxy_fixed<Reference> type;
        static type make(Reference x)
        {
            return type(x);
        }
    };

# if BOOST_WORKAROUND(BOOST_MSVC, < 1300)
    // Deal with ETI
    template<>
    struct operator_arrow_result_fixed<int, int, int>
    {
        typedef int type;
    };
# endif
}

  //
  // iterator_facade - use as a public base class for defining new
  // standard-conforming iterators.
  //
  template <
      class Derived             // The derived iterator type being constructed
    , class Value
    , class CategoryOrTraversal
    , class Reference   = Value&
    , class Difference  = std::ptrdiff_t
  >
  class iterator_facade_fixed
# ifdef BOOST_ITERATOR_FACADE_NEEDS_ITERATOR_BASE
    : public boost::detail::iterator_facade_types<
         Value, CategoryOrTraversal, Reference, Difference
      >::base
#  undef BOOST_ITERATOR_FACADE_NEEDS_ITERATOR_BASE
# endif
  {
   private:
      //
      // Curiously Recurring Template interface.
      //
      Derived& derived()
      {
          return *static_cast<Derived*>(this);
      }

      Derived const& derived() const
      {
          return *static_cast<Derived const*>(this);
      }

      typedef boost::detail::iterator_facade_types<
         Value, CategoryOrTraversal, Reference, Difference
      > associated_types;

      typedef boost::detail::operator_arrow_result_fixed<
        typename associated_types::value_type
        , Reference
        , typename associated_types::pointer
      > pointer_;

   protected:
      // For use by derived classes
      typedef iterator_facade_fixed<Derived,Value,CategoryOrTraversal,Reference,Difference> iterator_facade_;
      
   public:

      typedef typename associated_types::value_type value_type;
      typedef Reference reference;
      typedef Difference difference_type;

      typedef typename pointer_::type pointer;

      typedef typename associated_types::iterator_category iterator_category;

      reference operator*() const
      {
          return iterator_core_access_fixed::dereference(this->derived());
      }

      pointer operator->() const
      {
          return pointer_::make(*this->derived());
      }
        
      typename boost::detail::operator_brackets_result<Derived,Value,reference>::type
      operator[](difference_type n) const
      {
          typedef boost::detail::use_operator_brackets_proxy<Value,Reference> use_proxy;
          
          return boost::detail::make_operator_brackets_result<Derived>(
              this->derived() + n
            , use_proxy()
          );
      }

      Derived& operator++()
      {
          iterator_core_access_fixed::increment(this->derived());
          return this->derived();
      }

# if BOOST_WORKAROUND(BOOST_MSVC, < 1300)
      typename boost::detail::postfix_increment_result<Derived,Value,Reference,CategoryOrTraversal>::type
      operator++(int)
      {
          typename boost::detail::postfix_increment_result<Derived,Value,Reference,CategoryOrTraversal>::type
          tmp(this->derived());
          ++*this;
          return tmp;
      }
# endif
      
      Derived& operator--()
      {
          iterator_core_access_fixed::decrement(this->derived());
          return this->derived();
      }

      Derived operator--(int)
      {
          Derived tmp(this->derived());
          --*this;
          return tmp;
      }

      Derived& operator+=(difference_type n)
      {
          iterator_core_access_fixed::advance(this->derived(), n);
          return this->derived();
      }

      Derived& operator-=(difference_type n)
      {
          iterator_core_access_fixed::advance(this->derived(), -n);
          return this->derived();
      }

      Derived operator-(difference_type x) const
      {
          Derived result(this->derived());
          return result -= x;
      }

# if BOOST_WORKAROUND(BOOST_MSVC, < 1300)
      // There appears to be a bug which trashes the data of classes
      // derived from iterator_facade when they are assigned unless we
      // define this assignment operator.  This bug is only revealed
      // (so far) in STLPort debug mode, but it's clearly a codegen
      // problem so we apply the workaround for all MSVC6.
      iterator_facade_fixed& operator=(iterator_facade_fixed const&)
      {
          return *this;
      }
# endif
  };

# if !BOOST_WORKAROUND(BOOST_MSVC, < 1300)
  template <class I, class V, class TC, class R, class D>
  inline typename boost::detail::postfix_increment_result<I,V,R,TC>::type
  operator++(
      iterator_facade_fixed<I,V,TC,R,D>& i
    , int
  )
  {
      typename boost::detail::postfix_increment_result<I,V,R,TC>::type
          tmp(*static_cast<I*>(&i));
      
      ++i;
      
      return tmp;
  }
# endif 

  
  //
  // Comparison operator implementation. The library supplied operators
  // enables the user to provide fully interoperable constant/mutable
  // iterator types. I.e. the library provides all operators
  // for all mutable/constant iterator combinations.
  //
  // Note though that this kind of interoperability for constant/mutable
  // iterators is not required by the standard for container iterators.
  // All the standard asks for is a conversion mutable -> constant.
  // Most standard library implementations nowadays provide fully interoperable
  // iterator implementations, but there are still heavily used implementations
  // that do not provide them. (Actually it's even worse, they do not provide
  // them for only a few iterators.)
  //
  // ?? Maybe a BOOST_ITERATOR_NO_FULL_INTEROPERABILITY macro should
  //    enable the user to turn off mixed type operators
  //
  // The library takes care to provide only the right operator overloads.
  // I.e.
  //
  // bool operator==(Iterator,      Iterator);
  // bool operator==(ConstIterator, Iterator);
  // bool operator==(Iterator,      ConstIterator);
  // bool operator==(ConstIterator, ConstIterator);
  //
  //   ...
  //
  // In order to do so it uses c++ idioms that are not yet widely supported
  // by current compiler releases. The library is designed to degrade gracefully
  // in the face of compiler deficiencies. In general compiler
  // deficiencies result in less strict error checking and more obscure
  // error messages, functionality is not affected.
  //
  // For full operation compiler support for "Substitution Failure Is Not An Error"
  // (aka. enable_if) and boost::is_convertible is required.
  //
  // The following problems occur if support is lacking.
  //
  // Pseudo code
  //
  // ---------------
  // AdaptorA<Iterator1> a1;
  // AdaptorA<Iterator2> a2;
  //
  // // This will result in a no such overload error in full operation
  // // If enable_if or is_convertible is not supported
  // // The instantiation will fail with an error hopefully indicating that
  // // there is no operator== for Iterator1, Iterator2
  // // The same will happen if no enable_if is used to remove
  // // false overloads from the templated conversion constructor
  // // of AdaptorA.
  //
  // a1 == a2;
  // ----------------
  //
  // AdaptorA<Iterator> a;
  // AdaptorB<Iterator> b;
  //
  // // This will result in a no such overload error in full operation
  // // If enable_if is not supported the static assert used
  // // in the operator implementation will fail.
  // // This will accidently work if is_convertible is not supported.
  //
  // a == b;
  // ----------------
  //

# ifdef BOOST_NO_ONE_WAY_ITERATOR_INTEROP
#  define BOOST_ITERATOR_CONVERTIBLE(a,b) mpl::true_()
# else
#  define BOOST_ITERATOR_CONVERTIBLE(a,b) is_convertible<a,b>()
# endif

# define BOOST_ITERATOR_FACADE_INTEROP(op, result_type, return_prefix, base_op) \
  BOOST_ITERATOR_FACADE_INTEROP_HEAD(inline, op, result_type)                   \
  {                                                                             \
      /* For those compilers that do not support enable_if */                   \
      BOOST_STATIC_ASSERT((                                                     \
          is_interoperable< Derived1, Derived2 >::value                         \
      ));                                                                       \
      return_prefix iterator_core_access_fixed::base_op(                              \
          *static_cast<Derived1 const*>(&lhs)                                   \
        , *static_cast<Derived2 const*>(&rhs)                                   \
        , BOOST_ITERATOR_CONVERTIBLE(Derived2,Derived1)                         \
      );                                                                        \
  }

# define BOOST_ITERATOR_FACADE_RELATION(op, return_prefix, base_op) \
  BOOST_ITERATOR_FACADE_INTEROP(                                    \
      op                                                            \
    , boost::detail::always_bool2                                   \
    , return_prefix                                                 \
    , base_op                                                       \
  )

  BOOST_ITERATOR_FACADE_RELATION(==, return, equal)
  BOOST_ITERATOR_FACADE_RELATION(!=, return !, equal)

  BOOST_ITERATOR_FACADE_RELATION(<, return 0 >, distance_from)
  BOOST_ITERATOR_FACADE_RELATION(>, return 0 <, distance_from)
  BOOST_ITERATOR_FACADE_RELATION(<=, return 0 >=, distance_from)
  BOOST_ITERATOR_FACADE_RELATION(>=, return 0 <=, distance_from)
# undef BOOST_ITERATOR_FACADE_RELATION

  // operator- requires an additional part in the static assertion
  BOOST_ITERATOR_FACADE_INTEROP(
      -
    , boost::detail::choose_difference_type
    , return
    , distance_from
  )
# undef BOOST_ITERATOR_FACADE_INTEROP
# undef BOOST_ITERATOR_FACADE_INTEROP_HEAD

# define BOOST_ITERATOR_FACADE_PLUS(args)           \
  BOOST_ITERATOR_FACADE_PLUS_HEAD(inline, args)     \
  {                                                 \
      Derived tmp(static_cast<Derived const&>(i));  \
      return tmp += n;                              \
  }

BOOST_ITERATOR_FACADE_PLUS((
  iterator_facade_fixed<Derived, V, TC, R, D> const& i
  , typename Derived::difference_type n
))

BOOST_ITERATOR_FACADE_PLUS((
    typename Derived::difference_type n
    , iterator_facade_fixed<Derived, V, TC, R, D> const& i
))
# undef BOOST_ITERATOR_FACADE_PLUS
# undef BOOST_ITERATOR_FACADE_PLUS_HEAD

} // namespace boost

#include <boost/iterator/detail/config_undef.hpp>
#else
#define SHARK_ITERATOR_FACADE boost::iterator_facade
#define SHARK_ITERATOR_CORE_ACCESS boost::iterator_core_access
#endif

#endif // BOOST_ITERATOR_FACADE_23022003THW_HPP
