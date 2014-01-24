/*!
 * 
 * \file        Impl/Initialize.h
 *
 * \brief       Easy initialization of vectors
 * 
 * 
 *
 * \author      O.Krause, T.Glasmachers
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
#ifndef SHARK_LINALG_IMPL_INITIALIZE_H
#define SHARK_LINALG_IMPL_INITIALIZE_H

#include <shark/Core/Exception.h>
#include <shark/LinAlg/BLAS/blas.h>
#include <iterator>
#include <boost/type_traits/remove_reference.hpp>
namespace shark{
namespace blas{
namespace detail{
	
//The  following looks a bit complicated and in fact, there ARE easier solutions.
//However, I wanted to be able to check the sizes in debug. and this is only possible if you know
//1) how big the whole sequence a,b,c is
//2) that c is the last element. 
//also 3) i wanted to prevent init(vec),a,b,c as initialization abuse and instead force init(vec)<<a,b,c
//without 1-3, one would only need to pass pointers around and just increment them at every call of operator<< or ,
	
//so how does it work?
//when the user writes an expression like init(vec)<<a,b,c; the operators creates a complex template type which is a tree form of the expression.
//every operator forms a node of the tree and the arguments are the leaves. During evaluation, the tree is traversed recursively to initialize or split the vector. 
//Construction of the TreeType: it is just a nested template where operators are represented by a Type InitializerNode<left,right>
//depending of the Type of the Argument, the Type of the leaf is chosen.  so  if a is a vector and b and c are doubles, we get something similar to
//Node<Node<InitializerEnd,VectorExpression >,Scalar<double> >,Scalar<double> > where InitializerEnd marks the end of the expression.

//sometimes a wrapper is constructed using calls like vectors(). In this case, the Wrappers all have a common base class Initialize<T>, where
//T is the own type. e.g. class A:public Initialized<A>{}; Using this technique, only one operator overload is needed for all wrappers since the compiler
//will choose the version which takes canst Initializes<T>& as argument.

///\brief Wrapper to enable argument dependent lookup for ublas vectors in shark namespace.
///
///Without, the compiler is not able to find the correct operator overload. We create it using init(Vector)
template<class T>
struct ADLVector{
	T vector;
	ADLVector(T vector):vector(vector){}
};


//We implement first some wrappers which handle the different types of expressions which can be used.

//implementation detail: 
//there exists a convention in ublas, that only const iterators of sparse vectors are sparse, while mutable iterators
//iterate over all elements. This is only documented on their mailing list.
//so in the following...200 lines of code we will always iterate using the const iterators and use the indices when the values need to be changed.


//the following types are basic wrappers which linearizes their arguments for initialization with init()
//or split their part from the big vector using split(). They form the leaves of the expression-tree.
//Additionally they have a size element, which is used to calculate the size of the complete right side expression in debug mode
	
///\brief Wrapper representing a single arithmetic value.
template<class T>
class Scalar{
private:
	T& m_value;
public:
	Scalar(T& value):m_value(value){}
	template<class Iter>
	void init(Iter& pos)const{
		*pos = m_value;
		++pos;
	}
	template<class Iter>
	void split(Iter& pos){
		m_value = *pos;
		++pos;
	}
	std::size_t size()const{
		return 1;
	}
};

///\brief Expression template as base class for all initializer expressions.
template<class Self>
struct InitializerBase{
	Self& operator()(){
		return static_cast<Self&>(*this);
	}
	const Self& operator()()const{
		return static_cast<const Self&>(*this);
	}
};

///\brief Wrapper for all kinds of vectorexpression which saves the (sparse) vector in the target vector.
template<class Expression>
class VectorExpression :public InitializerBase<VectorExpression<Expression> >{
private:
	Expression m_expression;
	typedef typename boost::remove_reference<Expression>::type::const_iterator IndexIterator;
public:
	VectorExpression(Expression expression):m_expression(expression){}
	template<class Iter>
	void init(Iter& pos)const{
		for(IndexIterator vecPos = m_expression.begin(); vecPos != m_expression.end();++vecPos, ++pos){
			*pos = *vecPos;
		}
	}
	template<class Iter>
	void split(Iter& pos){
		for(IndexIterator vecPos = m_expression.begin(); vecPos != m_expression.end();++vecPos, ++pos){
			m_expression(vecPos.index()) = *pos;
		}
	}
	std::size_t size()const{
		IndexIterator begin = m_expression.begin();
		IndexIterator end = m_expression.end();
		return std::distance(begin,end);
	}
};

///\brief Wrapper for all kinds of matrix expressions which saves the (sparse) matrix row by row in the target vector.
template<class Matrix>
class MatrixExpression:public InitializerBase<MatrixExpression<Matrix> >{
private:
	Matrix& m_matrix;
	typedef typename blas::major_iterator<Matrix>::type major_iterator;
public:
	MatrixExpression(Matrix& matrix):m_matrix(matrix){}
	template<class Iter>
	void init(Iter& pos)const{
		std::size_t major = Matrix::orientation::size_M(m_matrix.size1(), m_matrix.size2());
		for(std::size_t i = 0; i != major; ++i){
			major_iterator end = major_end(m_matrix,i);
			for(major_iterator it = major_begin(m_matrix,i);it != end; ++it,++pos){
				*pos = *it;
			}
		}
	}
	template<class Iter>
	void split(Iter& pos){
		std::size_t major = Matrix::orientation::size_M(m_matrix.size1(), m_matrix.size2());
		for(std::size_t i = 0; i != major; ++i){
			major_iterator end = major_end(m_matrix,i);
			for(major_iterator it = major_begin(m_matrix,i);it != end; ++it,++pos){
				*it = *pos;
			}
		}
	}
	std::size_t size()const{
		std::size_t elements = 0;
		std::size_t major = Matrix::orientation::size_M(m_matrix.size1(), m_matrix.size2());
		for(std::size_t i = 0; i != major; ++i){
			elements += std::distance(major_begin(m_matrix,i),major_end(m_matrix,i));
		}
		return elements;
	}
};

///\brief Wrapps a parametrizable Object so that it can be used to initialize or split a vector
template<class T>
class ParameterizableExpression :public InitializerBase<ParameterizableExpression<T> >{
private:
	T& m_parameterizable;
public:
	ParameterizableExpression(T& parameterizable):m_parameterizable(parameterizable){}
	template<class Iter>
	void init(Iter& pos)const{
		blas::vector<double> parameters = m_parameterizable.parameterVector();
		VectorExpression<blas::vector<double>&> vectorWriter(parameters);
		vectorWriter.init(pos);
	}
	template<class Iter>
	void split(Iter& pos){
		blas::vector<double> parameters(size());
		VectorExpression<blas::vector<double>&> vectorWriter(parameters);
		vectorWriter.split(pos);
		m_parameterizable.setParameterVector(parameters);
	}
	std::size_t size()const{
		return m_parameterizable.numberOfParameters();
	}
};
///\brief Wrapps a pointer to a parametrizable Object so that it can be used to initialize or split a vector
template<class T>
class ParameterizableExpression<T*> :public InitializerBase<ParameterizableExpression<T*> >{
private:
	T* m_parameterizable;
public:
	ParameterizableExpression(T* parameterizable):m_parameterizable(parameterizable){}
	template<class Iter>
	void init(Iter& pos)const{
		blas::vector<double> parameters = m_parameterizable->parameterVector();
		VectorExpression<blas::vector<double>&> vectorWriter(parameters);
		vectorWriter.init(pos);
	}
	template<class Iter>
	void split(Iter& pos){
		blas::vector<double> parameters(size());
		VectorExpression<blas::vector<double>&> vectorWriter(parameters);
		vectorWriter.split(pos);
		m_parameterizable->setParameterVector(parameters);
	}
	std::size_t size()const{
		return m_parameterizable->numberOfParameters();
	}
};

template<class T>
class ParameterizableExpression<T* const> :public InitializerBase<ParameterizableExpression<T* const> >{
private:
	T* m_parameterizable;
public:
	ParameterizableExpression(T* parameterizable):m_parameterizable(parameterizable){}
	template<class Iter>
	void init(Iter& pos)const{
		blas::vector<double> parameters = m_parameterizable->parameterVector();
		VectorExpression<blas::vector<double>&> vectorWriter(parameters);
		vectorWriter.init(pos);
	}
	template<class Iter>
	void split(Iter& pos){
		blas::vector<double> parameters(size());
		VectorExpression<blas::vector<double>&> vectorWriter(parameters);
		vectorWriter.split(pos);
		m_parameterizable->setParameterVector(parameters);
	}
	std::size_t size()const{
		return m_parameterizable->numberOfParameters();
	}
};

///\brief Wrapper representing a range of vectors or matrices like std::vector<blas::vector<double>> or std::list<SparseRealMatrix>.
///
///The first template parameter is the iterator of the range, the second is a wrapper which handles the write or split
///operation for the underlying type. for a vector it would be VectorExpression.
template<class Iterator,class Wrapper>
class InitializerRange:public InitializerBase<InitializerRange<Iterator,Wrapper> >{
private:
	Iterator m_begin;
	Iterator m_end;
	//if Iterator is a const_iterator of a container, we must get the const value type
	//the only way to get the reference, which is in the case of const iterator
	//const value_type& and remove the reference part
	//this is done using remove_reference
	//typedef typename boost::remove_reference<typename Iterator::reference>::type value_type;
public:
	InitializerRange(Iterator begin,Iterator end):m_begin(begin),m_end(end){}
	template<class Iter>
	void init(Iter& pos)const{
		for(Iterator elem = m_begin; elem!= m_end; ++elem){
			Wrapper writer(*elem);
			writer.init(pos);
		}
	}
	template<class Iter>
	void split(Iter& pos){
		for(Iterator elem= m_begin; elem!= m_end; ++elem){
			Wrapper vectorWriter(*elem);
			vectorWriter.split(pos);
		}
	}
	std::size_t size()const{
		std::size_t elements = 0;
		for(Iterator elem = m_begin; elem != m_end; ++elem){
			Wrapper vectorWriter(*elem);
			elements += vectorWriter.size();
		}
		return elements;
	}
};

//The following types are not used as right hand side argument in the expression, so they don't need the base class

///\brief  Marks the end of the initializer list template.
struct InitializerEnd{
	template<class Iterator>
	void init(Iterator& pos)const{}
	
	template<class Iterator>
	void split(Iterator& pos){}
		
	std::size_t size()const{
		return 0;
	}
};

///\brief The InitializerNode can initialize a vector on the basis of multiple vector or scalar expressions.
///
///It can be used to construct a big vector out of several small ones or even single values .
///Alternatively, it can split a big vector in several smaller ones.
///VectorInitializer< VectorInitializer< InitializerEnd, A >, B > represents the vector (A,B)
template<class Parent, class Expression>
class InitializerNode{
private:
	Parent m_parent;
	Expression m_expression;
public:
	InitializerNode(const Parent& parent, Expression expression)
	:m_parent(parent),m_expression(expression){}
	
	///initializes the range [pos,pos+size()] with the vector (Parent,Expression)
	template<class Iterator>
	void init(Iterator& pos)const{
		m_parent.init(pos);
		m_expression.init(pos);
	}
	
	///writes the contents of the iterator into the stored expression so that is split into (Parent,Expression)
	template<class Iterator>
	void split(Iterator& pos){
		m_parent.split(pos);
		m_expression.split(pos);
	}
	///this is used only while debugging the program to ensure, that there occurs no size mismatch during initialization.
	std::size_t size()const{
		return m_parent.size()+m_expression.size();
	}
};

//==================CREATION OF BIG VECTORS FROM SMALL ONES============================


///\brief VectorInitializer takes a Vector and an initialization expression to initialize the vector during destruction. 
///
///Since we later recursively define the initialization sequence, the initialization can be disabled.
template<class VectorExpression,class InitExpression>
class VectorInitializer{
private:
	InitExpression m_expression;///<expression which initializes the vector
	mutable bool m_init;///<disables the initialization on destruction when set to false
public:
	//can't be private
	VectorExpression m_vector;

	VectorInitializer(VectorExpression vector,const InitExpression& expression)
		:m_expression(expression), m_vector(vector) {
		m_init = true;
	}
	///if another initializer is constructed on the basis of this, than we disable it to
	///prevent double initialization
	void disable()const{
		m_init=false;
	}
	///returns the internal initialization expression
	const InitExpression& expression()const{
		return m_expression;
	}
	///the destructor initializes the vector
	~VectorInitializer(){
		if(m_init){
			SIZE_CHECK(m_vector.size() == m_expression.size());
			typename boost::remove_reference<VectorExpression>::type::iterator iter = m_vector.begin();
			m_expression.init(iter);
		}
	}
};
	

//Operator<< to begin the expression vec<<a
#define SHARK_INIT_INIT(Type,Argument)\
template<class Sink,class Source>\
VectorInitializer<Sink,InitializerNode<InitializerEnd, Type> > operator<<(const ADLVector<Sink>& sink,const Argument& source){\
	typedef InitializerNode<InitializerEnd,Type > Init;\
	return VectorInitializer<Sink, Init>(sink.vector,Init(InitializerEnd(),source()));\
}
///\brief Begins the initialization argument with a vector as first right hand side argument.
SHARK_INIT_INIT(VectorExpression<const Source&>,shark::blas::vector_expression<Source>)
///\brief Begins the initialization argument with a arbitrary source as first right hand side argument.
SHARK_INIT_INIT(Source,InitializerBase<Source>)
#undef SHARK_INIT_INIT

///\brief Begins the initialization sequence with a single scalar value.
template<class Sink>
VectorInitializer<Sink,InitializerNode<InitializerEnd,Scalar<const typename Sink::value_type> > >
operator<<(const ADLVector<Sink>& sink,const typename Sink::value_type& value){
	typedef InitializerNode<InitializerEnd,Scalar<const typename Sink::value_type> > Init;
	return VectorInitializer<Sink,Init>(sink.vector,Init(InitializerEnd(),value));
}
template<class Sink>
VectorInitializer<Sink&,InitializerNode<InitializerEnd,Scalar<const typename Sink::value_type> > >
operator<<(const ADLVector<Sink&>& sink,const typename Sink::value_type& value){
	typedef InitializerNode<InitializerEnd,Scalar<const typename Sink::value_type> > Init;
	return VectorInitializer<Sink&,Init>(sink.vector,Init(InitializerEnd(),value));
}
//operator, to complete it: vec<<a,b,c;
//every operator comma disables the old VectoInitializer to prevent that objects get initialized twice.

#define SHARK_INIT_COMMA(Type, Argument)\
template<class Sink,class Init,class Source>\
VectorInitializer<Sink,InitializerNode<Init,Type > > operator,(const VectorInitializer<Sink,Init >& init,const Argument& vec){\
	init.disable();\
	typedef InitializerNode<Init,Type > newExpression;\
	return VectorInitializer<Sink, newExpression>(init.m_vector,newExpression(init.expression(),vec()));\
}
///\brief Appends a single vector expression c to the expression vec<<a,b -> vec<<a,b,c.
SHARK_INIT_COMMA(VectorExpression<const Source&>,shark::blas::vector_expression<Source>)
///\brief Appends a initialization expression c to the expression vec<<a,b -> vec<<a,b,c.
SHARK_INIT_COMMA(Source,InitializerBase<Source>)
#undef SHARK_INIT_COMMA

///\brief Operator comma concatenates more vectors and values for initialization.
///
///Special case forsingle values.
template<class Sink,class Init>
VectorInitializer<Sink,InitializerNode<Init,Scalar<const typename Sink::value_type> > > 
operator,(const VectorInitializer<Sink,Init >& init,const typename Sink::value_type& value){
	init.disable();//we don't need it anymore for initialization
	typedef InitializerNode<Init, Scalar<const typename Sink::value_type> > newExpression;
	return VectorInitializer<Sink, newExpression>(init.m_vector,newExpression(init.expression(),value));
}
template<class Sink,class Init>
VectorInitializer<Sink&,InitializerNode<Init,Scalar<const typename Sink::value_type> > > 
operator,(const VectorInitializer<Sink&,Init >& init,const typename Sink::value_type& value){
	init.disable();//we don't need it anymore for initialization
	typedef InitializerNode<Init, Scalar<const typename Sink::value_type> > newExpression;
	return VectorInitializer<Sink&, newExpression>(init.m_vector,newExpression(init.expression(),value));
}
//=====================SPLITTING OF A VECTOR IN SEVERAL SMALLER ONES===========================

//the same stuff again, only for splitting...

///\brief VectorSplitter takes a Vector and a mutable initialization expression to split the vector during destruction. 
///
///Since we later recursively define the initialization sequence, the splitting can be disabled.
template<class VectorExpression,class SplittingExpression>
class VectorSplitter{
private:
	SplittingExpression m_expression;///<expression which splits the vector
	mutable bool m_init;///<disables the splitting on destruction when set to false
public:
	//can't be private
	VectorExpression m_vector;

	VectorSplitter(VectorExpression vector,const SplittingExpression& expression)
		:m_expression(expression), m_vector(vector){
		m_init = true;
	}
	///if another initializer is constructed on the basis of this, than we disable it to
	///prevent double initialization
	void disable()const{
		m_init=false;
	}
	///returns the internal initialization expression
	const SplittingExpression& expression()const{
		return m_expression;
	}
	///the destructor initializes the vector
	~VectorSplitter(){
		if(m_init){
			SIZE_CHECK(m_vector.size() == m_expression.size());
			typename boost::remove_reference<VectorExpression>::type::const_iterator iter = m_vector.begin();
			m_expression.split(iter);
		}
	}
};

//operator <<
#define SHARK_SPLIT_INIT(Type,Argument)\
template<class Source,class Sink>\
VectorSplitter<Source,InitializerNode<InitializerEnd, Type> >\
operator>>(const ADLVector<Source>& source,Argument& sink){\
	typedef InitializerNode<InitializerEnd,Type > Init;\
	return VectorSplitter<Source, Init>(source.vector,Init(InitializerEnd(),sink()));\
}
///\brief Appends a single mutable vector expression.
SHARK_SPLIT_INIT(VectorExpression<Sink&>,shark::blas::vector_expression<Sink>)
///\brief Appends an arbitrary source.
SHARK_SPLIT_INIT(Sink,const InitializerBase<Sink>)
#undef SHARK_SPLIT_INIT

///\brief Appends a single variable.
template<class Sink>
VectorSplitter<Sink,InitializerNode<InitializerEnd,Scalar<typename Sink::value_type> > >
operator>>(const ADLVector<Sink>& sink,typename Sink::value_type& value){
	typedef InitializerNode<InitializerEnd,Scalar<typename Sink::value_type> > Init;
	return VectorSplitter<Sink,Init>(sink.vector,Init(InitializerEnd(),value));
}
template<class Sink>
VectorSplitter<Sink&,InitializerNode<InitializerEnd,Scalar<typename Sink::value_type> > >
operator>>(const ADLVector<Sink&>& sink,typename Sink::value_type& value){
	typedef InitializerNode<InitializerEnd,Scalar<typename Sink::value_type> > Init;
	return VectorSplitter<Sink&,Init>(sink.vector,Init(InitializerEnd(),value));
}
//version for vector proxies...
#define SHARK_SPLIT_PROXY_INIT(Argument)\
template<class Source,class Sink>\
VectorSplitter<Source,InitializerNode<InitializerEnd, VectorExpression<Argument> > >\
operator>>(const ADLVector<Source>& source,Argument sink){\
	typedef InitializerNode<InitializerEnd,VectorExpression<Argument> > Init;\
	return VectorSplitter<Source, Init>(source.vector,Init(InitializerEnd(),sink));\
}
///\brief Appends a single mutable vector expression.
SHARK_SPLIT_PROXY_INIT(shark::blas::vector_range<Sink>)
///\brief Appends a matrix row.
SHARK_SPLIT_PROXY_INIT(shark::blas::matrix_row<Sink>)
///\brief Appends a matrix column.
SHARK_SPLIT_PROXY_INIT(shark::blas::matrix_column<Sink>)
#undef SHARK_SPLIT_PROXY_INIT


//operator,
///\brief Appends a single vector expression.
template<class Source,class Init,class Sink>
VectorSplitter<Source,InitializerNode<Init,VectorExpression<Sink&> > >
operator,(const VectorSplitter<Source,Init >& source, shark::blas::vector_expression<Sink>& vec){
	source.disable();
	typedef InitializerNode<Init,VectorExpression<Sink&> > newExpression;
	return VectorSplitter<Source, newExpression>(source.m_vector,newExpression(source.expression(),vec()));
}
///\brief Appends a initialization expression.
template<class Source,class Init,class Sink>
VectorSplitter<Source,InitializerNode<Init,Sink > >
operator,(const VectorSplitter<Source,Init >& source,const InitializerBase<Sink>& vec){
	source.disable();
	typedef InitializerNode<Init,Sink > newExpression;
	return VectorSplitter<Source, newExpression>(source.m_vector,newExpression(source.expression(),vec()));
}

///\brief Operator comma adds another value for splitting.
template<class Source,class Init>
VectorSplitter<Source,InitializerNode<Init,Scalar<typename Source::value_type> > > 
operator,(const VectorSplitter<Source,Init >& source,typename Source::value_type& value){
	source.disable();//we don't need it anymore for initialization
	typedef InitializerNode<Init, Scalar<typename Source::value_type> > newExpression;
	return VectorSplitter<Source, newExpression>(source.m_vector,newExpression(source.expression(),value));
}
template<class Source,class Init>
VectorSplitter<Source&,InitializerNode<Init,Scalar<typename Source::value_type> > > 
operator,(const VectorSplitter<Source&,Init >& source,typename Source::value_type& value){
	source.disable();//we don't need it anymore for initialization
	typedef InitializerNode<Init, Scalar<typename Source::value_type> > newExpression;
	return VectorSplitter<Source&, newExpression>(source.m_vector,newExpression(source.expression(),value));
}
#define SHARK_SPLIT_PROXY_COMMA(Argument)\
template<class Source,class Init,class Sink>\
VectorSplitter<Source,InitializerNode<Init,VectorExpression<Argument> > >\
operator,(const VectorSplitter<Source,Init >& source, Argument vec){\
	source.disable();\
	typedef InitializerNode<Init,VectorExpression<Argument> > newExpression;\
	return VectorSplitter<Source, newExpression>(source.m_vector,newExpression(source.expression(),vec));\
}
SHARK_SPLIT_PROXY_COMMA(shark::blas::vector_range<Sink>)
SHARK_SPLIT_PROXY_COMMA(shark::blas::matrix_row<Sink>)
SHARK_SPLIT_PROXY_COMMA(shark::blas::matrix_column<Sink>)
#undef SHARK_SPLIT_PROXY_COMMA

}}}

#endif
