//===========================================================================
/*!
 * 
 *
 * \brief       Json class used internally by the OpenML module.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2016
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

#ifndef SHARK_OPENML_DETAIL_JSON_H
#define SHARK_OPENML_DETAIL_JSON_H


#include <shark/Core/Exception.h>

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <istream>
#include <ostream>
#include <iostream>
#include <iomanip>
#include <cmath>


namespace shark {
namespace openML {
namespace detail {


// minimal template meta programming stuff
struct _true { };
struct _false { };


// This type is the Variant's default argument type.
struct NoArgument2 { };
struct NoArgument3 { };
struct NoArgument4 { };
struct NoArgument5 { };
struct NoArgument6 { };
struct NoArgument7 { };

template <typename T>
struct ArgumentTraits
{
	typedef _true type;
	static const size_t num = 1;
};

template <> struct ArgumentTraits<NoArgument2> { typedef _false type; static const size_t num = 0; };
template <> struct ArgumentTraits<NoArgument3> { typedef _false type; static const size_t num = 0; };
template <> struct ArgumentTraits<NoArgument4> { typedef _false type; static const size_t num = 0; };
template <> struct ArgumentTraits<NoArgument5> { typedef _false type; static const size_t num = 0; };
template <> struct ArgumentTraits<NoArgument6> { typedef _false type; static const size_t num = 0; };
template <> struct ArgumentTraits<NoArgument7> { typedef _false type; static const size_t num = 0; };

// This type acts as the end iterator of any sequence.
struct EmptySequence
{
	static const size_t size = 0;
	static const size_t storage_size = 0;
};

// A sequence represents several things, namely:
// (*) A collection of types.
// (*) An iterator into a sequence of types.
// The first argument is a plain type to be stored.
// In contrast, the second argument is in itself a sequence!
template <typename FIRST, typename REST>
struct Sequence
{
	typedef FIRST first_type;
	typedef REST rest_type;

	static const size_t size = REST::size + ArgumentTraits<FIRST>::num;
	static const size_t storage_size = (sizeof(FIRST) >= REST::storage_size) ? sizeof(FIRST) : REST::storage_size;
};

// move forward in sequence
template <typename SEQUENCE, int INDEX>
struct forward
{
	typedef typename forward<typename SEQUENCE::rest_type, INDEX - 1>::type type;
};

// move forward in sequence
template <typename SEQUENCE>
struct forward<SEQUENCE, 0>
{
	typedef SEQUENCE type;
};

// access type in sequence
template <typename SEQUENCE, int INDEX>
struct at
{
	typedef typename forward<SEQUENCE, INDEX>::type ElemType;
	typedef typename ElemType::first_type type;
};

// check whether an element is contained in the sequence

// return index of element in sequence
template <typename ELEMENT, typename SEQUENCE>
struct index_of
{
	static const int index = index_of<ELEMENT, typename SEQUENCE::rest_type>::index + 1;
};

template <typename ELEMENT, typename REST>
struct index_of<ELEMENT, Sequence<ELEMENT, REST> >
{
	static const int index = 0;
};


#define VARIANT_PER_TYPE(N, T) \
public: \
	Variant(T const& value) \
	: m_type(N) \
	{ \
		new (m_storage) T(value); \
	} \
\
	T const& operator = (T const& other) \
	{ \
		if ((void*)m_storage == (void*)(&other)) return other; \
		if (m_type == N) get< N >() = other; \
		else \
		{ \
			invalidate(); \
			new (m_storage) T(other); \
			m_type = N; \
		} \
		return other; \
	}




template<
	typename T0,
	typename T1,
	typename T2 = detail::NoArgument2,
	typename T3 = detail::NoArgument3,
	typename T4 = detail::NoArgument4,
	typename T5 = detail::NoArgument5,
	typename T6 = detail::NoArgument6,
	typename T7 = detail::NoArgument7
>
class Variant
{
private:
	typedef
			detail::Sequence<T0,
				detail::Sequence<T1,
					detail::Sequence<T2,
						detail::Sequence<T3,
							detail::Sequence<T4,
								detail::Sequence<T5,
									detail::Sequence<T6,
										detail::Sequence<T7,
											detail::EmptySequence
										>
									>
								>
							>
						>
					>
				>
			>
		SequenceType;

	char m_storage[SequenceType::storage_size];
	int m_type;

	template <int INDEX>
	typename detail::at<SequenceType, INDEX>::type* ptr()
	{
		if (m_type != INDEX) throw SHARKEXCEPTION("[variant] attempt to access data as wrong type"); \
		typedef typename detail::at<SequenceType, INDEX>::type ReturnType;
		return ((ReturnType*)(void*)(&m_storage));
	}

	template <int INDEX>
	const typename detail::at<SequenceType, INDEX>::type* const_ptr() const
	{
		if (m_type != INDEX) throw SHARKEXCEPTION("[variant] attempt to access data as wrong type"); \
		typedef typename detail::at<SequenceType, INDEX>::type ReturnType;
		return ((const ReturnType*)(void*)(&m_storage));
	}

	// in-place destruct object - afterwards state is undefined
	void invalidate()
	{
		if (SequenceType::size <= 4 || m_type < 4)
		{
			if (SequenceType::size <= 2 || m_type < 2)
			{
				if (SequenceType::size <= 1 || m_type < 1) ptr<0>()->~T0();
				else ptr<1>()->~T1();
			}
			else
			{
				if (SequenceType::size <= 3 || m_type < 3) ptr<2>()->~T2();
				else ptr<3>()->~T3();
			}
		}
		else
		{
			if (SequenceType::size <= 6 || m_type < 6)
			{
				if (SequenceType::size <= 5 || m_type < 5) ptr<4>()->~T4();
				else ptr<5>()->~T5();
			}
			else
			{
				if (SequenceType::size <= 7 || m_type < 7) ptr<6>()->~T6();
				else ptr<7>()->~T7();
			}
		}
	}

	// in-place construct - assuming that the state was undefined before
	void create(Variant const& other)
	{
		if (SequenceType::size <= 4 || m_type < 4)
		{
			if (SequenceType::size <= 2 || m_type < 2)
			{
				if (SequenceType::size <= 1 || m_type < 1) new (m_storage) T0(other.get<0>());
				else new (m_storage) T1(other.get<1>());
			}
			else
			{
				if (SequenceType::size <= 3 || m_type < 3) new (m_storage) T2(other.get<2>());
				else new (m_storage) T3(other.get<3>());
			}
		}
		else
		{
			if (SequenceType::size <= 6 || m_type < 6)
			{
				if (SequenceType::size <= 5 || m_type < 5) new (m_storage) T4(other.get<4>());
				else new (m_storage) T5(other.get<5>());
			}
			else
			{
				if (SequenceType::size <= 7 || m_type < 7) new (m_storage) T6(other.get<6>());
				else new (m_storage) T7(other.get<7>());
			}
		}
	}

public:
	// (zero-based) index of the current type
	// or -1 in case of undefined state
	int type() const
	{ return m_type; }

	// data access by index
	template <int INDEX>
	typename detail::at<SequenceType, INDEX>::type& get()
	{ return *(ptr<INDEX>()); }

	template <int INDEX>
	typename detail::at<SequenceType, INDEX>::type const& get() const
	{ return *(const_ptr<INDEX>()); }

	// data access by type
	template <typename T>
	typename detail::at<SequenceType, detail::index_of<T, SequenceType>::index>::type& as()
	{ return *(ptr<detail::index_of<T, SequenceType>::index>()); }

	template <typename T>
	typename detail::at<SequenceType, detail::index_of<T, SequenceType>::index>::type const& as() const
	{ return *(const_ptr<detail::index_of<T, SequenceType>::index>()); }

	Variant()
	: m_type(0)
	{
		new (m_storage) T0();
	}

	~Variant()
	{ invalidate(); }

	Variant(Variant const& other)
	: m_type(other.m_type)
	{
		create(other);
	}

	Variant& operator = (Variant& other)
	{
		if (this == &other) return other;
		if (m_type == other.m_type)
		{
			if (SequenceType::size <= 4 || m_type < 4)
			{
				if (SequenceType::size <= 2 || m_type < 2)
				{
					if (SequenceType::size <= 1 || m_type < 1) get<0>() = other.get<0>();
					else get<1>() = other.get<1>();
				}
				else
				{
					if (SequenceType::size <= 3 || m_type < 3) get<2>() = other.get<2>();
					else get<3>() = other.get<3>();
				}
			}
			else
			{
				if (SequenceType::size <= 6 || m_type < 6)
				{
					if (SequenceType::size <= 5 || m_type < 5) get<4>() = other.get<4>();
					else get<5>() = other.get<5>();
				}
				else
				{
					if (SequenceType::size <= 7 || m_type < 7) get<6>() = other.get<6>();
					else get<7>() = other.get<7>();
				}
			}
		}
		else
		{
			invalidate();
			m_type = other.m_type;
			create(other);
		}
		return other;
	}

	VARIANT_PER_TYPE(0, T0)
	VARIANT_PER_TYPE(1, T1)
	VARIANT_PER_TYPE(2, T2)
	VARIANT_PER_TYPE(3, T3)
	VARIANT_PER_TYPE(4, T4)
	VARIANT_PER_TYPE(5, T5)
	VARIANT_PER_TYPE(6, T6)
	VARIANT_PER_TYPE(7, T7)
};


////////////////////////////////////////////////////////////


// tags for convenient construction of null objects and empty containers
enum ConstructionTypename
{ null, object, array };

// for convenient construction from a string representation
enum ConstructionParse
{ parse };


/// \brief Json class used internally by the OpenML wrapper.
class Json
{
protected:
	friend class std::map<std::string, Json>;  // needs access to the default constructor

	struct Undefined { };
	struct Null { };
	typedef std::map<std::string, Json> Object;
	typedef std::vector<Json> Array;

	enum Type
	{
		type_undefined = 0,
		type_null = 1,
		type_boolean = 2,
		type_number = 3,
		type_string = 4,
		type_object = 5,
		type_array = 6,
	};

	typedef Variant<Undefined, Null, bool, double, std::string, Object, Array> Data;
	typedef std::shared_ptr<Data> DataPtr;

public:
	// iterators
	typedef Object::iterator          object_iterator;
	typedef Object::const_iterator    const_object_iterator;
	typedef Array::iterator           array_iterator;
	typedef Array::const_iterator     const_array_iterator;

	// construction
	Json();                     // create an "undefined" or "invalid" json object
	Json(bool value);
	Json(double value);
	Json(const char* value);
	Json(std::string const& value);
	Json(Json const& other);
	Json(ConstructionTypename tn);
	Json(ConstructionParse p, std::string s);
	Json(std::istream& str);
	Json(std::vector<bool> const& arr);
	Json(std::vector<double> const& arr);
	Json(std::vector<std::string> const& arr);
	Json(std::map<std::string, bool> const& obj);
	Json(std::map<std::string, double> const& obj);
	Json(std::map<std::string, std::string> const& obj);

	template <class ITER>
	Json(ITER begin, ITER end)
	{ parseJson(begin, end, begin); }

	// type information
	inline bool isUndefined() const
	{ return (data().type() == type_undefined); }
	inline bool isNull() const
	{ return (data().type() == type_null); }
	inline bool isBoolean() const
	{ return (data().type() == type_boolean); }
	inline bool isNumber() const
	{ return (data().type() == type_number); }
	inline bool isString() const
	{ return (data().type() == type_string); }
	inline bool isObject() const
	{ return (data().type() == type_object); }
	inline bool isArray() const
	{ return (data().type() == type_array); }

	// type asserts
	inline void assertValid() const
	{ if (! isValid()) throw SHARKEXCEPTION("Json value is undefined"); }
	inline void assertNull() const
	{ if (! isNull()) throw SHARKEXCEPTION("Json value is not null"); }
	inline void assertBoolean() const
	{ if (! isBoolean()) throw SHARKEXCEPTION("Json value is not boolean"); }
	inline void assertNumber() const
	{ if (! isNumber()) throw SHARKEXCEPTION("Json value is not a number"); }
	inline void assertString() const
	{ if (! isString()) throw SHARKEXCEPTION("Json value is not a string"); }
	inline void assertObject() const
	{ if (! isObject()) throw SHARKEXCEPTION("Json value is not an object"); }
	inline void assertArray() const
	{ if (! isArray()) throw SHARKEXCEPTION("Json value is not an array"); }
	inline void assertArray(std::size_t index) const
	{
		assertArray();
#ifdef DEBUG
		if (index >= m_ptr->as<Array>().size()) throw SHARKEXCEPTION("Json array index out of bounds");
#endif
	}

	// elementary type reference access
	inline bool asBoolean() const
	{ assertBoolean(); return data().as<bool>(); }
	inline int asInteger() const
	{ assertNumber(); return (int)(data().as<double>()); }
	inline double asNumber() const
	{ assertNumber(); return data().as<double>(); }
	inline std::string const& asString() const
	{ assertString(); return data().as<std::string>(); }

	// elementary type read access
	inline operator bool() const
	{ return asBoolean(); }
	inline operator int() const
	{ return asInteger(); }
	inline operator double() const
	{ return asNumber(); }
	inline operator std::string() const
	{ return asString(); }

	// elementary type read access with default value
	inline bool operator () (bool defaultvalue)
	{ if (isValid()) return asBoolean(); else return defaultvalue; }
	inline double operator () (double defaultvalue)
	{ if (isValid()) return asNumber(); else return defaultvalue; }
	inline std::string operator () (const char* defaultvalue)
	{ if (isValid()) return asString(); else return defaultvalue; }
	inline std::string operator () (std::string const& defaultvalue)
	{ if (isValid()) return asString(); else return defaultvalue; }

	// container (array or object) size
	inline std::size_t size() const
	{
		if (isObject()) return data().as<Object>().size();
		else if (isArray()) return data().as<Array>().size();
		else throw SHARKEXCEPTION("Json value is not a container (object or array)");
	}

	// object member test
	inline bool has(const char* key) const
	{ return has(std::string(key)); }
	inline bool has(std::string const& key) const
	{ assertObject(); return (data().as<Object>().find(key) != object_end()); }

	// container operator access
	inline Json& operator [] (const char* key)
	{ return operator [] (std::string(key)); }
	inline Json& operator [] (std::string const& key)
	{ assertObject(); return data().as<Object>()[key]; }
	inline Json const& operator [] (const char* key) const
	{ return operator [] (std::string(key)); }
	inline Json const& operator [] (std::string const& key) const
	{ assertObject(); if (! has(key)) return m_undefined; else return (const_cast<Object&>(data().as<Object>()))[key]; }
	inline Json& operator [] (int index)
	{ assertArray(index); return data().as<Array>()[index]; }
	inline Json const& operator [] (int index) const
	{ assertArray(index); return data().as<Array>()[index]; }

	// container iterator access
	inline object_iterator object_begin()
	{ assertObject(); return data().as<Object>().begin(); }
	inline const_object_iterator object_begin() const
	{ assertObject(); return data().as<Object>().begin(); }
	inline object_iterator object_end()
	{ assertObject(); return data().as<Object>().end(); }
	inline const_object_iterator object_end() const
	{ assertObject(); return data().as<Object>().end(); }
	inline array_iterator array_begin()
	{ assertArray(); return data().as<Array>().begin(); }
	inline const_array_iterator array_begin() const
	{ assertArray(); return data().as<Array>().begin(); }
	inline array_iterator array_end()
	{ assertArray(); return data().as<Array>().end(); }
	inline const_array_iterator array_end() const
	{ assertArray(); return data().as<Array>().end(); }

	// pure container access
	std::vector<bool> asBooleanArray();
	std::vector<double> asNumberArray();
	std::vector<std::string> asStringArray();
	std::map<std::string, bool> asBooleanObject();
	std::map<std::string, double> asNumberObject();
	std::map<std::string, std::string> asStringObject();

	// comparison
	bool operator == (Json const& other) const;
	inline bool operator == (bool other) const
	{ return (isBoolean() && asBoolean() == other); }
	inline bool operator == (double other) const
	{ return (isNumber() && asNumber() == other); }
	inline bool operator == (const char* other) const
	{ return (isString() && asString() == other); }
	inline bool operator == (std::string const& other) const
	{ return (isString() && asString() == other); }
	inline bool operator != (Json const& other) const
	{ return ! (operator == (other)); }
	inline bool operator != (bool other) const
	{ return ! (operator == (other)); }
	inline bool operator != (double other) const
	{ return ! (operator == (other)); }
	inline bool operator != (const char* other) const
	{ return ! (operator == (other)); }
	inline bool operator != (std::string const& other) const
	{ return ! (operator == (other)); }

	// value assignment and container preparation (write access)
	inline Json& operator = (bool value)
	{ m_ptr = std::make_shared<Data>(value); return *this; }
	inline Json& operator = (double value)
	{ m_ptr = std::make_shared<Data>(value); return *this; }
	inline Json& operator = (const char* value)
	{ m_ptr = std::make_shared<Data>(std::string(value)); return *this; }
	inline Json& operator = (std::string const& value)
	{ m_ptr = std::make_shared<Data>(value); return *this; }
	inline Json& operator = (std::vector<bool> const& arr)
	{
		m_ptr = std::make_shared<Data>(Array(arr.size(), Json(false)));
		for (std::size_t i=0; i<arr.size(); i++) m_ptr->as<Array>()[i] = Json(arr[i]);
		return *this;
	}
	inline Json& operator = (std::vector<double> const& arr)
	{
		m_ptr = std::make_shared<Data>(Array(arr.size(), Json(0.0)));
		for (std::size_t i=0; i<arr.size(); i++) m_ptr->as<Array>()[i] = Json(arr[i]);
		return *this;
	}
	inline Json& operator = (std::vector<std::string> const& arr)
	{
		m_ptr = std::make_shared<Data>(Array(arr.size(), Json(std::string())));
		for (std::size_t i=0; i<arr.size(); i++) m_ptr->as<Array>()[i] = Json(arr[i]);
		return *this;
	}
	inline Json& operator = (std::map<std::string, bool> const& obj)
	{
		m_ptr = std::make_shared<Data>(Object());
		for (std::map<std::string, bool>::const_iterator it=obj.begin(); it != obj.end(); ++it)
		{
			m_ptr->as<Object>()[it->first] = Json(it->second);
		}
		return *this;
	}
	inline Json& operator = (std::map<std::string, double> const& obj)
	{
		m_ptr = std::make_shared<Data>(Object());
		for (std::map<std::string, double>::const_iterator it=obj.begin(); it != obj.end(); ++it)
		{
			m_ptr->as<Object>()[it->first] = Json(it->second);
		}
		return *this;
	}
	inline Json& operator = (std::map<std::string, std::string> const& obj)
	{
		m_ptr = std::make_shared<Data>(Object());
		for (std::map<std::string, std::string>::const_iterator it=obj.begin(); it != obj.end(); ++it)
		{
			m_ptr->as<Object>()[it->first] = Json(it->second);
		}
		return *this;
	}
	inline Json& operator = (Json const& value)
	{ m_ptr = value.m_ptr; return *this; }
	inline Json& operator = (ConstructionTypename tn)
	{
		if (tn == null) m_ptr = std::make_shared<Data>(Null());
		else if (tn == object) m_ptr = std::make_shared<Data>(Object());
		else if (tn == array) m_ptr = std::make_shared<Data>(Array());
		else throw SHARKEXCEPTION("json internal error");
		return *this;
	}
	inline void push_back(Json const& value)
	{ assertArray(); data().as<Array>().push_back(value); }
	inline void insert(std::size_t index, Json const& value)
	{
		assertArray();
		if (index > data().as<Array>().size()) throw SHARKEXCEPTION("json array index out of bounds");
		data().as<Array>().insert(array_begin() + index, value);
	}
	inline void erase(std::size_t index)
	{ assertArray(index); data().as<Array>().erase(array_begin() + index); }
	template <class T> Json& operator << (T value)
	{ push_back(value); return *this; }
	inline void erase(std::string const& key)
	{ assertObject(); data().as<Object>().erase(key); }

	// deep copy
	Json clone() const;

	// string <-> object conversion
	inline void parse(std::string const& s)
	{ std::stringstream ss(s); ss >> *this; }
	inline std::string stringify() const
	{ std::stringstream ss; ss << *this; return ss.str(); }
	friend std::istream& operator >> (std::istream& is, Json& json);
	friend std::ostream& operator << (std::ostream& os, Json const& json);

	// file I/O
	bool load(std::string const& filename);
	bool save(std::string const& filename) const;

protected:
	inline bool isValid() const
	{ return (m_ptr->type() != type_undefined); }

#define JSON_INTERNAL_FAIL { std::stringstream ss; ss << "json parse error at position " << position ; throw SHARKEXCEPTION(ss.str()); }
#define JSON_INTERNAL_PEEK(c) { if (iter == end) JSON_INTERNAL_FAIL; c = *iter; }
#define JSON_INTERNAL_READ(c) { JSON_INTERNAL_PEEK(c); ++iter; ++position; }
#define JSON_INTERNAL_SKIP \
	while (true) \
	{ \
		JSON_INTERNAL_PEEK(c); \
		if (isspace(c)) { ++iter; ++position; } \
		else if (c == '/') \
		{ \
			++iter; ++position; \
			JSON_INTERNAL_READ(c); \
			if (c == '/') \
			{ \
				while (c != '\n') { JSON_INTERNAL_READ(c); } \
			} \
			else if (c == '*') \
			{ \
				while (true) \
				{ \
					JSON_INTERNAL_READ(c); \
					if (c == '*') \
					{ \
						JSON_INTERNAL_PEEK(c); \
						if (c == '/') { ++iter; ++position; break; } \
					} \
				} \
			} \
		} \
		else break; \
	}

	template <class ITER>
	static std::string parseString(std::size_t& position, ITER const& end, ITER& iter)
	{
		char c;
		std::string ret;
		while (true)
		{
			JSON_INTERNAL_READ(c)
			if (c == '\"') return ret;
			else if (c == '\\')
			{
				JSON_INTERNAL_READ(c)
				if (c == '\"') ret.push_back('\"');
				else if (c == '\\') ret.push_back('\\');
				else if (c == '/') ret.push_back('/');
				else if (c == 'b') ret.push_back('\b');
				else if (c == 'f') ret.push_back('\f');
				else if (c == 'n') ret.push_back('\n');
				else if (c == 'r') ret.push_back('\r');
				else if (c == 't') ret.push_back('\t');
				else if (c == 'u')
				{
					char buf[5];
					JSON_INTERNAL_READ(buf[0]);
					JSON_INTERNAL_READ(buf[1]);
					JSON_INTERNAL_READ(buf[2]);
					JSON_INTERNAL_READ(buf[3]);
					buf[4] = 0;
					unsigned int value;
					std::stringstream ss;
					ss << std::hex << buf;
					ss >> value;
					// encode as utf-8
					if (value < 0x007f) ret.push_back((char)value);
					else if (value < 0x07ff)
					{
						ret.push_back((char)(192 + (value >> 6)));
						ret.push_back((char)(128 + (value & 63)));
					}
					else
					{
						ret.push_back((char)(224 + (value >> 12)));
						ret.push_back((char)(128 + ((value >> 6) & 63)));
						ret.push_back((char)(128 + (value & 63)));
					}
				}
				else JSON_INTERNAL_FAIL
			}
			else ret.push_back(c);
		}
	}

	template <class ITER>
	void parseJson(std::size_t& position, ITER const& end, ITER& iter)
	{
		char c;
		JSON_INTERNAL_SKIP
		JSON_INTERNAL_READ(c)
		if (c == '{')
		{
			*this = object;
			JSON_INTERNAL_SKIP
			JSON_INTERNAL_PEEK(c)
			if (c != '}')
			{
				do
				{
					JSON_INTERNAL_SKIP
					JSON_INTERNAL_READ(c)
					if (c != '\"') JSON_INTERNAL_FAIL;
					std::string key = parseString(position, end, iter);
					JSON_INTERNAL_SKIP
					JSON_INTERNAL_READ(c)
					if (c != ':') JSON_INTERNAL_FAIL
					Json sub;
					sub.parseJson(position, end, iter);
					(*this)[key] = sub;
					JSON_INTERNAL_SKIP
					JSON_INTERNAL_READ(c)
				}
				while (c == ',');
				if (c != '}') JSON_INTERNAL_FAIL
			}
			else JSON_INTERNAL_READ(c);
		}
		else if (c == '[')
		{
			*this = array;
			JSON_INTERNAL_SKIP
			JSON_INTERNAL_PEEK(c)
			if (c != ']')
			{
				do
				{
					Json sub;
					sub.parseJson(position, end, iter);
					push_back(sub);
					JSON_INTERNAL_SKIP
					JSON_INTERNAL_READ(c)
				}
				while (c == ',');
				if (c != ']') JSON_INTERNAL_FAIL
			}
			else JSON_INTERNAL_READ(c);
		}
		else if (c == '\"')
		{ (*this) = parseString(position, end, iter); }
		else if (c == 'n')
		{
			JSON_INTERNAL_READ(c); if (c != 'u') JSON_INTERNAL_FAIL;
			JSON_INTERNAL_READ(c); if (c != 'l') JSON_INTERNAL_FAIL;
			JSON_INTERNAL_READ(c); if (c != 'l') JSON_INTERNAL_FAIL;
			*this = null;
		}
		else if (c == 't')
		{
			JSON_INTERNAL_READ(c); if (c != 'r') JSON_INTERNAL_FAIL;
			JSON_INTERNAL_READ(c); if (c != 'u') JSON_INTERNAL_FAIL;
			JSON_INTERNAL_READ(c); if (c != 'e') JSON_INTERNAL_FAIL;
			(*this) = true;
		}
		else if (c == 'f')
		{
			JSON_INTERNAL_READ(c); if (c != 'a') JSON_INTERNAL_FAIL;
			JSON_INTERNAL_READ(c); if (c != 'l') JSON_INTERNAL_FAIL;
			JSON_INTERNAL_READ(c); if (c != 's') JSON_INTERNAL_FAIL;
			JSON_INTERNAL_READ(c); if (c != 'e') JSON_INTERNAL_FAIL;
			(*this) = false;
		}
		else
		{
			// parse number
			double value = 0.0;
			bool neg = false;
			if (c == '-') { JSON_INTERNAL_READ(c); neg = true; }
			if (c == '0') { }
			else if (c >= '1' && c <= '9')
			{
				value = (c - '0');
				while (true)
				{
					JSON_INTERNAL_PEEK(c)
					if (c >= '0' && c <= '9') { JSON_INTERNAL_READ(c); value *= 10.0; value += (c - '0'); }
					else break;
				}
			}
			else JSON_INTERNAL_FAIL
			JSON_INTERNAL_PEEK(c)
			if (c == '.')
			{
				JSON_INTERNAL_READ(c)
				JSON_INTERNAL_READ(c)
				if (c < '0' || c > '9') JSON_INTERNAL_FAIL
				double p = 0.1;
				value += p * (c - '0');
				while (true)
				{
					JSON_INTERNAL_PEEK(c)
					if (c >= '0' && c <= '9') { JSON_INTERNAL_READ(c); p *= 0.1; value += p * (c - '0'); }
					else break;
				}
			}
			if (c == 'e' || c == 'E')
			{
				JSON_INTERNAL_READ(c)
				bool eneg = false;
				int e = 0;
				JSON_INTERNAL_PEEK(c)
				if (c == '+') { JSON_INTERNAL_READ(c); JSON_INTERNAL_PEEK(c); }
				if (c == '-') { JSON_INTERNAL_READ(c); JSON_INTERNAL_PEEK(c); eneg = true; }
				if (c < '0' || c > '9') JSON_INTERNAL_FAIL
				while (true)
				{
					JSON_INTERNAL_PEEK(c)
					if (c >= '0' && c <= '9') { JSON_INTERNAL_READ(c); e *= 10; e += (c - '0'); }
					else break;
				}
				if (eneg) e = -e;
				value *= std::pow(10.0, e);
			}
			if (neg) value = -value;
			(*this) = value;
		}
	}

#undef JSON_INTERNAL_FAIL
#undef JSON_INTERNAL_READ
#undef JSON_INTERNAL_PEEK
#undef JSON_INTERNAL_SKIP

	static void outputString(std::ostream& str, std::string const& s);
	void outputJson(std::ostream& str, int depth = -1) const;

	DataPtr m_ptr;
	static Json m_undefined;  // "undefined" reference object, needed by const operator []

	inline Data& data()
	{ return *m_ptr; }
	inline Data const& data() const
	{ return *m_ptr; }
};


inline std::istream& operator >> (std::istream& str, Json& json)
{
	str.unsetf(std::ios::skipws);
	std::istream_iterator<char> iter(str), end;
	std::size_t position = 0;
	json.parseJson(position, end, iter);
	return str;
}

inline std::ostream& operator << (std::ostream& str, Json const& json)
{ json.outputJson(str, 0); return str; }


};  // namespace detail
};  // namespace openML
};  // namespace shark
#endif
