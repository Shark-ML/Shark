//===========================================================================
/*!
 *  \file config.h
 *
 *  \author  T. Glasmachers
 *  \date    2008
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-27974<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
//===========================================================================


#ifndef _config_H_
#define _config_H_


//
// easy to initialize and easy to use
// data structure for hierarchical
// configuration information with items
// of various data types.
//


#include <SharkDefs.h>
#include <vector>

class Configuration;


// #define SHARKEXCEPTION(a) a


class PropertyNode
{
public:
	enum eType
	{
		etBool,				// data only, must not have children
		etInt,				// data only, must not have children
		etDouble,			// data only, must not have children
		etString,			// data only, must not have children
		etSelectOne,		// value is the name of a child node, can be used as an enumeration
		etBranch,			// no value, just provides children
		etConst,			// no value, no children, can be used as an enumeration constant
	};

	union uValue
	{
		bool b;				// etBool
		int i;				// etInt
		double d;			// etDouble
		char* s;			// etString
		int e;				// etSelectOne (enumeration index)
	};


	PropertyNode(PropertyNode* parent, const char* name, eType type);
	~PropertyNode();


	inline const char* getName() const
	{
		return name;
	}

	// parent child access
	inline PropertyNode* getParent()
	{
		return parent;
	}

	inline const PropertyNode* getParent() const
	{
		return parent;
	}

	inline bool hasChildren() const
	{
		return (child.size() > 0);
	}

	inline int size() const
	{
		return child.size();
	}

	PropertyNode& operator [] (int index);
	const PropertyNode& operator [] (int index) const;
	PropertyNode& operator [] (const char* name);
	const PropertyNode& operator [] (const char* name) const;

	// data access
	inline eType getType() const
	{
		return type;
	}

	inline bool getBool() const
	{
		if (type != etBool) throw SHARKEXCEPTION("[PropertyNode::getBool] type mismatch");
		return value.b;
	}

	inline int getInt() const
	{
		if (type != etInt) throw SHARKEXCEPTION("[PropertyNode::getInt] type mismatch");
		return value.i;
	}

	inline double getDouble() const
	{
		if (type != etDouble) throw SHARKEXCEPTION("[PropertyNode::getDouble] type mismatch");
		return value.d;
	}

	inline const char* getString() const
	{
		if (type != etString) throw SHARKEXCEPTION("[PropertyNode::getString] type mismatch");
		return value.s;
	}

	inline int getIntMin() const
	{
		if (type != etInt) throw SHARKEXCEPTION("[PropertyNode::getIntMin] type mismatch");
		return minimum.i;
	}

	inline int getIntMax() const
	{
		if (type != etInt) throw SHARKEXCEPTION("[PropertyNode::getIntMax] type mismatch");
		return maximum.i;
	}

	inline double getDoubleMin() const
	{
		if (type != etDouble) throw SHARKEXCEPTION("[PropertyNode::getDoubleMin] type mismatch");
		return minimum.d;
	}

	inline double getDoubleMax() const
	{
		if (type != etDouble) throw SHARKEXCEPTION("[PropertyNode::getDoubleMax] type mismatch");
		return maximum.d;
	}

	inline bool isLogarithmic() const
	{
		if (type != etDouble) throw SHARKEXCEPTION("[PropertyNode::isLogarithmic] type mismatch");
		return logarithmic;
	}

	const char* getSelected() const;
	int getSelectedIndex() const;
	const PropertyNode& getSelectedNode() const;
	PropertyNode& getSelectedNode();

	void set(bool value);
	void set(int value);
	void set(double value);
	void set(const char* value);
	void set(const PropertyNode& child);
	void setRange(int minimum, int maximum);
	void setRange(double minimum, double maximum, bool logarithmic = false);

	void description(char* buffer, int indent = 0) const;

	friend class Configuration;

protected:
	PropertyNode(PropertyNode* parent = NULL);
	PropertyNode(const PropertyNode& other, PropertyNode* parent = NULL);

	char* name;

	// parent child relation
	PropertyNode* parent;
	std::vector<PropertyNode*> child;

	// data
	eType type;
	uValue value;
	uValue minimum;
	uValue maximum;
	bool logarithmic;
};


//
// The structure argument defines the configuration tree.
// This string has a strictly defined syntax.
//
// A node takes the form
//   { <name> <type> [<default> [<min> <max> [log]]] <children> }
// where arbitrary whitespace is used as a separator.
//
// Here, <name> is a token which must be unique within
// its siblings. <type> is one of the constants
//   bool, int, double, string, select, branch, const.
// The first five types take default values, where
// the constants true and false are the only possible
// bool values and string constants are within single
// quotes. The select type takes an integer value as a
// zero-based index into its childen.
// Int and double values require a range after the value
// argument, and double values can get the additional
// attirubte 'log' for logarithmic scale.
// The types select and branch can come with any number
// of child nodes, which follow the syntax of ordinary
// nodes.
//
class Configuration : public PropertyNode
{
public:
	Configuration(const char* structure);
	Configuration(const Configuration& other);
	~Configuration();

	inline const char* getStructure() const
	{
		return structure;
	}

protected:
	bool GetToken(char* structure, int& line, int& pos);
	bool SkipWhitespace(char* structure, int& pos, int& line, int len);
	bool RecBuildStructure(PropertyNode* property, char* structure, int& pos, int& line);

	const char* structure;
	static char exception[256];
};


#endif
