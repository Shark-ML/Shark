//===========================================================================
/*!
 *  \file config.cpp
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


#include <string.h>
#include <stdlib.h>
#include "config.h"


// static
char Configuration::exception[256];


////////////////////////////////////////////////////////////


PropertyNode::PropertyNode(PropertyNode* parent)
{
	this->parent = parent;
	if (parent != NULL)
	{
		if (parent->type != etSelectOne && parent->type != etBranch) throw SHARKEXCEPTION("[PropertyNode::PropertyNode] invalid parent node");
		parent->child.push_back(this);
	}

	this->name = NULL;

	this->type = etConst;
}

// create a deep copy, but with a different parent
PropertyNode::PropertyNode(const PropertyNode& other, PropertyNode* parent)
{
	// create parent child relation
	this->parent = parent;
	if (parent != NULL)
	{
		if (parent->type != etSelectOne && parent->type != etBranch) throw SHARKEXCEPTION("[PropertyNode::PropertyNode] invalid parent node");
		parent->child.push_back(this);
	}

// 	char empty = 0;
// 	printf("\n[PropertyNode copy constructor]\n");
// 	printf("            other [%p]\n", &other);
// 	printf("     other.parent [%p]\n", other.parent);
// 	printf("             this [%p]\n", this);
// 	printf("           parent [%p]\n", parent);
// 	printf("            other: name[%p] = '%s'\n", other.name, (other.name == NULL) ? &empty : other.name);
// 	if (other.parent != NULL)
// 		printf("     other.parent: name[%p] = '%s'\n", other.parent->name, (other.parent->name == NULL) ? &empty : other.parent->name);
// 	if (this->parent != NULL)
// 		printf("           parent: name[%p] = '%s'\n", parent->name, (parent->name == NULL) ? &empty : parent->name);
// 	printf("\n");

	// copy name
	if (other.name == NULL)
	{
		this->name = NULL;
		if (this->parent != NULL) throw SHARKEXCEPTION("[PropertyNode::PropertyNode] internal error");
	}
	else
	{
		int len = strlen(other.name);
		this->name = (char*)malloc(len + 1);
		if (this->name == NULL) throw SHARKEXCEPTION("[PropertyNode::PropertyNode] out of memory");
		memcpy(this->name, other.name, len + 1);
	}

	// copy data
	this->type = other.type;
	this->value = other.value;
	if (type == etString)
	{
		int len = strlen(other.value.s);
		this->value.s = (char*)malloc(len + 1);
		if (this->value.s == NULL) throw SHARKEXCEPTION("[PropertyNode::PropertyNode] out of memory");
		memcpy(this->value.s, other.value.s, len + 1);
	}
	this->minimum = other.minimum;
	this->maximum = other.maximum;
	this->logarithmic = other.logarithmic;

	// copy children
	int i, ic = other.child.size();
	for (i=0; i<ic; i++)
	{
		PropertyNode* c = new PropertyNode(*other.child[i], this);
		if (c == NULL) throw SHARKEXCEPTION("[PropertyNode::PropertyNode] out of memory");
	}
}

PropertyNode::PropertyNode(PropertyNode* parent, const char* name, PropertyNode::eType type)
{
	this->parent = parent;
	if (parent != NULL)
	{
		if (parent->type != etSelectOne && parent->type != etBranch) throw SHARKEXCEPTION("[PropertyNode::PropertyNode] invalid parent node");
		parent->child.push_back(this);
	}

	int len = strlen(name);
	this->name = (char*)malloc(len + 1);
	if (this->name == NULL) throw SHARKEXCEPTION("[PropertyNode::PropertyNode] out of memory");
	memcpy(this->name, name, len + 1);

	this->type = type;
	switch (type)
	{
		case etBool:
			value.b = false;
			break;
		case etInt:
			value.i = 0;
			break;
		case etDouble:
			value.d = 0.0;
			break;
		case etString:
			value.s = (char*)malloc(1);
			if (value.s == NULL) throw SHARKEXCEPTION("[PropertyNode::PropertyNode] out of memory");
			value.s[0] = 0;
			break;
		case etSelectOne:
			value.e = 0;
			break;
		default:
			break;
	}
}

PropertyNode::~PropertyNode()
{
	if (type == etString) free(value.s);

	if (name != NULL) free(name);

	if (parent != NULL)
	{
		int i, ic = parent->child.size();
		for (i=0; i<ic; i++)
		{
			if (parent->child[i] == this)
			{
				parent->child.erase(parent->child.begin() + i);
				break;
			}
		}
	}

	while (child.size() > 0)
	{
		delete child[0];
	}
}


PropertyNode& PropertyNode::operator [] (int index)
{
	if (index < 0 || index >= (int)child.size()) throw SHARKEXCEPTION("[PropertyNode::operator []] access violation");
	return *child[index];
}

const PropertyNode& PropertyNode::operator [] (int index) const
{
	if (index < 0 || index >= (int)child.size()) throw SHARKEXCEPTION("[PropertyNode::operator []] access violation");
	return *child[index];
}

PropertyNode& PropertyNode::operator [] (const char* name)
{
	int i, ic = child.size();
	for (i=0; i<ic; i++)
	{
		if (child[i]->name != NULL && strcmp(child[i]->name, name) == 0) return *child[i];
	}

	throw SHARKEXCEPTION("[PropertyNode::operator []] access violation");
	return *((PropertyNode*)NULL);		// dead code to avoid warning
}

const PropertyNode& PropertyNode::operator [] (const char* name) const
{
	int i, ic = child.size();
	for (i=0; i<ic; i++)
	{
		if (strcmp(child[i]->name, name) == 0) return *child[i];
	}

	throw SHARKEXCEPTION("[PropertyNode::operator []] access violation");
	return *((const PropertyNode*)NULL);		// dead code to avoid warning
}

const char* PropertyNode::getSelected() const
{
	if (type != etSelectOne) throw SHARKEXCEPTION("[PropertyNode::getSelected] type mismatch");
	if (value.e < 0 || value.e >= (int)child.size()) throw SHARKEXCEPTION("[PropertyNode::getSelected] invalid value");
	return child[value.e]->getName();
}

int PropertyNode::getSelectedIndex() const
{
	if (type != etSelectOne) throw SHARKEXCEPTION("[PropertyNode::getSelectedIndex] type mismatch");
	if (value.e < 0 || value.e >= (int)child.size()) throw SHARKEXCEPTION("[PropertyNode::getSelectedIndex] invalid value");
	return value.e;
}

const PropertyNode& PropertyNode::getSelectedNode() const
{
	if (type != etSelectOne) throw SHARKEXCEPTION("[PropertyNode::getSelectedNode] type mismatch");
	if (value.e < 0 || value.e >= (int)child.size()) throw SHARKEXCEPTION("[PropertyNode::getSelectedNode] invalid value");
	return *(child[value.e]);
}

PropertyNode& PropertyNode::getSelectedNode()
{
	if (type != etSelectOne) throw SHARKEXCEPTION("[PropertyNode::getSelectedNode] type mismatch");
	if (value.e < 0 || value.e >= (int)child.size()) throw SHARKEXCEPTION("[PropertyNode::getSelectedNode] invalid value");
	return *(child[value.e]);
}

void PropertyNode::set(bool value)
{
	if (type != etBool) throw SHARKEXCEPTION("[PropertyNode::set] type mismatch");
	this->value.b = value;
}

void PropertyNode::set(int value)
{
	if (type != etInt) throw SHARKEXCEPTION("[PropertyNode::set] type mismatch");
	this->value.i = value;
}

void PropertyNode::set(double value)
{
	if (type != etDouble) throw SHARKEXCEPTION("[PropertyNode::set] type mismatch");
	this->value.d = value;
}

void PropertyNode::set(const char* value)
{
	if (type != etString) throw SHARKEXCEPTION("[PropertyNode::set] type mismatch");
	free(this->value.s);
	int len = strlen(value);
	this->value.s = (char*)malloc(len + 1);
	if (this->value.s == NULL) throw SHARKEXCEPTION("[PropertyNode::set] out of memory");
	memcpy(this->value.s, value, len + 1);
}

void PropertyNode::set(const PropertyNode& child)
{
	if (type != etSelectOne) throw SHARKEXCEPTION("[PropertyNode::set] type mismatch");

	int i, ic = this->child.size();
	for (i=0; i<ic; i++)
	{
		if (this->child[i] == &child)
		{
			this->value.e = i;
			return;
		}
	}

	throw SHARKEXCEPTION("[PropertyNode::set] node is not a child");
}

void PropertyNode::setRange(int minimum, int maximum)
{
	if (type != etInt) throw SHARKEXCEPTION("[PropertyNode::setRange] type mismatch");
	this->minimum.i = minimum;
	this->maximum.i = maximum;
}

void PropertyNode::setRange(double minimum, double maximum, bool logarithmic)
{
	if (type != etDouble) throw SHARKEXCEPTION("[PropertyNode::setRange] type mismatch");
	this->minimum.d = minimum;
	this->maximum.d = maximum;
	this->logarithmic = logarithmic;
}

void PropertyNode::description(char* buffer, int indent) const
{
	char tmp[4096];
	int i;
	for (i=0; i<indent; i++) buffer[i] = ' ';
	buffer[indent] = 0;
	strcat(buffer, name);
	switch (type)
	{
		case etBool:
			if (value.b) strcat(buffer, ": true\n");
			else strcat(buffer, ": false\n");
			break;
		case etInt:
			sprintf(tmp, ": %d\n", value.i);
			strcat(buffer, tmp);
			break;
		case etDouble:
			sprintf(tmp, ": %g\n", value.d);
			strcat(buffer, tmp);
			break;
		case etString:
			sprintf(tmp, ": %s\n", value.s);
			strcat(buffer, tmp);
			break;
		case etSelectOne:
// 			sprintf(tmp, ": %s\n", child[value.e]->getName());
			sprintf(tmp, ":\n");
			strcat(buffer, tmp);
			child[value.e]->description(tmp, indent+2);
			strcat(buffer, tmp);
			break;
		case etBranch:
			strcat(buffer, ":\n");
			for (i=0; i<(int)child.size(); i++)
			{
				child[i]->description(tmp, indent+2);
				strcat(buffer, tmp);
			}
			break;
		case etConst:
			strcat(buffer, "\n");
			break;
	}
}


////////////////////////////////////////////////////////////


Configuration::Configuration(const char* structure)
: PropertyNode(NULL)
{
	this->structure = structure;

	int pos = 0;
	int line = 1;
	int len = strlen(structure);
	char* copy = (char*)malloc(len + 1);
	if (copy == NULL) throw SHARKEXCEPTION("[Configuration::Configuration] out of memory");
	memcpy(copy, structure, len + 1);
	bool result = RecBuildStructure(this, copy, pos, line);
	free(copy);
	if (result == false || pos != len)
	{
		sprintf(exception, "[Configuration::Configuration] error in structure definition in line %d (position %d)", line, pos);
		throw SHARKEXCEPTION(exception);
	}
}

// deep copy
Configuration::Configuration(const Configuration& other)
: PropertyNode(other, NULL)
{
	structure = other.getStructure();
}

Configuration::~Configuration()
{
}


bool Configuration::GetToken(char* structure, int& pos, int& line)
{
	int len = strlen(structure);
	while (true)
	{
		char c = structure[pos];
		if (c != ' ' && c != '\t' && c != '\r' && c != '\n')
		{
			pos++;
			if (pos == len) return false;
		}
		else
		{
			if (c == '\n') line++;
			structure[pos] = 0;
			pos++;
			if (pos == len) return false;
			break;
		}
	}
	return true;
}

bool Configuration::SkipWhitespace(char* structure, int& pos, int& line, int len)
{
	while (structure[pos] == ' ' || structure[pos] == '\t' || structure[pos] == '\r' || structure[pos] == '\n')
	{
		if (structure[pos] == '\n') line++;
		pos++;
		if (pos == len) return false;
	}

	return true;
}

bool Configuration::RecBuildStructure(PropertyNode* property, char* structure, int& pos, int& line)
{
	int len = strlen(structure);
	char* name;
	char* type;
	char* defaultvalue;
	PropertyNode* p = property->getParent();
	eType t;

	if (structure[pos] != '{') return false;
	pos++; if (pos == len) return false;

	if (! SkipWhitespace(structure, pos, line, len)) return false;

	// get the name and test its uniqueness
	name = structure + pos;
	if (! GetToken(structure, pos, line)) return false;
	if (p != NULL)
	{
		int i, ic = p->size();
		for (i=0; i<ic; i++)
		{
			const char* n = (*p)[i].getName();
			if (n != NULL && strcmp(n, name) == 0) return false;
		}
	}
	int namelen = strlen(name);
	property->name = (char*)malloc(namelen + 1);
	if (property->name == NULL) throw SHARKEXCEPTION("[Configuration::RecBuildStructure] out of memory");
	memcpy(property->name, name, namelen + 1);

	if (! SkipWhitespace(structure, pos, line, len)) return false;

	// get the type and check its validity
	type = structure + pos;
	if (! GetToken(structure, pos, line)) return false;
	if (strcmp(type, "bool") == 0) t = etBool;
	else if (strcmp(type, "int") == 0) t = etInt;
	else if (strcmp(type, "double") == 0) t = etDouble;
	else if (strcmp(type, "string") == 0) t = etString;
	else if (strcmp(type, "select") == 0) t = etSelectOne;
	else if (strcmp(type, "branch") == 0) t = etBranch;
	else if (strcmp(type, "const") == 0) t = etConst;
	else return false;
	property->type = t;

	if (! SkipWhitespace(structure, pos, line, len)) return false;

	if (t == etBool || t == etInt || t == etDouble || t == etString)
	{
		// get the default value
		if (structure[pos] != '}')
		{
			defaultvalue = structure + pos;
			if (! GetToken(structure, pos, line)) return false;

			if (t == etBool)
			{
				if (strcmp(defaultvalue, "false") == 0) property->set(false);
				else if (strcmp(defaultvalue, "true") == 0) property->set(true);
				else return false;
			}
			else if (t == etInt)
			{
				char* end;
				long int v = strtol(defaultvalue, &end, 10);
				if (end != structure + pos - 1) return false;
				property->set((int)v);

				// read the range
				int l, u;
				if (! SkipWhitespace(structure, pos, line, len)) return false;
				char* lower = structure + pos;
				if (! GetToken(structure, pos, line)) return false;
				l = strtol(lower, &end, 10);
				if (end != structure + pos - 1) return false;
				if (! SkipWhitespace(structure, pos, line, len)) return false;
				char* upper = structure + pos;
				if (! GetToken(structure, pos, line)) return false;
				u = strtol(upper, &end, 10);
				if (end != structure + pos - 1) return false;

				property->setRange(l, u);
			}
			else if (t == etDouble)
			{
				char* end;
				double v = strtod(defaultvalue, &end);
				if (end != structure + pos - 1) return false;
				property->set(v);

				// read the range
				double l, u;
				if (! SkipWhitespace(structure, pos, line, len)) return false;
				char* lower = structure + pos;
				if (! GetToken(structure, pos, line)) return false;
				l = strtod(lower, &end);
				if (end != structure + pos - 1) return false;
				if (! SkipWhitespace(structure, pos, line, len)) return false;
				char* upper = structure + pos;
				if (! GetToken(structure, pos, line)) return false;
				u = strtod(upper, &end);
				if (end != structure + pos - 1) return false;

				// check log attribute
				bool lg = false;
				if (! SkipWhitespace(structure, pos, line, len)) return false;
				if (structure[pos] == 'l' && structure[pos+1] == 'o' && structure[pos+2] == 'g')
				{
					char* logattr = structure + pos;
					if (! GetToken(structure, pos, line)) return false;
					if (strcmp(logattr, "log") != 0) return false;
					lg = true;
				}

				property->setRange(l, u, lg);
			}
			else if (t == etString)
			{
				if (defaultvalue[0] != '\'') return false;
				int i, ic = strlen(defaultvalue);
				for (i=1; i<ic-1; i++) if (defaultvalue[i] == '\'') return false;
				if (defaultvalue[ic-1] != '\'') return false;
				defaultvalue[ic-1] = 0;
				defaultvalue++;
				property->set((const char*)defaultvalue);
			}

			if (! SkipWhitespace(structure, pos, line, len)) return false;
		}
	}
	else if (t == etSelectOne || t == etBranch)
	{
		defaultvalue = NULL;
		PropertyNode* selected = NULL;
		if (t == etSelectOne)
		{
			if (structure[pos] != '{' && structure[pos] != '}')
			{
				defaultvalue = structure + pos;
				if (! GetToken(structure, pos, line)) return false;

				if (! SkipWhitespace(structure, pos, line, len)) return false;
			}
		}

		// read children
		while (structure[pos] == '{')
		{
			PropertyNode* sub = new PropertyNode(property);
			if (sub == NULL) throw SHARKEXCEPTION("[Configuration::RecBuildStructure] out of memory");
			if (! RecBuildStructure(sub, structure, pos, line)) return false;
			if (pos == len) return false;
			if (defaultvalue != NULL && strcmp(defaultvalue, sub->getName()) == 0) selected = sub;
		}

		if (t == etSelectOne)
		{
			if (property->size() == 0) SHARKEXCEPTION("[Configuration::RecBuildStructure] empty selection");
			if (selected != NULL) property->set(*selected);
			else property->set((*property)[0]);
		}
	}

	if (structure[pos] != '}') return false;
	pos++; if (pos == len) return true;

	while (structure[pos] == ' ' || structure[pos] == '\t' || structure[pos] == '\r' || structure[pos] == '\n')
	{
		if (structure[pos] == '\n') line++;
		pos++;
	}

	return true;
}
