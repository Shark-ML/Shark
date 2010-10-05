
/**
 * \file RCObject.cpp
 *
 * \brief Base class for Reference Counted Objects
 * 
 * \authors Marc Nunkesser, Copyright (c) 2008, Marc Nunkesser (taken from More Effective C++ by Scott Meyers)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */

/* $log$ */

#include <iostream>
#include "Fuzzy/RCObject.h"

RCObject::RCObject() : refCount(0), shareable(true) {};

RCObject::RCObject(const RCObject&): refCount(0), shareable(true) {}

RCObject& RCObject::operator=(const RCObject&) {
	return *this;
}

RCObject::~RCObject() {}

void RCObject::addReference() {
	++refCount;
	//cout<<"addRef:"<<refCount<<endl;
};

void RCObject::removeReference() {
	if (--refCount == 0) {
		// cout<<"delete this in removeReference"<<endl;
		delete this;
	};
// else { cout<<"simples removeReference"<< endl;};
};

//void RCObject::markUnshareable() {
//	shareable = false;
//}

//bool RCObject::isShareable() const {
//	return shareable;
//}

//bool RCObject::isShared() const {
//	return refCount > 1;
//}


