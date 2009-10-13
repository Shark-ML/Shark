
/**
 * \file RCObject.cpp
 *
 * \brief Base class for Reference Counted Objects
 * 
 * \authors Marc Nunkesser (taken from More Effective C++ by Scott Meyers)
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


