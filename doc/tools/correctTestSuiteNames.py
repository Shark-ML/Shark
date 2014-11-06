#!/usr/bin/python

# 
# \file        correctTestSuiteNames.py
#
# \brief       a small tool that will try to fix the test suite names for better xunit view
#
# \author      Aydin Demircioglu
# \date        2014
#
#
# \par Copyright 1995-2014 Shark Development Team
# 
# <BR><HR>
# This file is part of Shark.
# <http://image.diku.dk/shark>
# 
# Shark is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published 
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Shark is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with Shark.  If not, see <http:#www.gnu.org/licenses/>.
#

import os
import hashlib
import shutil
import argparse
import re
import time
from pyparsing import cStyleComment
from pyparsing import javaStyleComment
from pyparsing import cppStyleComment
from pyparsing import dblSlashComment



verbose = False


print "\nShark test suite name replacement tool v0.1\n\n"


# parse arguments
parser = argparse.ArgumentParser (description='Replace headers.')
parser.add_argument('path', metavar='path', type=str, nargs='+', help='path')
args = parser.parse_args()

# parse working directory
originalDir = os.path.abspath(args.path[0])
print 'Working in directory: ', originalDir


verbose = True

originalFiles = []

count = 0
countOtherFiles = 0

for root, subFolders, files in os.walk(originalDir):
	for file in files:
		filepath = os.path.abspath (os.path.join(root, file))
		filesize = os.path.getsize(filepath)
	
		# only accept .h .hpp .c .cpp .tpp .tut 
		r=re.compile('.*.cpp$')
		if r.match (filepath):
			if (verbose == True):
				print filepath
		else:
			continue

		# relative path+filename-cpp = suitename 
#		relpath = os.path.relpath(root, originalDir)
		relpath = os.path.relpath(os.path.join(root, file), originalDir)
		relpath = re.sub(r'(?is).cpp', '', relpath)
		suitename = re.sub(r'(?is)/', '_', relpath)
		if (verbose == True):
			print "  suitename: "+suitename
		
		# read whole file
		with open(filepath) as f:
			data = f.read()

		# -- check if at least one test case is defined 
		r=re.compile('(?is).*BOOST_AUTO_TEST_CASE.*')
		if r.match (data):
			print "  -has at least one auto test case"
		else:
			countOtherFiles += 1
			continue

		# check whether we have a fixture suite or not
		if (re.match(r'(?is).*BOOST_FIXTURE_TEST_SUITE.*', data)):
			# this is different
			print "  -but is a fixture test suite.";
			replacement = "BOOST_FIXTURE_TEST_SUITE (" + suitename + ",\\1)"
			data = re.sub(r'(?is)BOOST_FIXTURE_TEST_SUITE.?\(.*?,(.*?)\)', replacement, data, 1)
			# sanity check: is the end also there?
			# add to the end of file, but check existence first, just to be sure
			if (re.match(r'(?is).*BOOST_AUTO_TEST_SUITE_END.*', data)):
				# everything is ok.
				print ("    found end tag, ok.")
			else:
				# oops?
				if (verbose == True):
					print ("      warning: file had NO test_suite_end! adding it!");
				data = data + "\nBOOST_AUTO_TEST_SUITE_END()\n";
		else:
			# check if we have a boost_test_module or not
			if (re.match(r'(?is).*BOOST_AUTO_TEST_SUITE.*', data)):
				# there is one already.
				# so we need to replace it.
				print "  -but alrady has an auto test suite environment.";
				print "  -will replace that."
				replacement = "BOOST_AUTO_TEST_SUITE (" + suitename + ")"
				data = re.sub(r'(?is)BOOST_AUTO_TEST_SUITE.?\(.*?\)', replacement, data, 1)
				
				# sanity check: is the end also there?
				# add to the end of file, but check existence first, just to be sure
				if (re.match(r'(?is).*BOOST_AUTO_TEST_SUITE_END.*', data)):
					# everything is ok.
					print ("    found end tag, ok.")
				else:
					# oops?
					if (verbose == True):
						print ("      warning: file had NO test_suite_end! adding it!");
					data = data + "\nBOOST_AUTO_TEST_SUITE_END()\n";
			else:
				# there is none.
				# so we have to add one
				# we do this just before the first BOOST_AUTO_TEST_CASE.
				# we know it exists.
				replacement = "BOOST_AUTO_TEST_SUITE (" + suitename + ")\n\nBOOST_AUTO_TEST_CASE"
				data = re.sub(r'(?is)BOOST_AUTO_TEST_CASE', replacement, data, 1)
				
				# add to the end of file, but check existence first, just to be sure
				if (re.match(r'(?is).*BOOST_AUTO_TEST_SUITE_END.*', data)):
					# oops?
					if (verbose == True):
						print ("      warning: file already has test_suite_end!");
				else:
					data = data + "\nBOOST_AUTO_TEST_SUITE_END()\n";
					
		# rewrite data
		print "  -rewriting...";
		newfile = open (filepath, "w")
		newfile.write ("%s"%data)
		newfile.close()
			
		print "  -done.";
		count = count + 1
		
print "\nProcessed", count, "files."
print "Found", countOtherFiles, "unrelated files without any boost tests.\n"
