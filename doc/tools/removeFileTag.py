#!/usr/bin/python

# 
# \file        removeFileTag.py
#
# \brief       a small tool that will remove all \file tags from the header
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


print "\nShark file tag remove tool v0.1\n\n"


# parse arguments
parser = argparse.ArgumentParser (description='Remove file tags.')
parser.add_argument('path', metavar='path', type=str, nargs='+', help='path')
args = parser.parse_args()

# parse working directory
originalDir = os.path.abspath(args.path[0])
print 'Working in directory: ', originalDir
 


originalFiles = []

count = 0

for root, subFolders, files in os.walk(originalDir):
    for file in files:
        filepath = os.path.abspath (os.path.join(root, file))
        filesize = os.path.getsize(filepath)
    
        # only accept .h .hpp .c .cpp .tpp .tut 
        r=re.compile('.*(\.h|\.hpp|\.c|\.cpp|\.tpp|\.tut)$')
        if r.match (filepath):
            if (verbose == True):
                print filepath
        else:
            continue

        # read whole file
        with open(filepath) as f:
            data = f.read()

        try:
            # grep block-comments
            (match, start, end) = cStyleComment.scanString(data).next()
       
            # copy for later use
            cmt = match[0]
            
            # check whether we have a brief and a copyright
            if (re.match(r'(?is)(.*)\\file(.*)', cmt)):
                # do some replacement

                # grep the old brief, author, and date
                if (verbose == True):
                    match = re.search(r'(?is).*([ \t]*\*[ \t]*\\file[ \t].*?\n)', cmt)
                    if match:
                        rawbrief = match.group(1)

                newHeader = re.sub(r'(?is)([ \t]*\*[ \t]*\\file[ \t].*?\n)', '', cmt)
                
                # replace header by new one
                newdata = data[:start] + newHeader + data[end:]

                # rewrite whole file
                newfile = open (filepath, "w")
                newfile.write ("%s"%newdata)
                newfile.close()

                if (verbose == True):
                    print cmt
                    print "======================================================================"
                    time.sleep(3)
                
                print "  Processed", filepath
                count = count + 1
                
        except (StopIteration) as e:
            # no header found, decide if we should add one
            
            continue
        else:
            pass
        
print "\nProcessed", count, "files.\n\n"
