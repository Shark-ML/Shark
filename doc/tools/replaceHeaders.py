#!/usr/bin/python

# 
# \file        replaceHeaders.py
#
# \brief       a small tool that will rewrite headers from shark source files with the current header template default.
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


# FIXME:
# 1) \brief A\n \n B C \file... newlines get deleted
# 2) \brief Implements \f$.. \f is interpreted as next doxygen, so gets cut
# 3) */\n#ifndef got replaced to ndef??
# 4)

 
verbose = False


print "\nShark header replacement tool v0.1\n\n"


# parse arguments
parser = argparse.ArgumentParser (description='Replace headers.')
parser.add_argument('path', metavar='path', type=str, nargs='+', help='path')
args = parser.parse_args()

# parse working directory
originalDir = os.path.abspath(args.path[0])
print 'Working in directory: ', originalDir
 

# read header template
with open(os.path.expanduser("../header_template.txt")) as f:
    header = f.read()


# replace fixed dummy strings in header
homepage = "http://image.diku.dk/shark"
header = re.sub(r'(?is)%YEAR%', time.strftime("%Y"), header)
header = re.sub(r'(?is)%HOMEPAGE%', homepage, header)



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

        # as far as i see:
        # -header is never beyond the very first c-style/block-comment match
        #       so it is enough to look at the first occurence of a c-style-comment
        # -there are // headers, but none of them contain a shark-copyright-notice,
        #       except ./include/shark/Models/RecurrentStructure.h

        # read whole file
        with open(filepath) as f:
            data = f.read()

        try:
            # grep block-comments
            (match, start, end) = cStyleComment.scanString(data).next()
       
            # copy for later use
            cmt = match[0]
            
            # check whether we have a brief and a copyright
            if (re.match(r'(?is).*Copyright.*', cmt) or re.match (r'(?is).*GNU General Public License as published.*', cmt)):
                # do some replacement

                # grep the old brief, author, and date
                rawbrief = "-"
                match = re.search(r'(?is)\\brief\s*(.*?)(\\a|\\d|<BR>)', cmt)
                if match:
                    rawbrief = match.group(1)
                    # remove comment things
                    rawbrief = re.sub(r'(?ims)[ \t]*\*[ \t]*', r'', rawbrief)
                    # FIXME: this is not nice, adding back the comment signs, but for now..
                    rawbrief = re.sub(r'(?ims)^[ \t]*(\S+)', r' * \1', rawbrief)
                    # reinsert comment at empty lines
                    rawbrief = re.sub(r'(?ims)^$', r' * ', rawbrief)
                    rawbrief = re.sub(r'(?ims)\s+\*[ \t]*(.*)', r'\1', rawbrief)
                    
                rawauthor = "-"
                match = re.search(r'(?is)\\author\s*(.*?)(\\b|\\d|<BR>)', cmt)
                if match:
                    rawauthor = match.group(1)
                    rawauthor = re.sub(r'(?ims)^\s*\*\s*$', '', rawauthor)
                    rawauthor = re.sub(r'(?ims)\n$', '', rawauthor) 

                rawdate = "-"
                match = re.search(r'(?is)\\date\s*(.*?)[\\|\*/]', cmt)
                if match:
                    rawdate = match.group(1)
                    rawdate = re.sub(r'(?ims)^\s*\*\s*$', '', rawdate)
                    rawdate = re.sub(r'(?ims)^\s*\*\s*(\S+)', r'\1', rawdate)
                    rawdate = re.sub(r'(?ims)(.*)\n(.*)', r'\1\2', rawdate)
                    rawdate = re.sub(r'(?ims)(.*?)\s+$', r'\1', rawdate)
                  
                if (verbose == True):
                    print "brief", rawbrief
                    print "author", rawauthor
                    print "date", rawdate
                        
                #cmt = re.sub(r'(?is)\\brief(.*)\\?', r'\1', cmt)
                cmt = re.sub(r'(?is)\\brief(.*?)\\', r'BRIEF\\', cmt)
                cmt = re.sub(r'(?is)\\author(.*?)\\', r'AUTHOR\\', cmt)
                cmt = re.sub(r'(?is)\\date(.*?)\\', r'DATE\\', cmt)

                # remove the old copyright statement
                cmt = re.sub(r'(?is)\\par Copyright (.*)bochum.de(<BR>)?','COPYRIGHT', cmt)
                cmt = re.sub(r'(?is)<BR><HR>(.*)free software(.*)licenses/>','LGPL', cmt)

                # replace the other dummies: %BRIEF% and %FILE%
                currentHeader = re.sub(r'(?is)%FILE%', file, header)
                currentHeader = re.sub(r'(?is)%BRIEF%', rawbrief, currentHeader)
                currentHeader = re.sub(r'(?is)%DATE%', rawdate, currentHeader)
                currentHeader = re.sub(r'(?is)%AUTHOR%', rawauthor, currentHeader)
                
                # make sure there are no dummies left
                #print currentHeader
                dummiesleft = re.match('(?is).*(%.*%).*', currentHeader)
                if dummiesleft:
                    print "Error: Dummy", dummiesleft.group(1), "seems not have been replaced."
                    exit()

                # replace header by new one
                newdata = data[:start] + currentHeader + data[end:]

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