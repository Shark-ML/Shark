
//
// TODO:
// (-) generate system specific QT project files
//     and add build support for the corresponding
//     examples to the main and the library makefiles
//

//
// This program generates makefiles and
// microsoft visual studio project and
// workspace files for the Shark library.
// It can and should be extended in case
// further build environments need to be
// supported.
//
//
// File format
// -----------
//
// This program searches the program
// directory tree for *.SharkConfig files.
// These files have the following XML
// based structure:
//
// <config name="..." basedir="...">
//   <project name="..." type="..." lib="..." libpath="..." dependflag="..." system="...">
//     <source path="rel-path" name="filename or mask"></source>
//     <source path="rel-path" name="filename or mask"></source>
//     ...
//     <header path="rel-path" name="filename or mask"></header>
//     <header path="rel-path" name="filename or mask"></header>
//     ...
//   </project>
//   <project ...>...</project>
//   ...
// </config>
//
// The config parameters are:
//   name        configuration name as a single token string that must be a valid filename on all supported systems
//   basedir     relative position of the Shark main directory
//
// The project parameters are:
//   name        project name as a single token string that must be a valid filename on all supported systems
//   type        "executable" or "library"
//   lib         optional, link this library into the project, may occur more than once
//   libpath     optional, additional library path, may occur more than once
//   dependflag  optional, project depends on this flag, e.g. "FLAG_SIMANN" or "FLAG_PVM"
//   system      optional, may be "makefile" or "windows" and restricts this project to the given build system
//
// The program generates one makefile.in and
// one workspace file for each *.SharkConfig
// file, based on the configuration name.
// There will be one project file per project
// tag. The makefile will contain two commands
// per project (release and debug version) as
// well as the general commands "all",
// "debug", and "clean".
//
// Furthermore, the program generates a main
// makefile and two main workspaces
// SharkAll.dsw and SharkLibs.dsw. The main
// makefile provides the commands "libs",
// "debug", "examples", 
// "all", "install", and "clean".
//
// SharkConfig also generates two central
// CMakeLists.txt files, one in the main
// folder and one in the examples subfolder.
//
// The QT examples are supported through the
// generation of qmake project files.
// Furthermore, the projects are included in
// the cmake lists.
//
// Jobs like "make doc" are no longer found
// in the makefiles. Instead, shell scripts
// handle all administrative jobs.
//
// There is a central SharkConfig.Libs file
// making library names and pathes known to
// the build system. It has the following
// format:
//
// <libs>
//   <lib path="..." name="..."></lib>
//   <lib path="..." name="..."></lib>
//   ...
// </libs>
//
// Whenever a lib="..." tag is found in a
// project configuration the central <lib>
// entry with the corresponding name tag
// describes the -L and -l options passed to
// the compiler. This greatly simplifies the
// usage of Shark components by other
// components or examples. If a lib usage
// tag specifies an unknown name an external
// dependency is assumed, like e.g.
// lib="pvm". In this case the corresponding
// path should be added by hand using the
// libpath="..." syntax.
//


#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "include/SharkDefs.h"


struct FileDesc
{
	char path[256];		// without trailing delimiter
	char name[224];		// name
	char extension[32];	// extension with dot
};

enum eProjectType
{
	epExecutable,
	epExecutableQt,
	epLibrary,
};

enum eSystemType
{
	esMakefile = 1,
	esWindows = 2,
	esAll = 255,
};

struct XmlTag
{
	char tag[256];
	char value[256];
};

class XmlKnot
{
public:
	char name[256];
	std::vector<XmlKnot*> sub;
	std::vector<XmlTag> tag;
};

class Workspace;

class Project
{
public:
	Project()
	{
		path[0] = 0;
		name[0] = 0;
		type = epExecutable;
		dependflag[0] = 0;
		system = esAll;
	}

	char path[256];
	char name[256];
	eProjectType type;
	std::vector<char*> lib;
	std::vector<char*> libpath;
	char dependflag[256];
	eSystemType system;
	std::vector<FileDesc> source;
	std::vector<FileDesc> header;
	Workspace* workspace;
};

class Workspace
{
public:
	Workspace()
	{
		path[0] = 0;
		name[0] = 0;
		basedir[0] = 0;
		onlyQT = false;
	}

	char path[256];
	char name[256];
	char basedir[256];
	std::vector<Project*> project;
	bool onlyQT;
};

class Library
{
public:
	Library()
	{
		path[0] = 0;
		name[0] = 0;
	}

	char path[256];
	char name[256];
};

std::vector<Library*> libraries;

Library* findLibrary(const char* name)
{
	int i, ic = libraries.size();
	for (i = 0; i < ic; i++)
	{
		if (strcmp(libraries[i]->name, name) == 0) return libraries[i];
	}
	return NULL;
}

// compare two c strings
bool Compare(char* str1, char* str2)
{
	return (strcmp(str1, str2) < 0);
}

// nearly thread safe, safe enough for printf
char tmp_path[10][256];
int tmp_path_index = 0;
char* WindowsPath(char* path)
{
	int i;
	char* p = tmp_path[tmp_path_index];

	for (i = 0; path[i] != 0; i++)
	{
		char c = path[i];
		if (c == '/') p[i] = '\\'; else p[i] = c;
	}
	p[i] = 0;

	tmp_path_index++;
	if (tmp_path_index == 10) tmp_path_index = 0;

	return p;
}

// nearly thread safe, safe enough for printf
char tmp_projname[10][256];
int tmp_projname_index = 0;
char* TokenName(char* name)
{
	int i, j;
	char* p = tmp_projname[tmp_projname_index];

	for (i = 0, j = 0; name[i] != 0; i++)
	{
		char c = name[i];
		if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
		{
			p[j] = c;
			j++;
		}
	}
	p[j] = 0;

	tmp_projname_index++;
	if (tmp_projname_index == 10) tmp_projname_index = 0;

	return p;
}

// The mask may contain star wildcards ('*').
bool Fits(const char* string, const char* mask)
{
	int ml = strlen(mask);
	int sl = strlen(string);
	int i, star;
	for (star = ml - 1; star >= 0; star--) if (mask[star] == '*') break;
	if (star == -1) return(strcmp(string, mask) == 0);
	if (star > sl) return false;
	if (star > 0)
	{
		if (memcmp(string, mask, star) != 0) return false;
	}
	for (i = star; i < sl; i++)
	{
		if (Fits(string + i, mask + star + 1)) return true;
	}
	return false;
}

// fill the list with all files fitting the mask
void ScanDirectory(const char* startpath, const char* subpath, const char* mask, std::vector<FileDesc>& list, bool recurse)
{
	char path[256];
	sprintf(path, "%s/%s", startpath, subpath);
	DIR* dir = opendir(path);
	if (dir == NULL) return;

	dirent* entry;
	int len;
	while (true)
	{
		entry = readdir(dir);
		if (entry == NULL) break;

		len = strlen(entry->d_name);
		if (len == 0) break;

		if (entry->d_name[0] == '.') continue;

		if (Fits(entry->d_name, mask))
		{
			if (memcmp("moc_", entry->d_name, 4) != 0)
			{
				FileDesc fd;
				strcpy(fd.path, subpath);
				strcpy(fd.name, entry->d_name);
				int dot;
				for (dot = strlen(entry->d_name) - 1; dot >= 0; dot--) if (entry->d_name[dot] == '.') break;
				if (dot == -1) fd.extension[0] = 0;
				else
				{
					fd.name[dot] = 0;
					strcpy(fd.extension, entry->d_name + dot);
				}
				list.push_back(fd);
			}
		}
		else if (recurse)
		{
			char sub[256];
			sprintf(sub, "%s/%s", subpath, entry->d_name);
			ScanDirectory(startpath, sub, mask, list, recurse);
		}
	}

	closedir(dir);
}

// Read a string up to a length of 255
// characters plus terminating 0 character.
bool ReadString(FILE* file, char* string)
{
	char c;
	bool quotes = false;
	int len = 0;

	if (fread(&c, 1, 1, file) == 0) return false;
	if (c == '\"') quotes = true;
	else
	{
		string[len] = c;
		len++;
	}
	while (len < 255)
	{
		if (fread(&c, 1, 1, file) == 0) return false;
		if (quotes)
		{
			if (c == '\"')
			{
				string[len] = 0;
				return true;
			}
		}
		else
		{
			if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')))
			{
				fseek(file, -1, SEEK_CUR);
				string[len] = 0;
				return true;
			}
		}
		string[len] = c;
		len++;
	}
	return false;
}

// read an XML knot
XmlKnot* ReadXmlKnot(FILE* file)
{
	XmlKnot* ret = new XmlKnot();
	char c;
	int state = 0;

	// states:
	// -1  [done]
	//  0  <
	//  1  name
	//  2  tag or >
	//  3  =
	//  4  value
	//  5  subknot or </
	//  6  name
	//  7  >

	while (state != -1)
	{
		if (fread(&c, 1, 1, file) == 0) return NULL;
		if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
		switch (state)
		{
		case 0:
		{
			if (c != '<') return NULL;
			state = 1;
			break;
		}
		case 1:
		{
			fseek(file, -1, SEEK_CUR);
			if (! ReadString(file, ret->name)) return NULL;
			state = 2;
			break;
		}
		case 2:
		{
			if (c == '>') state = 5;
			else
			{
				fseek(file, -1, SEEK_CUR);
				ret->tag.resize(ret->tag.size() + 1);
				if (! ReadString(file, ret->tag[ret->tag.size() - 1].tag)) return NULL;
				state = 3;
			}
			break;
		}
		case 3:
		{
			if (c != '=') return NULL;
			state = 4;
			break;
		}
		case 4:
		{
			fseek(file, -1, SEEK_CUR);
			if (! ReadString(file, ret->tag[ret->tag.size() - 1].value)) return NULL;
			state = 2;
			break;
		}
		case 5:
		{
			if (c != '<') return NULL;
			if (fread(&c, 1, 1, file) == 0) return NULL;
			if (c == '/') state = 6;
			else
			{
				fseek(file, -2, SEEK_CUR);
				XmlKnot* sub = ReadXmlKnot(file);
				if (sub == NULL) return NULL;
				ret->sub.push_back(sub);
			}
			break;
		}
		case 6:
		{
			char tmp[256];
			fseek(file, -1, SEEK_CUR);
			if (! ReadString(file, tmp)) return NULL;
			if (strcmp(tmp, ret->name) != 0) return NULL;
			state = 7;
			break;
		}
		case 7:
		{
			if (c != '>') return NULL;
			state = -1;
			break;
		}
		}
	}

	return ret;
}

Project* CreateProject(const char* path, const char* basedir, XmlKnot* knot)
{
	if (strcmp(knot->name, "project") != 0) return NULL;

	Project* ret = new Project();
	strcpy(ret->path, path);
	int i, ic;
	int j, jc;

	// extract properties
	ic = knot->tag.size();
	for (i = 0; i < ic; i++)
	{
		if (strcmp(knot->tag[i].tag, "name") == 0)
		{
			strcpy(ret->name, knot->tag[i].value);
		}
		else if (strcmp(knot->tag[i].tag, "type") == 0)
		{
			if (strcmp(knot->tag[i].value, "executable") == 0) ret->type = epExecutable;
			else if (strcmp(knot->tag[i].value, "executable-QT") == 0) ret->type = epExecutableQt;
			else if (strcmp(knot->tag[i].value, "library") == 0) ret->type = epLibrary;
			else return NULL;
		}
		else if (strcmp(knot->tag[i].tag, "lib") == 0)
		{
			char* str = (char*)malloc(strlen(knot->tag[i].value) + 1);
			strcpy(str, knot->tag[i].value);
			ret->lib.push_back(str);
		}
		else if (strcmp(knot->tag[i].tag, "libpath") == 0)
		{
			char* str = (char*)malloc(strlen(knot->tag[i].value) + 1);
			strcpy(str, knot->tag[i].value);
			ret->libpath.push_back(str);
		}
		else if (strcmp(knot->tag[i].tag, "dependflag") == 0)
		{
			strcpy(ret->dependflag, knot->tag[i].value);
		}
		else if (strcmp(knot->tag[i].tag, "system") == 0)
		{
			if (strcmp(knot->tag[i].value, "makefile") == 0) ret->system = esMakefile;
			else if (strcmp(knot->tag[i].value, "windows") == 0) ret->system = esWindows;
			else return NULL;
		}
	}

	// collect files
	ic = knot->sub.size();
	for (i = 0; i < ic; i++)
	{
		XmlKnot* sub = knot->sub[i];
		if (strcmp(sub->name, "source") == 0)
		{
			if (sub->tag.size() != 2) return NULL;
			if (strcmp(sub->tag[0].tag, "path") != 0) return NULL;
			if (strcmp(sub->tag[1].tag, "name") != 0) return NULL;
			char p[256];
			sprintf(p, "%s/%s", path, basedir);
			ScanDirectory(p, sub->tag[0].value, sub->tag[1].value, ret->source, false);
		}
		else if (strcmp(sub->name, "header") == 0)
		{
			if (sub->tag.size() != 2) return NULL;
			if (strcmp(sub->tag[0].tag, "path") != 0) return NULL;
			if (strcmp(sub->tag[1].tag, "name") != 0) return NULL;
			char p[256];
			sprintf(p, "%s/%s", path, basedir);
			ScanDirectory(p, sub->tag[0].value, sub->tag[1].value, ret->header, false);
		}
		else return NULL;
	}

	return ret;
}

Workspace* CreateWorkspace(const char* path, XmlKnot* knot)
{
	if (strcmp(knot->name, "config") != 0) return NULL;
	if (knot->tag.size() != 2) return NULL;
	if (strcmp(knot->tag[0].tag, "name") != 0) return NULL;
	if (strcmp(knot->tag[1].tag, "basedir") != 0) return NULL;

	Workspace* ret = new Workspace();
	strcpy(ret->path, path);
	strcpy(ret->name, knot->tag[0].value);
	strcpy(ret->basedir, knot->tag[1].value);

	bool onlyQT = true;
	int i, ic = knot->sub.size();
	for (i = 0; i < ic; i++)
	{
		XmlKnot* sub = knot->sub[i];
		if (strcmp(sub->name, "project") != 0) return NULL;
		Project* proj = CreateProject(path, ret->basedir, sub);
		proj->workspace = ret;
		if (proj == NULL) return NULL;
		ret->project.push_back(proj);
		if (proj->type != epExecutableQt) onlyQT = false;
	}
	ret->onlyQT = onlyQT;

	return ret;
}

Library* CreateLibrary(XmlKnot* knot)
{
	if (strcmp(knot->name, "lib") != 0) return NULL;
	if (knot->tag.size() != 2) return NULL;
	if (strcmp(knot->tag[0].tag, "path") != 0) return NULL;
	if (strcmp(knot->tag[1].tag, "name") != 0) return NULL;

	Library* ret = new Library();
	strcpy(ret->path, knot->tag[0].value);
	strcpy(ret->name, knot->tag[1].value);

	return ret;
}

char commentheader[1024] = "\r\n#\r\n# This file is automatically generated by SharkConfig.\r\n#\r\n\r\n";

void WriteCommentHeader(FILE* file)
{
	fputs( commentheader, file );
}

// Generate a unix Makefile
void GenerateMakefile(Workspace* ws)
{
	if (ws->onlyQT) return;

	char filename[512];
	int i, ic = ws->project.size();
	int j, jc;

	sprintf(filename, "%s/Makefile.in", ws->path);
	FILE* file = fopen(filename, "w+");
	if (file == NULL)
	{
		printf("        ERROR opening file %s for writing\n", filename);
		return;
	}
	WriteCommentHeader(file);

	fprintf(file, "\r\nHOST = @HOST@\r\nARCH_MACH = @ARCH_MACH@\r\nLDLIBS = @LIBS@\r\nDEFS = @DEFS@\r\n");
	fprintf(file, "prefix = @prefix@\r\n");
	fprintf(file, "exec_prefix = @exec_prefix@\r\n");
	fprintf(file, "includedir = @includedir@\r\n");
	fprintf(file, "libdir = @libdir@\r\n");
	fprintf(file, "datarootdir = @datarootdir@\r\n");
	fprintf(file, "LDFLAGS = @LDFLAGS@\r\nRPATHFLAG = @RPATHFLAG@\r\nCC = @CC@\r\nCXX = @CXX@\r\nLIBEXT = @LIBEXT@\r\nLIBSONAMEEXT = @LIBSONAMEEXT@\r\nLIBFULLEXT = @LIBFULLEXT@\r\nCFLAGS_ND = @CFLAGS_ND@\r\nCFLAGS_D = @CFLAGS_D@\r\nCFLAGS = @CFLAGS@\r\nLD_SHARED = @LD_SHARED@\r\nSONAMEFLAG = @SONAMEFLAG@\r\nSONAMEFLAG_D = @SONAMEFLAG_D@\r\nCFLAGS += $(DEFS)\r\nFLAG_PVM = @FLAG_PVM@\r\nLIBS_PVM = @LIBS_PVM@\r\nPVM_LDLIBS = @PVM_LDLIBS@\r\n");
	fprintf(file, "CFLAGS += -I%s/include\r\nCXXFLAGS = $(CFLAGS)\r\n", ws->basedir);
	fprintf(file, "\r\n");

	// target variables
	for (i = 0; i < ic; i++)
	{
		Project* proj = ws->project[i];
		if ((proj->system & esMakefile) == 0) continue;

		// targets/objectives, possibly with external dependencies
		if (proj->dependflag[0] != 0) fprintf(file, "ifeq ($(%s),__is_existent__)\r\n", proj->dependflag);

		if (proj->type == epExecutable) fprintf(file, "TARGET_%s = %s/%s/%s\r\n", TokenName(proj->name), ws->basedir, proj->path, proj->name);
		else if (proj->type == epLibrary) fprintf(file, "TARGET_%s = %s/lib/lib%s${LIBEXT}\r\n", TokenName(proj->name), ws->basedir, proj->name);
		if (proj->type == epExecutable) fprintf(file, "TARGET_%s_debug = %s/%s/%s_debug\r\n", TokenName(proj->name), ws->basedir, proj->path, proj->name);
		else if (proj->type == epLibrary) fprintf(file, "TARGET_%s_debug = %s/lib/lib%s_debug${LIBEXT}\r\n", TokenName(proj->name), ws->basedir, proj->name);

		if (proj->dependflag[0] != 0) fprintf(file, "endif\r\n");
	}
	fprintf(file, "\r\n");

	// implicit rules
	fprintf(file, "%%.o: %%.cpp\r\n\t$(CXX) $(CFLAGS) $(CFLAGS_ND) -o $@ -c $<\r\n\r\n");
	fprintf(file, "%%_debug.o: %%.cpp\r\n\t$(CXX) $(CFLAGS) $(CFLAGS_D) -o $@ -c $<\r\n\r\n");

	// all command
	fprintf(file, "all:");
	for (i = 0; i < ic; i++)
	{
		Project* proj = ws->project[i];
		if ((proj->system & esMakefile) == 0) continue;
		fprintf(file, " ${TARGET_%s}", TokenName(proj->name));
	}
	fprintf(file, "\r\n");
	fprintf(file, "\r\n");

	// debug command
	fprintf(file, "debug:");
	for (i = 0; i < ic; i++)
	{
		Project* proj = ws->project[i];
		if ((proj->system & esMakefile) == 0) continue;
		fprintf(file, " ${TARGET_%s_debug}", TokenName(proj->name));
	}
	fprintf(file, "\r\n");
	fprintf(file, "\r\n");

	// lib command
	fprintf(file, "lib:\r\n\t@-mkdir -p %s/lib\r\n\r\n", ws->basedir, ws->basedir);

	// commands per project
	for (i = 0; i < ic; i++)
	{
		Project* proj = ws->project[i];
		if ((proj->system & esMakefile) == 0) continue;

		// short names
		fprintf(file, "Project-%s: ${TARGET_%s}\r\n\r\n", proj->name, TokenName(proj->name));
		fprintf(file, "Project-%s-debug: ${TARGET_%s_debug}\r\n\r\n", proj->name, TokenName(proj->name));

		// rule for the target
		if (proj->type == epExecutable) fprintf(file, "%s/%s/%s:", ws->basedir, proj->path, proj->name);
		else if (proj->type == epLibrary) fprintf(file, "%s/lib/lib%s${LIBEXT}: lib", ws->basedir, proj->name);
		jc = proj->source.size();
		for (j = 0; j < jc; j++) fprintf(file, " %s/%s/%s.o", ws->basedir, proj->source[j].path, proj->source[j].name);
		fprintf(file, "\r\n");

		// linker command
		if (proj->type == epExecutable) fprintf(file, "\t$(CXX) $(LDFLAGS) $(LDLIBS) -o %s/%s/%s $(RPATHFLAG)", ws->basedir, proj->path, proj->name);
		else if (proj->type == epLibrary) {
			fprintf(file, "\t$(CXX) $(LD_SHARED) $(LDFLAGS) $(SONAMEFLAG) $(LDLIBS) -o %s/lib/lib%s${LIBFULLEXT}", ws->basedir, proj->name);
		}
		jc = proj->source.size();
		for (j = 0; j < jc; j++) fprintf(file, " %s/%s/%s.o", ws->basedir, proj->source[j].path, proj->source[j].name);
		jc = proj->lib.size();
		for (j = 0; j < jc; j++)
		{
			Library* lib = findLibrary(proj->lib[j]);
			if (lib != NULL) fprintf(file, " -L%s/lib -l%s", ws->basedir, lib->name);
			else fprintf(file, " -l%s", proj->lib[j]);
		}
		jc = proj->libpath.size();
		for (j = 0; j < jc; j++) fprintf(file, " -L%s/lib ", ws->basedir);
		if (proj->type == epLibrary) {
			fprintf(file, "\r\n\tchmod a+r %s/lib/lib%s${LIBFULLEXT}", ws->basedir, proj->name);
			fprintf(file, "\r\n\t(cd %s/lib/; ln -fs lib%s${LIBFULLEXT} lib%s${LIBEXT})", ws->basedir, proj->name, proj->name);
			fprintf(file, "\r\n\t(cd %s/lib/; ln -fs lib%s${LIBFULLEXT} lib%s${LIBSONAMEEXT})", ws->basedir, proj->name, proj->name);
			fprintf(file, "\r\n\t(cd %s/lib/; ln -fs lib%s${LIBFULLEXT} lib%c%s${LIBEXT})", ws->basedir, proj->name, toupper(proj->name[0]), &proj->name[1]);
		}
		fprintf(file, "\r\n\r\n");

		// rule for the debug target
		if (proj->type == epExecutable) fprintf(file, "%s/%s/%s_debug:", ws->basedir, proj->path, proj->name);
		else if (proj->type == epLibrary) fprintf(file, "%s/lib/lib%s_debug${LIBEXT}: lib", ws->basedir, proj->name);
		jc = proj->source.size();
		for (j = 0; j < jc; j++) fprintf(file, " %s/%s/%s_debug.o", ws->basedir, proj->source[j].path, proj->source[j].name);
		fprintf(file, "\r\n");

		// linker command
		if (proj->type == epExecutable) fprintf(file, "\t$(CXX) $(LDFLAGS_DEBUG) $(LDLIBS) -o %s/%s/%s_debug $(RPATHFLAG)", ws->basedir, proj->path, proj->name);
		else if (proj->type == epLibrary) {
			fprintf(file, "\t$(CXX) $(LD_SHARED) $(LDFLAGS_DEBUG) $(SONAMEFLAG_D) $(LDLIBS) -o %s/lib/lib%s_debug${LIBFULLEXT}", ws->basedir, proj->name);
		}
		jc = proj->source.size();
		for (j = 0; j < jc; j++) fprintf(file, " %s/%s/%s_debug.o", ws->basedir, proj->source[j].path, proj->source[j].name);
		jc = proj->lib.size();
		for (j = 0; j < jc; j++)
		{
			Library* lib = findLibrary(proj->lib[j]);
			if (lib != NULL) fprintf(file, " -L%s/lib -l%s_debug", ws->basedir, lib->name);
			else fprintf(file, " -l%s", proj->lib[j]);
		}
		jc = proj->libpath.size();
		for (j = 0; j < jc; j++) fprintf(file, " -L%s/lib ", ws->basedir);
		if (proj->type == epLibrary) {
			fprintf(file, "\r\n\tchmod a+r %s/lib/lib%s_debug${LIBFULLEXT}", ws->basedir, proj->name);
			fprintf(file, "\r\n\t(cd %s/lib; ln -fs lib%s_debug${LIBFULLEXT} lib%s_debug${LIBEXT})", ws->basedir, proj->name, proj->name);
			fprintf(file, "\r\n\t(cd %s/lib; ln -fs lib%s_debug${LIBFULLEXT} lib%s_debug${LIBSONAMEEXT})", ws->basedir, proj->name, proj->name);
		}
		fprintf(file, "\r\n\r\n");
	}

	// clean command
	fprintf(file, "clean:\r\n\t-rm -f");
	for (i = 0; i < ic; i++)
	{
		Project* proj = ws->project[i];
		jc = proj->source.size();
		for (j = 0; j < jc; j++) fprintf(file, " %s/%s/%s.o", ws->basedir, proj->source[j].path, proj->source[j].name);
		for (j = 0; j < jc; j++) fprintf(file, " %s/%s/%s_debug.o", ws->basedir, proj->source[j].path, proj->source[j].name);
		if (proj->type == epExecutable)
		{
			fprintf(file, " %s/%s/%s", ws->basedir, proj->path, proj->name);
			fprintf(file, " %s/%s/%s_debug", ws->basedir, proj->path, proj->name);
		}
		else if (proj->type == epLibrary)
		{
			for (j = 0; j < jc; j++) fprintf(file, " %s/%s/%s_debug.o", ws->basedir, proj->source[j].path, proj->source[j].name);
			fprintf(file, " %s/lib/lib%s${LIBEXT}", ws->basedir, proj->name);
			fprintf(file, " %s/lib/lib%s_debug${LIBEXT}", ws->basedir, proj->name);
		}
	}
	fprintf(file, "\r\n");

	fclose(file);
}

// Generate a developer studio workspace file
void GenerateWorkspaceFile(Workspace* ws)
{
	char filename[512];
	sprintf(filename, "%s/%s.dsw", ws->path, WindowsPath(ws->name));

	FILE* file = fopen(filename, "w+");
	if (file == NULL)
	{
		printf("        ERROR opening file %s for writing\n", filename);
		return;
	}

	fprintf(file, "Microsoft Developer Studio Workspace File, Format Version 6.00\r\n# WARNING: DO NOT EDIT OR DELETE THIS WORKSPACE FILE!\r\n\r\n###############################################################################\r\n\r\n");

	int i, ic = ws->project.size();
	for (i = 0; i < ic; i++)
	{
		Project* proj = ws->project[i];
		if ((proj->system & esWindows) == 0) continue;
		fprintf(file, "Project: \"%s\"=\"%s\\%s\\%s.dsp\" - Package Owner=<4>\r\n\r\nPackage=<5>\r\n{{{\r\n}}}\r\n\r\nPackage=<4>\r\n{{{\r\n}}}\r\n\r\n###############################################################################\r\n\r\n", TokenName(proj->name), WindowsPath(ws->basedir), WindowsPath(proj->path), proj->name);
	}

	fprintf(file, "Global:\r\n\r\nPackage=<5>\r\n{{{\r\n}}}\r\n\r\nPackage=<3>\r\n{{{\r\n}}}\r\n\r\n###############################################################################\r\n\r\n");

	fclose(file);
}

// Generate a developer studio project file
void GenerateProjectFile(Project* proj)
{
	if ((proj->system & esWindows) == 0) return;
	if (proj->type == epExecutableQt) return;			// NOT SUPPORTED YET!

	char name[256];
	char path[256];
	char base[256];
	strcpy(name, TokenName(proj->name));
	strcpy(path, WindowsPath(proj->path));

	char filename[512];
	sprintf(filename, "%s/%s.dsp", proj->path, proj->name);
	strcpy(base, WindowsPath(proj->workspace->basedir));
	int i, ic;

	FILE* file = fopen(filename, "w+");
	if (file == NULL)
	{
		printf("        ERROR opening file %s for writing\n", filename);
		return;
	}

	fprintf(file, "# Microsoft Developer Studio Project File - Name=\"%s\" - Package Owner=<4>\r\n", name);
	fprintf(file, "# Microsoft Developer Studio Generated Build File, Format Version 6.00\r\n");
	fprintf(file, "# ** NICHT BEARBEITEN **\r\n");
	fprintf(file, "\r\n");
	if (proj->type == epExecutable) fprintf(file, "# TARGTYPE \"Win32 (x86) Console Application\" 0x0103\r\n");
	else  if (proj->type == epLibrary) fprintf(file, "# TARGTYPE \"Win32 (x86) Static Library\" 0x0104\r\n");
	fprintf(file, "\r\n");
	fprintf(file, "CFG=%s - Win32 Debug\r\n", name);
	fprintf(file, "!MESSAGE Dies ist kein g�ltiges Makefile. Zum Erstellen dieses Projekts mit NMAKE\r\n");
	fprintf(file, "!MESSAGE verwenden Sie den Befehl \"Makefile exportieren\" und f�hren Sie den Befehl\r\n");
	fprintf(file, "!MESSAGE \r\n");
	fprintf(file, "!MESSAGE NMAKE /f \"%s.mak\".\r\n", name);
	fprintf(file, "!MESSAGE \r\n");
	fprintf(file, "!MESSAGE Sie k�nnen beim Ausf�hren von NMAKE eine Konfiguration angeben\r\n");
	fprintf(file, "!MESSAGE durch Definieren des Makros CFG in der Befehlszeile. Zum Beispiel:\r\n");
	fprintf(file, "!MESSAGE \r\n");
	fprintf(file, "!MESSAGE NMAKE /f \"%s.mak\" CFG=\"%s - Win32 Debug\"\r\n", name, name);
	fprintf(file, "!MESSAGE \r\n");
	fprintf(file, "!MESSAGE F�r die Konfiguration stehen zur Auswahl:\r\n");
	fprintf(file, "!MESSAGE \r\n");
	if (proj->type == epExecutable)
	{
		fprintf(file, "!MESSAGE \"%s - Win32 Release\" (basierend auf  \"Win32 (x86) Console Application\")\r\n", name);
		fprintf(file, "!MESSAGE \"%s - Win32 Debug\" (basierend auf  \"Win32 (x86) Console Application\")\r\n", name);
	}
	else  if (proj->type == epLibrary)
	{
		fprintf(file, "!MESSAGE \"%s - Win32 Release\" (basierend auf  \"Win32 (x86) Static Library\")\r\n", name);
		fprintf(file, "!MESSAGE \"%s - Win32 Debug\" (basierend auf  \"Win32 (x86) Static Library\")\r\n", name);
	}
	fprintf(file, "!MESSAGE \r\n");
	fprintf(file, "\r\n# Begin Project\r\n# PROP AllowPerConfigDependencies 0\r\n# PROP Scc_ProjName \"\"\r\n# PROP Scc_LocalPath \"\"\r\nCPP=cl.exe\r\nRSC=rc.exe\r\n\r\n");

	if (proj->type == epExecutable)
	{
		fprintf(file, "!IF  \"$(CFG)\" == \"%s - Win32 Release\"\r\n", name);
		fprintf(file, "\r\n");
		fprintf(file, "# PROP BASE Use_MFC 0\r\n");
		fprintf(file, "# PROP BASE Use_Debug_Libraries 0\r\n");
		fprintf(file, "# PROP BASE Output_Dir \"Release\"\r\n");
		fprintf(file, "# PROP BASE Intermediate_Dir \"Release\"\r\n");
		fprintf(file, "# PROP BASE Target_Dir \"\"\r\n");
		fprintf(file, "# PROP Use_MFC 0\r\n");
		fprintf(file, "# PROP Use_Debug_Libraries 0\r\n");
		fprintf(file, "# PROP Output_Dir \".\\WinNT\\Release\"\r\n");
		fprintf(file, "# PROP Intermediate_Dir \"%s\\win_tmp\\Release\\%s\"\r\n", base, proj->name);
		fprintf(file, "# PROP Ignore_Export_Lib 0\r\n");
		fprintf(file, "# PROP Target_Dir \"\"\r\n");
		fprintf(file, "# ADD BASE CPP /nologo /w /W0 /GX /O2 /D \"WIN32\" /D \"NDEBUG\" /D \"_CONSOLE\" /D \"_MBCS\" /YX /FD /c\r\n");
		fprintf(file, "# ADD CPP /nologo /MD /w /W0 /GR /GX /O2 /I \"%s\\include\" /D \"WIN32\" /D \"NDEBUG\" /D \"_CONSOLE\" /D \"_MBCS\" /YX /FD /c\r\n", base);
		fprintf(file, "# ADD BASE RSC /l 0x407 /d \"NDEBUG\"\r\n");
		fprintf(file, "# ADD RSC /l 0x407 /d \"NDEBUG\"\r\n");
		fprintf(file, "BSC32=bscmake.exe\r\n");
		fprintf(file, "# ADD BASE BSC32 /nologo\r\n");
		fprintf(file, "# ADD BSC32 /nologo\r\n");
		fprintf(file, "LINK32=link.exe\r\n");
// 		fprintf(file, "# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386\r\n");
		fprintf(file, "# ADD BASE LINK32 /nologo /subsystem:console /machine:I386\r\n");
		fprintf(file, "# ADD LINK32");
		ic = proj->lib.size();
		for (i = 0; i < ic; i++) fprintf(file, " \"%s.lib\"", WindowsPath(proj->lib[i]));
// 		fprintf(file, " kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386 /libpath:\"%s\\lib\\WinNT\\Release\"\r\n", base);
        fprintf(file, " /nologo /subsystem:console /machine:I386 /libpath:\"%s\\lib\\WinNT\\Release\"\r\n", base);
		fprintf(file, "\r\n");
		fprintf(file, "!ELSEIF  \"$(CFG)\" == \"%s - Win32 Debug\"\r\n", name);
		fprintf(file, "\r\n");
		fprintf(file, "# PROP BASE Use_MFC 0\r\n");
		fprintf(file, "# PROP BASE Use_Debug_Libraries 1\r\n");
		fprintf(file, "# PROP BASE Output_Dir \"Debug\"\r\n");
		fprintf(file, "# PROP BASE Intermediate_Dir \"Debug\"\r\n");
		fprintf(file, "# PROP BASE Target_Dir \"\"\r\n");
		fprintf(file, "# PROP Use_MFC 0\r\n");
		fprintf(file, "# PROP Use_Debug_Libraries 1\r\n");
		fprintf(file, "# PROP Output_Dir \".\\WinNT\\Debug\"\r\n");
		fprintf(file, "# PROP Intermediate_Dir \"%s\\win_tmp\\Debug\%s\"\r\n", base, proj->name);
		fprintf(file, "# PROP Ignore_Export_Lib 0\r\n");
		fprintf(file, "# PROP Target_Dir \"\"\r\n");
		fprintf(file, "# ADD BASE CPP /nologo /w /W0 /Gm /GX /ZI /Od /D \"WIN32\" /D \"_DEBUG\" /D \"_CONSOLE\" /D \"_MBCS\" /YX /FD /GZ /c\r\n");
        fprintf(file, "# ADD CPP /nologo /MDd /w /W0 /GR /GX /Z7 /Od /I \"%s\\include\" /D \"WIN32\" /D \"_DEBUG\" /D \"_CONSOLE\" /D \"_MBCS\" /YX /FD /GZ /c\r\n", base);
		fprintf(file, "# ADD BASE RSC /l 0x407 /d \"_DEBUG\"\r\n");
		fprintf(file, "# ADD RSC /l 0x407 /d \"_DEBUG\"\r\n");
		fprintf(file, "BSC32=bscmake.exe\r\n");
		fprintf(file, "# ADD BASE BSC32 /nologo\r\n");
		fprintf(file, "# ADD BSC32 /nologo\r\n");
		fprintf(file, "LINK32=link.exe\r\n");
		// actually this line should have no effect because it's commented out ...
		// but somehow it seems to have!
        // fprintf(file, "# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept\r\n");
        fprintf(file, "# ADD BASE LINK32 /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept\r\n");
		fprintf(file, "# ADD LINK32");
		ic = proj->lib.size();
		for (i = 0; i < ic; i++) fprintf(file, " \"%s.lib\"", WindowsPath(proj->lib[i]));
		// see above
        // fprintf(file, " kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept /libpath:\"%s\\lib\\WinNT\\Debug\"\r\n", base);
        fprintf(file, " /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept /libpath:\"%s\\lib\\WinNT\\Debug\"\r\n", base);
		fprintf(file, "\r\n");
		fprintf(file, "!ENDIF \r\n");
	}
	else if (proj->type == epLibrary)
	{
		fprintf(file, "!IF  \"$(CFG)\" == \"%s - Win32 Release\"\r\n", name);
		fprintf(file, "\r\n");
		fprintf(file, "# PROP BASE Use_MFC 0\r\n");
		fprintf(file, "# PROP BASE Use_Debug_Libraries 0\r\n");
		fprintf(file, "# PROP BASE Output_Dir \"Release\"\r\n");
		fprintf(file, "# PROP BASE Intermediate_Dir \"Release\"\r\n");
		fprintf(file, "# PROP BASE Target_Dir \"\"\r\n");
		fprintf(file, "# PROP Use_MFC 0\r\n");
		fprintf(file, "# PROP Use_Debug_Libraries 0\r\n");
		fprintf(file, "# PROP Output_Dir \"%s\\lib\\WinNT\\Release\"\r\n", base);
		fprintf(file, "# PROP Intermediate_Dir \"%s\\win_tmp\\Release\\%s\"\r\n", base, proj->name);
		fprintf(file, "# PROP Target_Dir \"\"\r\n");
		fprintf(file, "# ADD BASE CPP /nologo /W3 /GX /O2 /D \"WIN32\" /D \"NDEBUG\" /D \"_MBCS\" /D \"_LIB\" /YX /FD /c\r\n");
		fprintf(file, "# ADD CPP /nologo /MD /W3 /GR /GX /O2 /I \"%s\\include\" /D \"WIN32\" /D \"NDEBUG\" /D \"_MBCS\" /D \"_LIB\" /YX /FD /c\r\n", base);
		fprintf(file, "# ADD BASE RSC /l 0x407 /d \"NDEBUG\"\r\n");
		fprintf(file, "# ADD RSC /l 0x407 /d \"NDEBUG\"\r\n");
		fprintf(file, "BSC32=bscmake.exe\r\n");
		fprintf(file, "# ADD BASE BSC32 /nologo\r\n");
		fprintf(file, "# ADD BSC32 /nologo\r\n");
		fprintf(file, "LIB32=link.exe -lib\r\n");
		fprintf(file, "# ADD BASE LIB32 /nologo\r\n");
		fprintf(file, "# ADD LIB32 /nologo\r\n");
		fprintf(file, "\r\n");
		fprintf(file, "!ELSEIF  \"$(CFG)\" == \"%s - Win32 Debug\"\r\n", name);
		fprintf(file, "\r\n");
		fprintf(file, "# PROP BASE Use_MFC 0\r\n");
		fprintf(file, "# PROP BASE Use_Debug_Libraries 1\r\n");
		fprintf(file, "# PROP BASE Output_Dir \"Debug\"\r\n");
		fprintf(file, "# PROP BASE Intermediate_Dir \"Debug\"\r\n");
		fprintf(file, "# PROP BASE Target_Dir \"\"\r\n");
		fprintf(file, "# PROP Use_MFC 0\r\n");
		fprintf(file, "# PROP Use_Debug_Libraries 1\r\n");
		fprintf(file, "# PROP Output_Dir \"%s\\lib\\WinNT\\Debug\"\r\n", base);
		fprintf(file, "# PROP Intermediate_Dir \"%s\\win_tmp\\Debug\\%s\"\r\n", base, proj->name);
		fprintf(file, "# PROP Target_Dir \"\"\r\n");
		fprintf(file, "# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D \"WIN32\" /D \"_DEBUG\" /D \"_MBCS\" /D \"_LIB\" /YX /FD /GZ /c\r\n");
		fprintf(file, "# ADD CPP /nologo /MDd /W3 /GR /GX /Z7 /Od /I \"%s\\include\" /D \"WIN32\" /D \"_DEBUG\" /D \"_MBCS\" /D \"_LIB\" /YX /FD /GZ /c\r\n", base);
		fprintf(file, "# ADD BASE RSC /l 0x407 /d \"_DEBUG\"\r\n");
		fprintf(file, "# ADD RSC /l 0x407 /d \"_DEBUG\"\r\n");
		fprintf(file, "BSC32=bscmake.exe\r\n");
		fprintf(file, "# ADD BASE BSC32 /nologo\r\n");
		fprintf(file, "# ADD BSC32 /nologo\r\n");
		fprintf(file, "LIB32=link.exe -lib\r\n");
		fprintf(file, "# ADD BASE LIB32 /nologo\r\n");
		fprintf(file, "# ADD LIB32 /nologo\r\n");
		fprintf(file, "\r\n");
		fprintf(file, "!ENDIF \r\n");
	}

	fprintf(file, "\r\n# Begin Target\r\n\r\n# Name \"%s - Win32 Release\"\r\n# Name \"%s - Win32 Debug\"\r\n# Begin Group \"Quellcodedateien\"\r\n\r\n# PROP Default_Filter \"cpp;c;cxx;rc;def;r;odl;idl;hpj;bat\"\r\n", name, name);
	ic = proj->source.size();
	for (i = 0; i < ic; i++)
	{
		fprintf(file, "# Begin Source File\r\n\r\nSOURCE=\"%s\\%s\\%s%s\"\r\n# End Source File\r\n", base, WindowsPath(proj->source[i].path), proj->source[i].name, proj->source[i].extension);
	}
	fprintf(file, "# End Group\r\n# Begin Group \"Header-Dateien\"\r\n\r\n# PROP Default_Filter \"h;hpp;hxx;hm;inl\"\r\n");
	ic = proj->header.size();
	for (i = 0; i < ic; i++)
	{
		fprintf(file, "# Begin Source File\r\n\r\nSOURCE=\"%s\\%s\\%s%s\"\r\n# End Source File\r\n", base, WindowsPath(proj->header[i].path), proj->header[i].name, proj->header[i].extension);
	}
	fprintf(file, "# End Group\r\n# End Target\r\n# End Project\r\n");

	fclose(file);
}

// Generate a qmake project file
void GenerateQmakeProject(Project* proj)
{
	if (proj->type != epExecutableQt) return;

	char filename[512];
	sprintf(filename, "%s/%s.pro", proj->path, proj->name);
	int i, ic;

	FILE* file = fopen(filename, "w+");
	if (file == NULL)
	{
		printf("        ERROR opening file %s for writing\n", filename);
		return;
	}
	WriteCommentHeader(file);

	fprintf(file, "#\r\n");
	fprintf(file, "# This is a QT project file.\r\n");
	fprintf(file, "# Use qmake to generate a\r\n");
	fprintf(file, "# platform specific makefile.\r\n");
	fprintf(file, "#\r\n");

	fprintf(file, "SHARKHOME = %s\r\n", proj->workspace->basedir);
	fprintf(file, "CONFIG = qt console warn_on\r\n");
	fprintf(file, "INCLUDEPATH = . $${SHARKHOME}/include\r\n");

	fprintf(file, "win32{\r\n");
	fprintf(file, "\tLIBS = -L$${SHARKHOME}/lib/winnt/release -lshark\r\n");
	fprintf(file, "}\r\n");

	fprintf(file, "!win32{\r\n");
	fprintf(file, "\tLIBS = -L$${SHARKHOME}/lib/ -lshark\r\n");
	fprintf(file, "}\r\n");
	fprintf(file, "ARCH = i686\r\n");

	ic = proj->source.size();
	if (ic > 0)
	{
		fprintf(file, "SOURCES =");
		for (i = 0; i < ic; i++)
		{
			fprintf(file, " %s/%s/%s%s", proj->workspace->basedir, proj->source[i].path, proj->source[i].name, proj->source[i].extension);
		}
		fprintf(file, "\r\n");
	}

	ic = proj->header.size();
	if (ic > 0)
	{
		fprintf(file, "HEADERS =");
		for (i = 0; i < ic; i++)
		{
			fprintf(file, " %s/%s/%s%s", proj->workspace->basedir, proj->header[i].path, proj->header[i].name, proj->header[i].extension);
		}
		fprintf(file, "\r\n");
	}
	fprintf(file, "TARGET = %s\r\n", proj->name);

	fclose(file);
}

// Generate the main Makefile.in
void GenerateMainMakefile(const std::vector<Workspace*>& workspace)
{
	char filename[512];
	sprintf(filename, "Makefile.in");

	FILE* file = fopen(filename, "w+");
	if (file == NULL)
	{
		printf("        ERROR opening file %s for writing\n", filename);
		return;
	}
	WriteCommentHeader(file);

	fprintf(file, "\r\n");
	fprintf(file, "ARCH_MACH = @ARCH_MACH@\r\n");
	fprintf(file, "HOST = @HOST@\r\n");
	fprintf(file, "CFLAGS = @CFLAGS@\r\n");
	fprintf(file, "DEFS = @DEFS@\r\n");
	fprintf(file, "LD_SHARED = @LD_SHARED@\r\n");
	fprintf(file, "LIBEXT = @LIBEXT@\r\n");
	fprintf(file, "LDLIBS = @LIBS@\r\n");
	fprintf(file, "PWD = $(shell pwd)\r\n");
	fprintf(file, "FLAG_PVM = @FLAG_PVM@\r\n");
	fprintf(file, "LIBS_PVM = @LIBS_PVM@\r\n");
	fprintf(file, "PVM_LDLIBS = @PVM_LDLIBS@\r\n");

	fprintf(file, "prefix = @prefix@\r\n");
	fprintf(file, "exec_prefix = @exec_prefix@\r\n");
	fprintf(file, "includedir = @includedir@\r\n");
	fprintf(file, "libdir = @libdir@\r\n");
	fprintf(file, "datarootdir = @datarootdir@\r\n");
	fprintf(file, "\r\n\r\n");

	fprintf(file, "info:\r\n");
	fprintf(file, "\t@echo \"\"\r\n");
	fprintf(file, "\t@echo \"Shark Main Makefile - List of Commands and Directories\"\r\n");
	fprintf(file, "\t@echo \"------------------------------------------------------\"\r\n");
	fprintf(file, "\t@echo \"\"\r\n");
	fprintf(file, "\t@echo \"all:             makes 'libs', 'debug', 'install', and 'examples'\"\r\n");
	fprintf(file, "\t@echo \"libs:            builds the shark library (in release mode)\"\r\n");
	fprintf(file, "\t@echo \"debug:           builds the shark library (in debug mode)\"\r\n");
	fprintf(file, "\t@echo \"install:         installs the libraries, the headers, and the documentation\"\r\n");
	fprintf(file, "\t@echo \"installlibs:     installs the libraries\"\r\n");
	fprintf(file, "\t@echo \"installdocs:     installs the documentation\"\r\n");
	fprintf(file, "\t@echo \"installheaders:  installs the header files\"\r\n");
	fprintf(file, "\t@echo \"examples:        builds example programs, make 'libs' and 'install' first\"\r\n");
	fprintf(file, "\t@echo \"clean:           removes all objects, libraries, and executable examples\"\r\n");
	fprintf(file, "\t@echo \"\"\r\n");
	fprintf(file, "\t@echo \"libs are installed to    $(libdir)\"\r\n");
	fprintf(file, "\t@echo \"headers are installed to $(includedir)\"\r\n");
	fprintf(file, "\t@echo \"docs are installed to    @docdir@\"\r\n");
	fprintf(file, "\t@echo \"\"\r\n\r\n");
 
	int i, ic = workspace.size();
	int j, jc;
	int k, kc;

	// libs command
	fprintf(file, "libs:");
	for (i = 0; i < ic; i++)
	{
		Workspace* ws = workspace[i];
		jc = ws->project.size();
		for (j = 0; j < jc; j++)
		{
			Project* proj = ws->project[j];
			if (proj->type == epLibrary)
			{
				fprintf(file, " %s", proj->name);
			}
		}
	}
	fprintf(file, "\r\n\r\n");

	// debug command
	fprintf(file, "debug:");
	for (i = 0; i < ic; i++)
	{
		Workspace* ws = workspace[i];
		jc = ws->project.size();
		for (j = 0; j < jc; j++)
		{
			Project* proj = ws->project[j];
			if (proj->type == epLibrary)
			{
				fprintf(file, " %s-debug", proj->name);
			}
		}
	}
	fprintf(file, "\r\n\r\n");

	// examples command
	fprintf(file, "examples: build-ex\r\n\r\nbuild-ex:\r\n");
	fprintf(file, "\t@echo \"\"\r\n");
	fprintf(file, "\t@echo \"### Building Examples\"\r\n");
	for (i = 0; i < ic; i++)
	{
		Workspace* ws = workspace[i];
		jc = ws->project.size();
		for (j = 0; j < jc; j++)
		{
			Project* proj = ws->project[j];
			if ((proj->system & esMakefile) == 0) continue;
			if (proj->type == epExecutable)
			{
				fprintf(file, "\t@( if test -d %s; then echo \"    example %s\"; cd %s; $(MAKE) -s Project-%s; fi )\r\n", proj->path, proj->name, proj->path, proj->name);
			}
		}
	}
	fprintf(file, "\t@echo \"### Examples built.\"\r\n");
	fprintf(file, "\r\n");

	// all command
	fprintf(file, "all: libs debug install examples\r\n\r\n");

	// install command
	fprintf(file, "install: installheaders installdocs installlibs\r\n\r\n");

	// install libs
	fprintf(file, "installlibs: libs\r\n");
	fprintf(file, "\t@-mkdir -p $(libdir)\r\n");
	fprintf(file, "\t@-cp -fP lib/lib*  $(libdir)  > /dev/null 2>&1 & echo \"copying libraries to $(libdir)\" \r\n");
	fprintf(file, "\t@echo \"remember to make the path to the libraries known to executables\"\r\n\r\n");

	// install includes
	fprintf(file, "installheaders:\r\n");
	fprintf(file, "\t@-mkdir -p $(includedir)\r\n");
	fprintf(file, "\t@-cp -rf include/*  $(includedir)  > /dev/null 2>&1 & echo \"copying include files to $(includedir)\"\r\n");
	fprintf(file, "\t@-chmod -f -R a+rX  $(includedir)\r\n\r\n");

	// install documentation
	fprintf(file, "installdocs:\r\n");
	fprintf(file, "\t@-mkdir -p @docdir@\r\n");
	fprintf(file, "\t@-cp -rf doc/*  @docdir@  > /dev/null 2>&1 & echo \"copying documentation files to @docdir@\" \r\n");
	fprintf(file, "\t@-chmod -f -R a+rX @docdir@ \r\n\r\n");

	// install examples
	//fprintf(file, "\t@echo \"copying example files to  $(docdir)\"\r\n");
	//fprintf(file, "\tcp -rf examples/*  $(datarootdir) \r\n");
	//fprintf(file, "\tchmod -f -R a+rX $(docdir) \r\n");

	// clean command
	fprintf(file, "clean:\r\n");
	for (i = 0; i < ic; i++)
	{
		Workspace* ws = workspace[i];
		if (! ws->onlyQT)
		{
			fprintf(file, "\t@( if test -d %s; then cd %s; $(MAKE) -s clean; fi )\r\n", ws->path, ws->path);
		}
	}
	fprintf(file, "\t@rm -fR win_tmp *~ *.ncb *.opt *.log\r\n");
	fprintf(file, "\r\n");

	// libraries release and debug commands
	for (i = 0; i < ic; i++)
	{
		Workspace* ws = workspace[i];
		jc = ws->project.size();
		for (j = 0; j < jc; j++)
		{
			Project* proj = ws->project[j];
			if ((proj->system & esMakefile) == 0) continue;
			if (proj->type == epLibrary)
			{
				kc = proj->lib.size();

				fprintf(file, "%s:\r\n", proj->name);
				fprintf(file, "\t@( if test -d %s; then echo \"\"; echo \"### Building %s library \"; cd %s; $(MAKE) -s; echo \"### %s library built.\"; fi )\r\n\r\n", proj->path, proj->name, proj->path, proj->name);

				fprintf(file, "%s-debug:\r\n", proj->name);
				fprintf(file, "\t@( if test -d %s; then echo \"\"; echo \"### Building %s library (debug)\"; cd %s; $(MAKE) -s debug; echo \"### %s library (debug) built.\"; fi )\r\n\r\n", proj->path, proj->name, proj->path, proj->name);
			}
		}
	}

	fclose(file);
}

// Generate the main SharkAll.dsw and SharkLibs.dsw files
void GenerateMainWorkspace(const std::vector<Workspace*>& workspace)
{
	char filename[512];
	FILE* file;
	int i, ic = workspace.size();
	int j, jc;
	int k, kc;

	// generate a list of all projects,
	// with libraries first
	int libs = 0;
	std::vector<Project*> projects;
	for (i = 0; i < ic; i++)
	{
		Workspace* ws = workspace[i];
		jc = ws->project.size();
		for (j = 0; j < jc; j++)
		{
			Project* proj = ws->project[j];
			if ((proj->system & esWindows) == 0) continue;
			if (proj->type == epLibrary)
			{
				projects.insert(projects.begin(), proj);
				libs++;
			}
			else if (proj->type == epExecutable)
			{
				projects.push_back(proj);
			}
		}
	}

	// reorder the libs to resolve dependencies
	bool changed;
	do
	{
		changed = false;
		for (i = 0; i < libs; i++)
		{
			Project* proj = projects[i];
			jc = proj->lib.size();
			for (j = 0; j < jc; j++)
			{
				Library* lib = findLibrary(proj->lib[j]);
				if (lib == NULL) continue;
				bool found = false;
				for (k = 0; k < i; k++)
				{
					if (strcmp(projects[k]->name, lib->name) == 0)
					{
						found = true;
						break;
					}
				}
				if (! found)
				{
					// swap
					if (i == libs - 1)
					{
						std::cerr << "SharkConfig: IRRESOLVABLE DEPENDENCY when generating main workspace\n\n" << std::endl;
						exit(EXIT_FAILURE);
					}
					projects[i] = projects[i+1];
					projects[i+1] = proj;
					changed = true;
					break;
				}
			}
		}
	}
	while (changed);

	sprintf(filename, "Shark_Libs.dsw");
	file = fopen(filename, "w+");
	if (file == NULL)
	{
		printf("        ERROR opening file %s for writing\n", filename);
		return;
	}
	fprintf(file, "Microsoft Developer Studio Workspace File, Format Version 6.00\r\n# WARNING: DO NOT EDIT OR DELETE THIS WORKSPACE FILE!\r\n\r\n###############################################################################\r\n\r\n");
	for (i = 0; i < libs; i++)
	{
		Project* proj = projects[i];
		jc = proj->lib.size();
		fprintf(file, "Project: \"%s\"=\"%s\\%s.dsp\" - Package Owner=<4>\r\n\r\nPackage=<5>\r\n{{{\r\n", TokenName(proj->name), WindowsPath(proj->path), proj->name);
		for (j = 0; j < jc; j++)
		{
			Library* lib = findLibrary(proj->lib[j]);
			if (lib == NULL) continue;
			fprintf(file, "    Begin Project Dependency\r\n    Project_Dep_Name %s\r\n    End Project Dependency\r\n", TokenName(lib->name));
		}
		fprintf(file, "}}}\r\n\r\nPackage=<4>\r\n{{{\r\n");
		for (j = 0; j < jc; j++)
		{
			Library* lib = findLibrary(proj->lib[j]);
			if (lib == NULL) continue;
			fprintf(file, "    Begin Project Dependency\r\n    Project_Dep_Name %s\r\n    End Project Dependency\r\n", TokenName(lib->name));
		}
		fprintf(file, "}}}\r\n\r\n###############################################################################\r\n\r\n");
	}
	fprintf(file, "Global:\r\n\r\nPackage=<5>\r\n{{{\r\n}}}\r\n\r\nPackage=<3>\r\n{{{\r\n}}}\r\n\r\n###############################################################################\r\n\r\n");
	fclose(file);

	sprintf(filename, "Shark_All.dsw");
	file = fopen(filename, "w+");
	if (file == NULL)
	{
		printf("        ERROR opening file %s for writing\n", filename);
		return;
	}
	fprintf(file, "Microsoft Developer Studio Workspace File, Format Version 6.00\r\n# WARNING: DO NOT EDIT OR DELETE THIS WORKSPACE FILE!\r\n\r\n###############################################################################\r\n\r\n");
	ic = projects.size();
	for (i = 0; i < ic; i++)
	{
		Project* proj = projects[i];
		jc = proj->lib.size();
		fprintf(file, "Project: \"%s\"=\"%s\\%s.dsp\" - Package Owner=<4>\r\n\r\nPackage=<5>\r\n{{{\r\n", TokenName(proj->name), WindowsPath(proj->path), proj->name);
		for (j = 0; j < jc; j++)
		{
			Library* lib = findLibrary(proj->lib[j]);
			if (lib == NULL) continue;
			fprintf(file, "    Begin Project Dependency\r\n    Project_Dep_Name %s\r\n    End Project Dependency\r\n", TokenName(lib->name));
		}
		fprintf(file, "}}}\r\n\r\nPackage=<4>\r\n{{{\r\n");
		for (j = 0; j < jc; j++)
		{
			Library* lib = findLibrary(proj->lib[j]);
			if (lib == NULL) continue;
			fprintf(file, "    Begin Project Dependency\r\n    Project_Dep_Name %s\r\n    End Project Dependency\r\n", TokenName(lib->name));
		}
		fprintf(file, "}}}\r\n\r\n###############################################################################\r\n\r\n");
	}
	fprintf(file, "Global:\r\n\r\nPackage=<5>\r\n{{{\r\n}}}\r\n\r\nPackage=<3>\r\n{{{\r\n}}}\r\n\r\n###############################################################################\r\n\r\n");
	fclose(file);
}

// Generate the CMakeLists.txt files
void GenerateCMakeLists(const std::vector<Workspace*>& workspace)
{
	char filename[512];
	FILE* file;

	// GENERATE MAIN LIST FILE
	mkdir("cmake", (mode_t)-1);
	sprintf(filename, "cmake/CMakeLists.txt");
	file = fopen(filename, "w+");
	if (file == NULL)
	{
		printf("        ERROR opening file %s for writing\n", filename);
		return;
	}
	WriteCommentHeader(file);

	fprintf(file, "\r\n");
	fprintf(file, "PROJECT( shark )\r\n");
	fprintf(file, "\r\n");
	fprintf(file, "CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )\r\n");
	fprintf(file, "\r\n");
	fprintf(file, "SET( LIBRARY_OUTPUT_PATH ../lib )\r\n");
	fprintf(file, "\r\n");
	fprintf(file, "INCLUDE_DIRECTORIES( ../include )\r\n");
	fprintf(file, "\r\n");

	int i, ic = workspace.size();
	int j, jc;
	int k, kc;

	// define sets
	for (i = 0; i < ic; i++)
	{
		Workspace* ws = workspace[i];
		jc = ws->project.size();
		for (j = 0; j < jc; j++)
		{
			Project* proj = ws->project[j];
			if (proj->type == epLibrary)
			{
				fprintf(file, "SET( %s_SRC\r\n", TokenName(proj->name));
				kc = proj->source.size();
				for (k = 0; k < kc; k++)
				{
					fprintf(file, "\t../%s/%s%s\r\n", proj->source[k].path, proj->source[k].name, proj->source[k].extension);
				}
				fprintf(file, ")\r\n");
			}
		}
	}
	fprintf(file, "\r\n");

	// define libraries
	for (i = 0; i < ic; i++)
	{
		Workspace* ws = workspace[i];
		jc = ws->project.size();
		for (j = 0; j < jc; j++)
		{
			Project* proj = ws->project[j];
			if (proj->type == epLibrary)
			{
				fprintf(file, "ADD_LIBRARY( %s SHARED ${%s_SRC} )\r\n", proj->name, TokenName(proj->name));
			}
		}
	}
	fprintf(file, "\r\n");

	// link to example programs
	fprintf(file, "OPTION( OPT_COMPILE_EXAMPLES \"Compile example programs.\" ON )\r\n");
	fprintf(file, "\r\n");
	fprintf(file, "IF( OPT_COMPILE_EXAMPLES )\r\n");
	fprintf(file, "\tADD_SUBDIRECTORY( ../examples ../examples )\r\n");
	fprintf(file, "ENDIF( OPT_COMPILE_EXAMPLES )\r\n");
	fprintf(file, "\r\n");

	fclose(file);


	// GENERATE EXAMPLES LIST FILE
	sprintf(filename, "examples/CMakeLists.txt");
	file = fopen(filename, "w+");
	if (file == NULL)
	{
		printf("        ERROR opening file %s for writing\n", filename);
		return;
	}
	WriteCommentHeader(file);

	fprintf(file, "\r\n");
	fprintf(file, "PROJECT( shark )\r\n");
	fprintf(file, "\r\n");
	fprintf(file, "CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )\r\n");
	fprintf(file, "\r\n");
	fprintf(file, "OPTION( ENABLE_QT_EXAMPLES \"\" ON )\r\n");
	fprintf(file, "\r\n");
	fprintf(file, "INCLUDE_DIRECTORIES( ../include )\r\n");
	fprintf(file, "\r\n");
	fprintf(file, "LINK_DIRECTORIES( ../lib )\r\n");
	fprintf(file, "\r\n");

	// define sets
	for (i = 0; i < ic; i++)
	{
		Workspace* ws = workspace[i];
		jc = ws->project.size();
		for (j = 0; j < jc; j++)
		{
			Project* proj = ws->project[j];
			if (proj->type == epExecutable)
			{
				fprintf(file, "SET( %s_SRC\r\n", TokenName(proj->name));
				kc = proj->source.size();
				for (k = 0; k < kc; k++)
				{
					fprintf(file, "\t../%s/%s%s\r\n", proj->source[k].path, proj->source[k].name, proj->source[k].extension);
				}
				fprintf(file, ")\r\n");
			}
		}
	}

	// define executables
	for (i = 0; i < ic; i++)
	{
		Workspace* ws = workspace[i];
		jc = ws->project.size();
		for (j = 0; j < jc; j++)
		{
			Project* proj = ws->project[j];
			if (proj->type == epExecutable)
			{
				fprintf(file, "\r\n");
				fprintf(file, "ADD_EXECUTABLE( ../%s/%s ${%s_SRC} )\r\n", proj->path, proj->name, TokenName(proj->name));
				fprintf(file, "TARGET_LINK_LIBRARIES( ../%s/%s", proj->path, proj->name);
				kc = proj->lib.size();
				for (k=0; k<kc; k++) fprintf(file, " %s", proj->lib[k]);
				fprintf(file, " )\r\n");
			}
		}
	}

	// QT examples
	fprintf(file, "\r\n");
	fprintf(file, "IF( ENABLE_QT_EXAMPLES )\r\n");
	fprintf(file, "\r\n");
	fprintf(file, "\tFIND_PACKAGE( Qt4 )\r\n");
	fprintf(file, "\tINCLUDE_DIRECTORIES(\r\n");
	fprintf(file, "\t\t../include\r\n");
	fprintf(file, "\t\t${QT_INCLUDE_DIR}\r\n");
	fprintf(file, "\t\t${QT_QTGUI_INCLUDE_DIR}\r\n");
	fprintf(file, "\t\t${QT_QTCORE_INCLUDE_DIR}\r\n");
	fprintf(file, "\t)\r\n");
	fprintf(file, "\tSET ( SHARK_QT_LIBS\r\n");
	fprintf(file, "\t\t${QT_QTCORE_LIBRARY}\r\n");
	fprintf(file, "\t\t${QT_QTGUI_LIBRARY}\r\n");
// 	fprintf(file, "\t\t${QT_QTOPENGL_LIBRARY}\r\n");
// 	fprintf(file, "\t\t${QT_QTNETWORK_LIBRARY}\r\n");
// 	fprintf(file, "\t\t${QT_QTXML_LIBRARY}\r\n");
// 	fprintf(file, "\t\t${QT_QTWEBKIT_LIBRARY}\r\n");
	fprintf(file, "\t)\r\n");
	fprintf(file, "\r\n");

	// define executables
	for (i = 0; i < ic; i++)
	{
		Workspace* ws = workspace[i];
		jc = ws->project.size();
		for (j = 0; j < jc; j++)
		{
			Project* proj = ws->project[j];
			if (proj->type == epExecutableQt)
			{
				fprintf(file, "\tSET (%s_H\r\n", TokenName(proj->name));
				kc = proj->header.size();
				for (k=0; k<kc; k++) fprintf(file, "\t\t../%s/%s%s\r\n", proj->header[k].path, proj->header[k].name, proj->header[k].extension);
				fprintf(file, "\t)\r\n");
				fprintf(file, "\tQT4_WRAP_CPP( %s_MOC_SRC ${%s_H} )\r\n", TokenName(proj->name), TokenName(proj->name));
				fprintf(file, "\r\n");

				fprintf(file, "\tSET (%s_SRC\r\n", TokenName(proj->name));
				fprintf(file, "\t\t${%s_MOC_SRC}\r\n", TokenName(proj->name));
				kc = proj->source.size();
				for (k=0; k<kc; k++) fprintf(file, "\t\t../%s/%s%s\r\n", proj->source[k].path, proj->source[k].name, proj->source[k].extension);
				fprintf(file, "\t)\r\n");
				fprintf(file, "\tADD_EXECUTABLE( ../%s/%s ${%s_SRC} )\r\n", proj->path, proj->name, TokenName(proj->name));
				fprintf(file, "\tTARGET_LINK_LIBRARIES( ../%s/%s", proj->path, proj->name);
				kc = proj->lib.size();
				for (k=0; k<kc; k++) fprintf(file, " %s", proj->lib[k]);
				fprintf(file, " ${SHARK_QT_LIBS} )\r\n");
				fprintf(file, "\r\n");
			}
		}
	}

	fprintf(file, "ENDIF( ENABLE_QT_EXAMPLES )\r\n");

	fclose(file);
}

int main(int argc, char** argv)
{
	int i, ic;
	int j, jc;
	std::vector<FileDesc> configfile;
	std::vector<Workspace*> workspace;
	int res;

	// read the central SharkConfig.Libs file
	{
		printf("reading library definitions\n"); fflush(stdout);
		FILE* file = fopen("SharkConfig.Libs", "r");
		XmlKnot* knot = ReadXmlKnot(file);
		fclose(file);
		libraries;
		if (knot == NULL)
		{
			printf("    XML parsing failed - no library definitions available!\n");
		}
		else
		{
			ic = knot->sub.size();
			for (i = 0; i < ic; i++)
			{
				Library* lib = CreateLibrary(knot->sub[i]);
				if (lib == NULL) printf("    Invalid library definition - skipped\n");
				else libraries.push_back(lib);
			}
		}
	}

	// collect all *.SharkConfig files
	printf("parsing directory tree for config files ..."); fflush(stdout);
	ScanDirectory(".", ".", "*.SharkConfig", configfile, true);
	printf(" done.\n");

	// read the configuration
	ic = configfile.size();
	for (i = 0; i < ic; i++)
	{
		char filename[512];
		sprintf(filename, "%s/%s%s", configfile[i].path, configfile[i].name, configfile[i].extension);
		printf("processing config file %s\n", filename);
		FILE* file = fopen(filename, "r");
		printf("    parsing XML structure\n");
		XmlKnot* knot = ReadXmlKnot(file);
		fclose(file);
		if (knot == NULL)
		{
			printf("                                        XML parsing failed!\n");
			continue;
		}
		printf("    creating project strucure and file lists\n");
		Workspace* ws = CreateWorkspace(configfile[i].path, knot);
		delete knot;
		if (ws == NULL)
		{
			printf("                                        structure extraction failed!\n");
			continue;
		}
		workspace.push_back(ws);
	}

	// output build system files
	ic = workspace.size();
	for (i = 0; i < ic; i++)
	{
		Workspace* ws = workspace[i];
		if (ws->onlyQT)
		{
			jc = ws->project.size();
			for (j = 0; j < jc; j++)
			{
				Project* proj = ws->project[j];
				if (proj->type == epExecutableQt)
				{
					printf("    generating %s.pro\n", ws->name);
					GenerateQmakeProject(proj);
				}
			}
		}
		else
		{
			printf("processing workspace %s\n", ws->name);
			printf("    generating Makefile.in\n");
			GenerateMakefile(ws);
			printf("    generating %s.dsw\n", ws->name);
			GenerateWorkspaceFile(ws);
			jc = ws->project.size();
			for (j = 0; j < jc; j++)
			{
				Project* proj = ws->project[j];
				if (proj->type == epExecutableQt)
				{
					printf("    generating %s.pro\n", ws->name);
					GenerateQmakeProject(proj);
				}
				else
				{
					printf("    generating %s.dsp\n", proj->name);
					GenerateProjectFile(proj);
				}
			}
		}
	}

	printf("generating main makefile.in\n");
	GenerateMainMakefile(workspace);

	printf("generating main workspace files\n");
	GenerateMainWorkspace(workspace);

	printf("generating cmake list files\n");
	GenerateCMakeLists(workspace);

	printf("executing autoconf\n");
	res = system("autoconf");
	if (res != 0) printf("WARNING: result code %d\n", res);

	printf("executing configure script\n");
	printf("-------------------------------------BEGIN-------------------------------------\n");
	res = system("./configure");
	printf("--------------------------------------END--------------------------------------\n");
	if (res != 0) printf("WARNING: result code %d\n", res);

	return EXIT_SUCCESS;
}
