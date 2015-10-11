
//
// Tool to automatically check the links in the Shark documentation.
//
// Author: Tobias Glasmachers
// Date: 2008
//


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>


#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif


class Link;
class Target;
class Document;


class String
{
public:
	String()
	{
		length = 0;
		buffer = (char*)malloc(1);
		buffer[0] = 0;
	}

	String(const char* s)
	{
		length = strlen(s);
		buffer = (char*)malloc(length + 1);
		memcpy(buffer, s, length + 1);
	}

	String(const char* s, int len)
	{
		length = len;
		buffer = (char*)malloc(length + 1);
		memcpy(buffer, s, length);
		buffer[length] = 0;
	}

	String(const String& s)
	{
		length = s.getLength();
		buffer = (char*)malloc(length + 1);
		memcpy(buffer, (const char*)s, length + 1);
	}

	~String()
	{
		// commenting this is desastrous as we waste several
		// megabytes of memory, but the error is deeply hidden...
		// if (buffer != NULL) free(buffer);
	}


	inline int getLength() const
	{
		return length;
	}

	inline bool isEmpty() const
	{
		return (length == 0);
	}

	inline operator char* ()
	{
		return buffer;
	}

	inline operator const char* () const
	{
		return buffer;
	}

	inline char& operator [] (int index)
	{
#ifdef DEBUG
		if (index > length) throw "[String::operator []] access violation";
#endif
		return buffer[index];
	}

	inline const char& operator [] (int index) const
	{
#ifdef DEBUG
		if (index > length) throw "[String::operator []] access violation";
#endif
		return buffer[index];
	}

	void operator = (const char* s)
	{
		free(buffer);
		length = strlen(s);
		buffer = (char*)malloc(length + 1);
		memcpy(buffer, s, length + 1);
	}

	inline bool operator == (const char* s) const
	{
		return (strcmp(buffer, s) == 0);
	}

	inline bool operator != (const char* s) const
	{
		return (strcmp(buffer, s) != 0);
	}

	inline bool operator < (const char* s) const
	{
		return (strcmp(buffer, s) < 0);
	}

	inline bool operator <= (const char* s) const
	{
		return (strcmp(buffer, s) <= 0);
	}

	inline bool operator > (const char* s) const
	{
		return (strcmp(buffer, s) > 0);
	}

	inline bool operator >= (const char* s) const
	{
		return (strcmp(buffer, s) >= 0);
	}

	int FindFirst(const char* pattern, int start = 0)
	{
		int plen = strlen(pattern);
		if (plen > length) return -1;
		int i;
		for (i=start; i<=length-plen; i++)
		{
			if (memcmp(buffer + i, pattern, plen) == 0) return i;
		}
		return -1;
	}

	String Substring(int start, int len = -1)
	{
		if (len == -1) return String(buffer + start);
		else return String(buffer + start, len);
	}

	void Delete(int start)
	{
		if (start > length) return;
		buffer[start] = 0;
		length = start;
	}

	int CompareNoCase(const char* s)
	{
		int l1 = length;
		int l2 = strlen(s);
		int i, l = (l1 < l2) ? l1 : l2;
		for (i=0; i<l; i++)
		{
			char c1 = buffer[i];
			char c2 = s[i];
			if (c1 >= 'A' && c1 <= 'Z') c1 += 32;
			if (c2 >= 'A' && c2 <= 'Z') c2 += 32;
			if (c1 < c2) return -1;
			else if (c1 > c2) return 1;
		}
		if (l1 < l2) return -1;
		else if (l1 > l2) return 1;
		else return 0;
	}

	String operator + (const char* s)
	{
		int l1 = length;
		int l2 = strlen(s);
		char* tmp = (char*)malloc(l1 + l2 + 1);
		memcpy(tmp, buffer, l1);
		memcpy(tmp + l1, s, l2);
		tmp[l1 + l2] = 0;
		String ret(tmp);
		delete tmp;
		return ret;
	}

	void operator += (const char* s)
	{
		int l1 = length;
		int l2 = strlen(s);
		char* tmp = (char*)malloc(l1 + l2 + 1);
		memcpy(tmp, buffer, l1);
		memcpy(tmp + l1, s, l2);
		tmp[l1 + l2] = 0;
		free(buffer);
		buffer = tmp;
		length = l1 + l2;
	}

	bool Read(String filename)
	{
		FILE* file = fopen((const char*)filename, "r");
		if (file == NULL) return false;
		fseek(file, 0, SEEK_END);
		length = ftell(file);
		free(buffer);
		buffer = (char*)malloc(length + 1);
		fseek(file, 0, SEEK_SET);
		int num = fread(buffer, 1, length, file);
		if (num != length) throw "[String::Read] error reading file";
		fclose(file);
		buffer[length] = 0;
		return true;
	}

protected:
	int length;
	char* buffer;
};


class StringList
{
public:
	StringList()
	{
	}

	~StringList()
	{
		int i, ic = data.size();
		for (i=0; i<ic; i++) delete data[i];
	}


	int size()
	{
		return data.size();
	}

	const String& operator [] (int index)
	{
#ifdef DEBUG
		if (index < 0 || index >= (int)data.size()) throw "[StringList::operator []] access violation";
#endif
		return *data[index];
	}

	int find(const char* s)
	{
		int i, ic = data.size();
		for (i=0; i<ic; i++) if (*data[i] == s) return i;
		return -1;
	}

	int findNoCase(const char* s)
	{
		int i, ic = data.size();
		for (i=0; i<ic; i++) if (data[i]->CompareNoCase(s) == 0) return i;
		return -1;
	}

	void push_back(const String& s)
	{
		String* ps = new String(s);
		data.push_back(ps);
	}

	void erase(int index)
	{
#ifdef DEBUG
		if (index < 0 || index >= (int)data.size()) throw "[StringList::erase] access violation";
#endif
		delete data[index];
		data.erase(data.begin() + index);
	}

protected:
	std::vector<String*> data;
};


template <class T> class MyPtrVector : protected std::vector<T*>
{
public:
	MyPtrVector()
	{
	}

	~MyPtrVector()
	{
	}


	inline unsigned int size() const
	{
		return std::vector<T*>::size();
	}

	inline T* operator [](int index)
	{
		return std::vector<T*>::operator [](index);
	}

	inline T* operator [](int index) const
	{
		return std::vector<T*>::operator [](index);
	}

	inline int find(T* element) const
	{
		unsigned int index;
		if (findIndex(element, index)) return index;
		else return -1;
	}

	bool insert(T* element)
	{
		unsigned int index;
		if (findIndex(element, index)) return false;

		std::vector<T*>::insert(std::vector<T*>::begin() + index, element);
		return true;
	}

	bool erase(T* element)
	{
		unsigned int index;
		if (! findIndex(element, index)) return false;

		std::vector<T*>::erase(std::vector<T*>::begin() + index);
		return true;
	}

protected:
	bool findIndex(T* element, unsigned int& index) const
	{
		// binary search for the initial step size
		unsigned int stepsize = 0;
		unsigned int b = 16;
		for (b = 16; b > 0; b >>= 1)
		{
			if ((int)std::vector<T*>::size() >= (1 << (stepsize + b))) stepsize += b;
		}
		stepsize = (1 << stepsize);

		// check if the element is larger than all elements in the list
		if (std::vector<T*>::size() == 0 || *std::vector<T*>::operator[](std::vector<T*>::size() - 1) < *element)
		{
			index = std::vector<T*>::size();
			return false;
		}

		// binary search in the ordered list
		index = 0;
		unsigned int i;
		while (stepsize > 0)
		{
			i = index + stepsize;
			if (i < std::vector<T*>::size())
			{
				if (*std::vector<T*>::operator[](i - 1) < *element) index = i;
			}
			stepsize >>= 1;
		}

		return (!(*element < *std::vector<T*>::operator[](index)));
	}
};


// representation of an html file after scanning
class Document
{
public:
	Document(String path)
	{
		this->path = path;
	}

	~Document()
	{
	}


	bool operator < (const Document& other) const
	{
		return (path < other.path);
	}

	String path;
	StringList anchor;
};


// description of the target document of a link
class Target
{
public:
	Target(String abspath)
	{
		this->abspath = abspath;
	}

	~Target()
	{
	}


	bool operator == (const Target& other) const
	{
		return (abspath == other.abspath);
	}

	bool operator < (const Target& other) const
	{
		return (abspath < other.abspath);
	}

	void Print();

	String abspath;
	MyPtrVector<Link> link;
};


// complete description of a link
class Link
{
public:
	Link(Target* target, Document* doc, String text, String href, String anchor = "")
	{
		this->target = target;
		this->doc = doc;
		this->text = text;
		this->href = href;
		this->anchor = anchor;
	}

	~Link()
	{
	}


	bool operator == (const Link& other) const
	{
		return (target == other.target && doc == other.doc && text == other.text && href == other.href && anchor == other.anchor);
	}

	bool operator < (const Link& other) const
	{
		if (target == other.target)
		{
			if (doc == other.doc)
			{
				if (text == other.text)
				{
					if (href == other.href)
					{
						return (anchor < other.anchor);
					}
					else return (href < other.href);
				}
				else return (text < other.text);
			}
			else return ((long long)doc < (long long)other.doc);
		}
		else return ((long long)target < (long long)other.target);
	}

	void Print()
	{
		std::cout << "  doc=\"" << doc->path << "\"" << std::endl;
		std::cout << "    text=\"" << text << "\"" << std::endl;
		std::cout << "    href=\"" << href << "\"" << std::endl;
	}

	Target* target;
	Document* doc;
	String text;
	String href;
	String anchor;
};


void Target::Print()
{
	std::cout << "target=\"" << abspath << "\"" << std::endl;
	int i, ic = link.size();
	for (i=0; i<ic; i++) link[i]->Print();
}


class LinkChecker
{
public:
	LinkChecker()
	: external("")
	{
	}

	~LinkChecker()
	{
	}


	void Help()
	{
		std::cout << std::endl;
		std::cout << "LinkChecker checks the links of html documents in and below the" << std::endl;
		std::cout << "current directory. It checks all local relative links including" << std::endl;
		std::cout << "anchors and all external http:// links, ignoring anchors." << std::endl;
		std::cout << "Further it reports all mailto and other links for manual checking." << std::endl;
		std::cout << std::endl;
		std::cout << "usage: LinkChecker [--quiet]" << std::endl;
		std::cout << std::endl;
		exit(0);
	}

	void Run(int argc, char** argv)
	{
		if (argc > 2) Help();
		verbose = (argc != 2);
		if (argc == 2 && strcmp(argv[1], "--quiet") != 0) Help();

		timeval start;
		gettimeofday(&start, NULL);

		numDocs = 0;
		numInternalLinks = 0;
		numExternalLinks = 0;
		numBroken = 0;
		numMailLinks = 0;
		numOtherLinks = 0;

		std::cout << std::endl;
		std::cout << "Shark documentation link checker" << std::endl;

		ScanDir("./");

		std::cout << std::endl;
		std::cout << "CHECKING INTERNAL LINKS" << std::endl;

		int i, ic;
		ic = internal.size();
		for (i=0; i<ic; i++) CheckLink(internal[i]);
		numInternalLinks = ic;

		CheckExternalLinks();

		timeval finish;
		gettimeofday(&finish, NULL);
		int secs = finish.tv_sec - start.tv_sec;
		int msecs = (finish.tv_usec - start.tv_usec) / 1000;
		double totaltime = secs + 0.001 * msecs;

		std::cout << std::endl;
		std::cout << "SUMMARY:" << std::endl;
		std::cout << "# documents:               " << numDocs << std::endl;
		std::cout << "# internal relative links: " << numInternalLinks << std::endl;
		std::cout << "# external http links:     " << numExternalLinks << std::endl;
		std::cout << "# KNOWN BROKEN LINKS:      " << numBroken << std::endl;
		std::cout << "# mailto links:            " << numMailLinks << std::endl;
		std::cout << "# other (unknown) links:   " << numOtherLinks << std::endl;
		std::cout << "time: " << totaltime << " seconds." << std::endl;
		std::cout << std::endl;
	}

	bool CheckHttpLink(String domain, String resource)
	{
		numExternalLinks++;

		// create socket
		int sock = socket(AF_INET, SOCK_STREAM, 0);
		if (sock <= 0)
		{
			std::cout << "ERROR opening socket for :" << std::endl;
			return false;
		}

		// connect socket
		hostent *host = gethostbyname(domain);
		if (host == NULL)
		{
			std::cout << "broken link (unknown host):" << std::endl;
			close(sock);
			return false;
		}
		sockaddr_in addr;
		addr.sin_family = AF_INET;
		addr.sin_port = htons(80);
		addr.sin_addr.s_addr = ((in_addr *)host->h_addr)->s_addr;
		memset(&(addr.sin_zero), 0, 8);
		if (connect(sock, (sockaddr*) & addr, sizeof(sockaddr)) < 0)
		{
			std::cout << "broken link (can not connect):" << std::endl;
			close(sock);
			return false;
		}

		// send http request
		String request = String("GET ") + resource + " HTTP/1.1\r\nhost: " + domain + "\r\n\r\n";
		if (send(sock, (const char*)request, request.getLength(), MSG_NOSIGNAL) <= 0)
		{
			std::cout << "broken link (can not send):" << std::endl;
			close(sock);
			return false;
		}

		// wait for reply
		timeval timeout;
		timeout.tv_sec = 5;
		timeout.tv_usec = 0;
		fd_set read_socks;
		FD_ZERO(&read_socks);
		FD_SET(sock, &read_socks);
		if (select(sock + 1, &read_socks, NULL, NULL, &timeout) != 1)
		{
			std::cout << "broken link (can not receive or timeout):" << std::endl;
			close(sock);
			return false;
		}
		FD_ZERO(&read_socks);

		// receive
		char answer[256];
		int n = recv(sock, answer, 255, 0);
		close(sock);
		if (n <= 0)
		{
			std::cout << "broken link (can not receive or timeout):" << std::endl;
			return false;
		}

		answer[n] = 0;
		if (memcmp(answer, "HTTP/1.", 7) != 0)
		{
			std::cout << "broken link (no HTTP/1.x answer):" << std::endl;
			return false;
		}
		answer[14] = 0;
		int code = atoi(answer + 10);
		if (code >= 400)
		{
			std::cout << "broken link (code " << code << "):" << std::endl;
			return false;
		}
		return true;
	}

	void CheckExternalLinks()
	{
		StringList mailto;
		StringList other;

		std::cout << std::endl;
		std::cout << "CHECKING EXTERNAL LINKS (this may take some time)" << std::endl;

		int i, ic = external.link.size();
		for (i=0; i<ic; i++)
		{
			String href = external.link[i]->href;
			if (href.Substring(0, 7).CompareNoCase("http://") == 0)
			{
				int a = href.FindFirst("#");
				if (a != -1) href.Delete(a);
				int s = href.FindFirst("/", 7);
				if (s == -1)
				{
					if (! CheckHttpLink(href.Substring(7), "/"))
					{
						numBroken++;
						external.link[i]->Print();
					}
				}
				else
				{
					if (! CheckHttpLink(href.Substring(7, s - 7), href.Substring(s)))
					{
						numBroken++;
						external.link[i]->Print();
					}
				}
			}
			else if (href.FindFirst("mailto:") != -1)
			{
				if (mailto.findNoCase(href) == -1) mailto.push_back(href);
			}
			else
			{
				if (other.findNoCase(href) == -1) other.push_back(href);
			}
		}
		std::cout << std::endl;

		std::cout << "MAIL LINKS (check manually):" << std::endl;
		ic = mailto.size();
		for (i=0; i<ic; i++) std::cout << "    " << mailto[i] << std::endl;
		std::cout << std::endl;
		numMailLinks = ic;

		std::cout << "OTHER LINKS (check manually):" << std::endl;
		ic = other.size();
		for (i=0; i<ic; i++) std::cout << "    " << other[i] << std::endl;
		std::cout << std::endl;
		numOtherLinks = ic;
	}

	String ConcatPaths(String path1, String path2)
	{
		StringList component;
		int a, b;

		a = 0;
		while (true)
		{
			b = path1.FindFirst("/", a);
			if (b == -1)
			{
				if (a < (int)path1.getLength()) component.push_back(path1.Substring(a));
				break;
			}
			if (a < b) component.push_back(path1.Substring(a, b - a));
			a = b+1;
		}

		a = 0;
		while (true)
		{
			b = path2.FindFirst("/", a);
			if (b == -1)
			{
				if (a < (int)path2.getLength()) component.push_back(path2.Substring(a));
				break;
			}
			if (a < b) component.push_back(path2.Substring(a, b - a));
			a = b+1;
		}

		for (a=0; a<(int)component.size(); a++)
		{
			if (component[a] == ".")
			{
				component.erase(a);
				a--;
			}
			else if (component[a] == "..")
			{
				if (a > 0 && component[a - 1] != "..")
				{
					component.erase(a);
					component.erase(a - 1);
					a -= 2;
				}
			}
		}

		if (component.size() == 0) return ".";

		String ret = component[0];
		for (a=1; a<(int)component.size(); a++)
		{
			ret += "/";
			ret += component[a];
		}
		return ret;
	}

	Target* getTarget(String abspath)
	{
		Target* target = new Target(abspath);
		int pos = internal.find(target);
		if (pos == -1)
		{
			internal.insert(target);
			return target;
		}
		else
		{
			delete target;
			return internal[pos];
		}
	}

	Link* getLink(Target* target, Document* doc, String text, String href, String anchor = "")
	{
		Link* link = new Link(target, doc, text, href, anchor);
		int pos = target->link.find(link);
		if (pos == -1)
		{
			target->link.insert(link);
			return link;
		}
		else
		{
			delete link;
			return target->link[pos];
		}
	}

	void ScanDir(String path)
	{
		if (path[path.getLength() - 1] != '/') path += "/";

		DIR* dir = opendir((const char*)path);
		if (dir == NULL) return;

		String s;
		dirent* entry;
		while (true)
		{
			entry = readdir(dir);
			if (entry == NULL) break;
			s = entry->d_name;
			int len = s.getLength();
			if (len == 0) break;
			if (len > 5 && s.Substring(len - 5).CompareNoCase(".html") == 0)
			{
				ScanFile(path, s);
			}
			if (s[0] == '.') continue;
			ScanDir(path + s);
		}
		closedir(dir);
	}

	void ScanFile(String path, String filename)
	{
		numDocs++;

		String content;
		int pos, end;
		if (! content.Read(path + filename))
		{
			std::cout << "failed to open " << (const char*)path << (const char*)filename << std::endl;
		}
		else
		{
			if (verbose) std::cout << "scanning " << (const char*)path << (const char*)filename << " ..." << std::flush;
			Document* doc = new Document(path + filename);
			scanned.insert(doc);

			// find links and anchors
			pos = 0;
			while (true)
			{
				pos = content.FindFirst("<a ", pos);
				if (pos == -1) break;
				end = content.FindFirst("</a>", pos + 4);
				if (end == -1) break;
				int m, n;
				String text;
				m = content.FindFirst(">", pos);
				if (m != -1 && m < end) text = content.Substring(m + 1, end - m - 1);
				m = content.FindFirst("href=\"", pos);
				if (m != -1 && m < end)
				{
					m += 6;
					n = content.FindFirst("\"", m);
					if (n != -1 && n < end)
					{
						// link
						String link = content.Substring(m, n - m);
						if (link.FindFirst("javascript:") == -1)
						{
							int a = link.FindFirst("#");
							if (a == -1) a = link.getLength();
							if ((link.FindFirst("://") == -1) && (link.FindFirst("mailto:") == -1))
							{
								Target* t = getTarget(ConcatPaths(path, link.Substring(0, a)));
								getLink(t, doc, text, link, link.Substring(a));
							}
							else
							{
								getLink(&external, doc, text, link);
							}
						}
					}
				}
				m = content.FindFirst("name=\"", pos);
				if (m != -1 && m < end)
				{
					m += 6;
					n = content.FindFirst("\"", m);
					if (n != -1 && n < end)
					{
						// anchor
						String anchor = content.Substring(m, n - m);
						doc->anchor.push_back(anchor);
					}
				}
				pos = end + 4;
			}
			if (verbose) std::cout << " done." << std::endl;
		}
	}

	void CheckLink(Target* target)
	{
		String msg;
		Document tmpdoc(target->abspath);
		int pos = scanned.find(&tmpdoc);
		if (pos == -1)
		{
			// check if the file exists
			FILE* file = fopen((const char*)target->abspath, "r");
			if (file != NULL) fclose(file);
			else
			{
				// error message
				msg = String("broken link: file '") + target->abspath + "' not found.\n";
				int i, ic = target->link.size();
				for (i=0; i<ic; i++)
				{
					Link* link = target->link[i];
					msg += String("  doc=\"") + link->doc->path + "\"\n";
					msg += String("    text=\"") + link->text + "\"\n";
					msg += String("    href=\"") + link->href + "\"\n";
				}
			}
		}
		else
		{
			// check all anchors
			bool err = false;
			Document* doc = scanned[pos];
			int i, ic = target->link.size();
			for (i=0; i<ic; i++)
			{
				Link* link = target->link[i];
				if (! link->anchor.isEmpty())
				{
					int pos = doc->anchor.find(link->anchor);
					if (pos == -1)
					{
						// error message
						if (! err)
						{
							msg = String("broken link: anchors in file '") + doc->path + "' not found.\n";
							err = true;
						}
						msg += String("  doc=\"") + link->doc->path + "\"\n";
						msg += String("    text=\"") + link->text + "\"\n";
						msg += String("    href=\"") + link->href + "\"\n";
					}
				}
			}
		}
		if (! msg.isEmpty())
		{
			numBroken++;
			std::cout << msg;
		}
	}

	// files already scanned
	MyPtrVector<Document> scanned;

	// links to test
	MyPtrVector<Target> internal;

	// external links to report
	Target external;

	// statistics
	int numDocs;
	int numInternalLinks;
	int numExternalLinks;
	int numBroken;
	int numMailLinks;
	int numOtherLinks;

	bool verbose;
};


int main(int argc, char** argv)
{
	try
	{
		LinkChecker theApp;
		theApp.Run(argc, argv);
	}
	catch (const char* exception)
	{
		std::cout << std::endl;
		std::cout << "EXCEPTION: " << exception << std::endl;
		std::cout << std::endl;
	}
}
