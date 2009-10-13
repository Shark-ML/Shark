/**
 * \file xmlParser.h
 * 
 * \brief Implementation file for basic %XML parser written in ANSI C++ for 
 * portability.
 * 
 * The parser works by using recursion and a node tree for breaking down the 
 * elements of an %XML document.
 *
 * @version     V2.37
 * @author      <a href="http://www.applied-mathematics.net/tools/xmlParser.html">Frank Vanden Berghen</a>
 *
 * Copyright (c) 2002, Frank Vanden Berghen
 * All rights reserved.
 * See the AFPL license at the end of this file about the licensing terms
 */

#ifndef __INCLUDE_XML_NODE__
#define __INCLUDE_XML_NODE__

#include <stdlib.h>

#ifdef _UNICODE

/**
 * \brief Indicates if the parser is allowed to switch to UNICODE mode.
 * 
 * If you comment the next "define" line then the library will never "switch to"
 * _UNICODE (wchar_t*) mode (16/32 bits per characters).<br>
 * This is useful when you get error messages like:<br>
 * <pre>   'XMLNode::openFileHelper' : cannot convert parameter 2 from 'const char [5]' to 'const wchar_t *'</pre><br>
 * The _XMLWIDECHAR preprocessor variable force the XMLParser library into 
 * either utf16/32-mode (the proprocessor variable must be defined) or 
 * utf8-mode(the pre-processor variable must be undefined).
 */
#define _XMLWIDECHAR
#endif

#if defined(WIN32) || defined(UNDER_CE) || defined(_WIN32) || defined(WIN64) || defined(__BORLANDC__)
// comment the next line if you are under windows and the compiler is not Microsoft Visual Studio (6.0 or .NET) or Borland
#define _XMLWINDOWS
#endif

#ifdef XMLDLLENTRY
#undef XMLDLLENTRY
#endif
#ifdef _USE_XMLPARSER_DLL
#ifdef _DLL_EXPORTS_
#define XMLDLLENTRY __declspec(dllexport)
#else
#define XMLDLLENTRY __declspec(dllimport)
#endif
#else
#define XMLDLLENTRY
#endif

// uncomment the next line if you want no support for wchar_t* (no need for the <wchar.h> or <tchar.h> libraries anymore to compile)
//#define XML_NO_WIDE_CHAR

#ifdef XML_NO_WIDE_CHAR
#undef _XMLWINDOWS
#undef _XMLWIDECHAR
#endif

#ifdef _XMLWINDOWS
#include <tchar.h>
#else
#define XMLDLLENTRY
#ifndef XML_NO_WIDE_CHAR
#include <wchar.h> // to have 'wcsrtombs' for ANSI version
                   // to have 'mbsrtowcs' for WIDECHAR version
#endif
#endif

// Some common types for char set portable code
#ifdef _XMLWIDECHAR
    #define _CXML(c) L ## c
    #define XMLCSTR const wchar_t *
    #define XMLSTR  wchar_t *
    #define XMLCHAR wchar_t
#else
    #define _CXML(c) c
    #define XMLCSTR const char *
    #define XMLSTR  char *
    #define XMLCHAR char
#endif
#ifndef FALSE
    #define FALSE 0
#endif /* FALSE */
#ifndef TRUE
    #define TRUE 1
#endif /* TRUE */


/**
 * \brief Enumeration for %XML parse errors.
 */
typedef enum XMLError
{
    eXMLErrorNone = 0,
    eXMLErrorMissingEndTag,
    eXMLErrorNoXMLTagFound,
    eXMLErrorEmpty,
    eXMLErrorMissingTagName,
    eXMLErrorMissingEndTagName,
    eXMLErrorUnmatchedEndTag,
    eXMLErrorUnmatchedEndClearTag,
    eXMLErrorUnexpectedToken,
    eXMLErrorNoElements,
    eXMLErrorFileNotFound,
    eXMLErrorFirstTagNotFound,
    eXMLErrorUnknownCharacterEntity,
    eXMLErrorCharacterCodeAbove255,
    eXMLErrorCharConversionError,
    eXMLErrorCannotOpenWriteFile,
    eXMLErrorCannotWriteFile,

    eXMLErrorBase64DataSizeIsNotMultipleOf4,
    eXMLErrorBase64DecodeIllegalCharacter,
    eXMLErrorBase64DecodeTruncatedData,
    eXMLErrorBase64DecodeBufferTooSmall
} XMLError;


/// Enumeration used to manage type of data. Use in conjunction with structure XMLNodeContents
typedef enum XMLElementType
{
    eNodeChild=0,
    eNodeAttribute=1,
    eNodeText=2,
    eNodeClear=3,
    eNodeNULL=4
} XMLElementType;

/// Structure used to obtain error details if the parse fails.
typedef struct XMLResults
{
    enum XMLError error;
    int  nLine,nColumn;
} XMLResults;

/// Structure for %XML clear (unformatted) node (usually comments)
typedef struct XMLClear {
    XMLCSTR lpszValue; XMLCSTR lpszOpenTag; XMLCSTR lpszCloseTag;
} XMLClear;

/// Structure for %XML attribute.
typedef struct XMLAttribute {
    XMLCSTR lpszName; XMLCSTR lpszValue;
} XMLAttribute;

/// XMLElementPosition are not interchangeable with simple indexes
typedef int XMLElementPosition;

struct XMLNodeContents;

/**
 * \brief Represents one node of the %XML tree.
 */
typedef struct XMLDLLENTRY XMLNode
{
  private:

    struct XMLNodeDataTag;

    // protected constructors: use one of these four methods to get your first instance of XMLNode:
    //  - parseString
    //  - parseFile
    //  - openFileHelper
    //  - createXMLTopNode
    XMLNode(struct XMLNodeDataTag *pParent, XMLSTR lpszName, char isDeclaration);
    XMLNode(struct XMLNodeDataTag *p);

  public:

    // You can create your first instance of XMLNode with these 4 functions:
    // (see complete explanation of parameters below)

	/**
	 * \brief Creates a new XMLNode.
	 */
    static XMLNode createXMLTopNode(XMLCSTR lpszName, char isDeclaration=FALSE);

	/**
	 * \brief Creates XMLNode from a String.
	 * 
	 * The tag parameter should be the name of the first tag inside the %XML 
	 * file. If the tag parameter is omitted, the function returns a node that 
	 * represents the head of the xml document including the declaration term 
	 * (&lt;? ... ?&gt;).
	 */
    static XMLNode parseString   (XMLCSTR  lpXMLString, XMLCSTR tag=NULL, XMLResults *pResults=NULL);

	/**
	 * \brief Creates XMLNode from file.
	 * 
	 * The tag parameter should be the name of the first tag inside the %XML 
	 * file. If the tag parameter is omitted, the function returns a node that 
	 * represents the head of the xml document including the declaration term 
	 * (&lt;? ... ?&gt;).
	 */
    static XMLNode parseFile     (XMLCSTR     filename, XMLCSTR tag=NULL, XMLResults *pResults=NULL);

	/**
	 * \brief Creates XMLNode from file and reports errors to screen.
	 * 
	 * The "openFileHelper" reports to the screen all the warnings & errors that 
	 * occurred during parsing of the %XML file. Since each application has its 
	 * own way to report and deal with errors, you should rather use the 
	 * "parseFile" function to parse %XML files and program yourself thereafter
	 * an "error reporting" tailored for your needs (instead of using the very 
	 * crude "error reporting" mechanism included inside the "openFileHelper" 
	 * function).
	 *
	 * If the %XML document is corrupted:
	 * <ul>
	 *   <li>
	 *     The "openFileHelper" method will:
	 *     <ul>
	 *       <li>
	 *         display an error message on the console (or inside a messageBox for windows).
	 *       </li>
	 *       <li>
	 *         stop execution (exit).
	 *       </li>
	 *     </ul>
	 *     I suggest that you write your own "openFileHelper" method tailored to your needs.
	 * 	 </li>
	 *   <li>
	 *	   The 2 other methods will initialize the "pResults" variable with some 
	 *     information that can be used to trace the error.
	 *   </li>
	 *   <li>
	 *     If you still want to parse the file, you can use the 
	 *     APPROXIMATE_PARSING option as explained inside the note at the 
	 *     beginning of the "xmlParser.cpp" file.
	 *   </li>
	 * </ul>
	 * 
	 * The tag parameter should be the name of the first tag inside the %XML 
	 * file. If the tag parameter is omitted, the function returns a node that 
	 * represents the head of the xml document including the declaration term 
	 * (&lt;? ... ?&gt;).
	 */
    static XMLNode openFileHelper(XMLCSTR     filename, XMLCSTR tag=NULL                           );

	/**
	 * \brief Returns a user-friendly explanation of the parsing error.
	 */
	static XMLCSTR getError(XMLError error);
	static XMLCSTR getVersion();									///< Returns version of xmlParser.

	XMLCSTR getName() const;                                      	///< Returns name of the node.
	XMLCSTR getText(int i=0) const;                                 ///< Returns i<i>th</i> text field.
	int nText() const;                                              ///< Returns nbr of text field.
	XMLNode getParentNode() const;                                  ///< Returns the parent node.
	XMLNode getChildNode(int i=0) const;                            ///< Returns i<i>th</i> child node.
	
	/**
	 * \brief Returns i<i>th</i> child node with specific name.
	 * \return empty node if failing
	 */
	XMLNode getChildNode(XMLCSTR name, int i)  const; 
	
	/**
	 * \brief Returns i<i>th</i> child node with specific name.
	 * \return empty node if failing
	 */
	XMLNode getChildNode(XMLCSTR name, int *i=NULL) const;
	
	/**
	 * \brief Returns child node with specific name/attribute.
	 * \return empty node if failing
	 */
	XMLNode getChildNodeWithAttribute(XMLCSTR tagName, 
	                                  XMLCSTR attributeName,
	                                  XMLCSTR attributeValue=NULL,
	                                  int *i=NULL)  const;
	
	int nChildNode(XMLCSTR name) const;                            	///< Returns the number of child node with specific name.
	int nChildNode() const;                                         ///< Returns nbr of child node.
	XMLAttribute getAttribute(int i=0) const;                       ///< Returns i<i>th</i> attribute.
	XMLCSTR      getAttributeName(int i=0) const;                   ///< Returns i<i>th</i> attribute name.
	XMLCSTR      getAttributeValue(int i=0) const;                  ///< Returns i<i>th</i> attribute value.
	char  isAttributeSet(XMLCSTR name) const;                       ///< Tests if an attribute with a specific name is given.
	
	/**
	 * \brief Returns i<i>th</i> attribute content with specific name.
	 * \return NULL if failing
	 */
	XMLCSTR getAttribute(XMLCSTR name, int i) const;

	/**
	 * \brief Returns next attribute content with specific name.
	 * \return NULL if failing
	 */
    XMLCSTR getAttribute(XMLCSTR name, int *i=NULL) const; 

    int nAttribute() const;                                         ///< Returns nbr of attribute.
    XMLClear getClear(int i=0) const;                               ///< Returns i<i>th</i> clear field (comments).
    int nClear() const;                                             ///< Returns nbr of clear field.
    
    /**
     * \brief Creates %XML string starting from current XMLNode.
     * 
	 * If nFormat==0, no formatting is required otherwise this returns an user 
	 * friendly %XML string from a given element with appropriate white spaces 
	 * and carriage returns.<br>
     * if pnSize is given it returns the size in character of the string.
     */
    XMLSTR createXMLString(int nFormat=1, int *pnSize=NULL) const;

    /**
     * \brief Saves the content of an xmlNode inside a file.
     * 
     * The nFormat parameter has the same meaning as in the createXMLString 
     * function. If the global parameter "characterEncoding==encoding_UTF8", 
     * then the "encoding" parameter is ignored and always set to "utf-8". 
     * If the global parameter "characterEncoding==encoding_ShiftJIS", then the 
     * "encoding" parameter is ignored and always set to "SHIFT-JIS". If 
     * "_XMLWIDECHAR=1", then the "encoding" parameter is ignored and always set 
     * to "utf-16". If no "encoding" parameter is given the "ISO-8859-1" 
     * encoding is used.
     */
    XMLError writeToFile(XMLCSTR filename, const char *encoding=NULL, char nFormat=1) const;
    
    /**
     * \brief Enumerates all the different contents of the current XMLNode.
     * 
     * Possible contents are: attribute, child, text, clear.<br>
     * The order is reflecting the order of the original file/string.<br>
 	 * NOTE: 0 <= i < nElement();
     */
    XMLNodeContents enumContents(XMLElementPosition i) const;

    int nElement() const;                                        	///< Returns nbr of different contents for current node.
    char isEmpty() const;                                           ///< Is this node Empty?
    char isDeclaration() const;                                     ///< Is this node a declaration <? .... ?>
    XMLNode deepCopy() const;                                       ///< Deep copies (duplicates/clones) a XMLNode.
    static XMLNode emptyNode();                                     ///< Returns XMLNode::emptyXMLNode;

// to allow shallow/fast copy:
    ~XMLNode();
    XMLNode(const XMLNode &A);
    XMLNode& operator=( const XMLNode& A );

    XMLNode(): d(NULL){};
    static XMLNode emptyXMLNode;
    static XMLClear emptyXMLClear;
    static XMLAttribute emptyXMLAttribute;

    // The following functions allows you to create from scratch (or update) a XMLNode structure
    // Start by creating your top node with the "createXMLTopNode" function and then add new nodes with the "addChild" function.
    // The parameter 'pos' gives the position where the childNode, the text or the XMLClearTag will be inserted.
    // The default value (pos=-1) inserts at the end. The value (pos=0) insert at the beginning (Insertion at the beginning is slower than at the end).
    // REMARK: 0 <= pos < nChild()+nText()+nClear()

    /**
     * \brief Adds a child node.
     * 
     * The parameter 'pos' gives the position where the childNode will be 
     * inserted.<br>
     * The default value (pos=-1) inserts at the end. The value (pos=0) inserts 
     * at the beginning. Insertion at the beginning is slower than at the end.
     * <br>
     * REMARK: 0 <= pos < nChild()+nText()+nClear()
     */
    XMLNode       addChild(XMLCSTR lpszName, char isDeclaration=FALSE, XMLElementPosition pos=-1);

    /**
     * \brief Adds an attribute.
     */
    XMLAttribute *addAttribute(XMLCSTR lpszName, XMLCSTR lpszValuev);

    /**
     * \brief Adds a text element.
     * 
     * The parameter 'pos' gives the position where the text will be inserted.
     * <br>
     * The default value (pos=-1) inserts at the end. The value (pos=0) inserts 
     * at the beginning. Insertion at the beginning is slower than at the end.
     * <br>
     * REMARK: 0 <= pos < nChild()+nText()+nClear()
     */
    XMLCSTR       addText(XMLCSTR lpszValue, XMLElementPosition pos=-1);

    /**
     * \brief Adds a XMLClearTag (comment).
     * 
     * \param pos
     * The parameter 'pos' gives the position where the XMLClearTag will be 
     * inserted.<br>
     * The default value (pos=-1) inserts at the end. The value (pos=0) inserts 
     * at the beginning. Insertion at the beginning is slower than at the end.
     * <br>
     * REMARK: 0 <= pos < nChild()+nText()+nClear()
     */
    XMLClear     *addClear(XMLCSTR lpszValue, XMLCSTR lpszOpen=NULL, XMLCSTR lpszClose=NULL, XMLElementPosition pos=-1);
                                                                       // default values: lpszOpen ="<![CDATA["
                                                                       //                 lpszClose="]]>"

    /**
     * \brief Adds an existing node as child.
     * 
     * \param pos
     * The parameter 'pos' gives the position where the XMLClearTag will be 
     * inserted.<br>
     * The default value (pos=-1) inserts at the end. The value (pos=0) inserts 
     * at the beginning. Insertion at the beginning is slower than at the end.
     * <br>
     * REMARK: 0 <= pos < nChild()+nText()+nClear()
     */
    XMLNode       addChild(XMLNode nodeToAdd, XMLElementPosition pos=-1);

    // Some update functions:
    XMLCSTR       updateName(XMLCSTR lpszName);		///< Changes the node's name.

    /**
     * \brief Updates an attribute.
     * 
     * If the attribute to update is missing, a new one will be added.
     */
    XMLAttribute *updateAttribute(XMLAttribute *newAttribute, XMLAttribute *oldAttribute);

    /**
     * \brief Updates an attribute.
     * 
     * If the attribute to update is missing, a new one will be added.
     */
    XMLAttribute *updateAttribute(XMLCSTR lpszNewValue, XMLCSTR lpszNewName=NULL,int i=0);

    /**
     * \brief Updates an attribute.
     *
     * Set lpszNewName=NULL if you don't want to change the name of the 
     * attribute.<br> 
     * If the attribute to update is missing, a new one will be added.
     */
    XMLAttribute *updateAttribute(XMLCSTR lpszNewValue, XMLCSTR lpszNewName,XMLCSTR lpszOldName);

    /**
     * \brief Updates text.
     * 
     * If the text to update is missing, a new one will be added.
     */
    XMLCSTR updateText(XMLCSTR lpszNewValue, int i=0);

    /**
     * \brief Updates text.
     * 
     * If the text to update is missing, a new one will be added.
     */
    XMLCSTR updateText(XMLCSTR lpszNewValue, XMLCSTR lpszOldValue);

    /**
     * \brief Updates clear tag.
     * 
     * If the clearTag to update is missing, a new one will be added.
     */
    XMLClear *updateClear(XMLCSTR lpszNewContent, int i=0);

    /**
     * \brief Updates clear tag.
     * 
     * If the clearTag to update is missing, a new one will be added.
     */
    XMLClear *updateClear(XMLClear *newP,XMLClear *oldP);

    /**
     * \brief Updates clear tag.
     * 
     * If the clearTag to update is missing, a new one will be added.
     */
    XMLClear *updateClear(XMLCSTR lpszNewValue, XMLCSTR lpszOldValue);

    // Some deletion functions:
    void deleteNodeContent();  ///< Deletes the content of this XMLNode and the subtree.
    //void deleteChildrenContent(); // delete the content of the subtree.
    void deleteAttribute(XMLCSTR lpszName);				///< Deletes an attribute.
    void deleteAttribute(int i=0);						///< Deletes an attribute.
    void deleteAttribute(XMLAttribute *anAttribute);	///< Deletes an attribute.
    void deleteText(int i=0);							///< Deletes text.
    void deleteText(XMLCSTR lpszValue);					///< Deletes text.
    void deleteClear(int i=0);							///< Deletes clear tag (comment).
    void deleteClear(XMLClear *p);						///< Deletes clear tag (comment).
    void deleteClear(XMLCSTR lpszValue);				///< Deletes clear tag (comment).

    // The strings given as parameters for the following add and update methods (all these methods have
    // a name with the postfix "_WOSD" that means "WithOut String Duplication" ) will be free'd by the
    // XMLNode class. For example, it means that this is incorrect:
    //    xNode.addText_WOSD("foo");
    //    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
    // In opposition, this is correct:
    //    xNode.addText("foo");
    //    xNode.addText_WOSD(stringDup("foo"));
    //    xNode.updateAttribute("#newcolor" ,NULL,"color");
    //    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
    // Typically, you will never do:
    //    char *b=(char*)malloc(...);
    //    xNode.addText(b);
    //    free(b);
    // ... but rather:
    //    char *b=(char*)malloc(...);
    //    xNode.addText_WOSD(b);
    //    ('free(b)' is performed by the XMLNode class)

	/**
	 * \brief Creates a new XMLNode without string duplication (WOSD).
	 * 
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
	 */
    static XMLNode createXMLTopNode_WOSD(XMLSTR lpszName, char isDeclaration=FALSE);
    
    /**
     * \brief Adds a child node without string duplication (WOSD).
     * 
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
     * 
     * \param pos
     * The parameter 'pos' gives the position where the childNode will be 
     * inserted.<br>
     * The default value (pos=-1) inserts at the end. The value (pos=0) inserts 
     * at the beginning. Insertion at the beginning is slower than at the end.
     * <br>
     * REMARK: 0 <= pos < nChild()+nText()+nClear()
     */
    XMLNode addChild_WOSD(XMLSTR lpszName, char isDeclaration=FALSE, XMLElementPosition pos=-1);

    /**
     * \brief Adds an attribute without string duplication (WOSD).
	 * 
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
     */
    XMLAttribute *addAttribute_WOSD(XMLSTR lpszName, XMLSTR lpszValue);

    /**
     * \brief Adds a text element without string duplication (WOSD).
	 *
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
     * 
     * \param pos
     * The parameter 'pos' gives the position where the text will be inserted.
     * <br>
     * The default value (pos=-1) inserts at the end. The value (pos=0) inserts 
     * at the beginning. Insertion at the beginning is slower than at the end.
     * <br>
     * REMARK: 0 <= pos < nChild()+nText()+nClear()
     */
    XMLCSTR addText_WOSD(XMLSTR lpszValue, XMLElementPosition pos=-1);

    /**
     * \brief Adds a XMLClearTag (comment) without string duplication (WOSD).
     *
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
     * 
     * \param pos
     * The parameter 'pos' gives the position where the XMLClearTag will be 
     * inserted.<br>
     * The default value (pos=-1) inserts at the end. The value (pos=0) inserts 
     * at the beginning. Insertion at the beginning is slower than at the end.
     * <br>
     * REMARK: 0 <= pos < nChild()+nText()+nClear()
     */
    XMLClear *addClear_WOSD(XMLSTR lpszValue, XMLCSTR lpszOpen=NULL, XMLCSTR lpszClose=NULL, XMLElementPosition pos=-1);

    /**
     * \brief Changes the node's name without string duplication (WOSD).
     * 
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
     */
    XMLCSTR updateName_WOSD(XMLSTR lpszName);

    /**
     * \brief Updates an attribute without string duplication (WOSD).
     * 
     * If the attribute to update is missing, a new one will be added.
     * 
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
     */
    XMLAttribute *updateAttribute_WOSD(XMLAttribute *newAttribute, XMLAttribute *oldAttribute);

    /**
     * \brief Updates an attribute without string duplication (WOSD).
     * 
     * If the attribute to update is missing, a new one will be added.
     * 
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
     */
    XMLAttribute *updateAttribute_WOSD(XMLSTR lpszNewValue, XMLSTR lpszNewName=NULL,int i=0);

    /**
     * \brief Updates an attribute without string duplication (WOSD).
     * 
     * If the attribute to update is missing, a new one will be added.
     * 
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
     */
    XMLAttribute *updateAttribute_WOSD(XMLSTR lpszNewValue, XMLSTR lpszNewName,XMLCSTR lpszOldName);

    /**
     * \brief Updates text without string duplication (WOSD).
     * 
     * If the text to update is missing, a new one will be added.
     * 
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
     */
    XMLCSTR updateText_WOSD(XMLSTR lpszNewValue, int i=0);

    /**
     * \brief Updates text without string duplication (WOSD).
     * 
     * If the text to update is missing, a new one will be added.
     * 
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
     */
    XMLCSTR updateText_WOSD(XMLSTR lpszNewValue, XMLCSTR lpszOldValue);

    /**
     * \brief Updates clear tag without string duplication (WOSD).
     * 
     * If the clearTag to update is missing, a new one will be added.
     * 
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
     */
    XMLClear *updateClear_WOSD(XMLSTR lpszNewContent, int i=0);

    /**
     * \brief Updates clear tag without string duplication (WOSD).
     * 
     * If the clearTag to update is missing, a new one will be added.
     * 
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
     */
    XMLClear *updateClear_WOSD(XMLClear *newP,XMLClear *oldP);

    /**
     * \brief Updates clear tag without string duplication (WOSD).
     * 
     * If the clearTag to update is missing, a new one will be added.
     * 
     * The strings given as parameters for this method will be free'd by the
     * XMLNode class. For example, it means that this is incorrect:
     * <pre><code>
     *    xNode.addText_WOSD("foo");
     *    xNode.updateAttribute_WOSD("#newcolor" ,NULL,"color");
     * </code></pre>
     * In opposition, this is correct:
     * <pre><code>
     *    xNode.addText("foo");
     *    xNode.addText_WOSD(stringDup("foo"));
     *    xNode.updateAttribute("#newcolor" ,NULL,"color");
     *    xNode.updateAttribute_WOSD(stringDup("#newcolor"),NULL,"color");
     * </code></pre>
     * Typically, you will never do:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText(b);
     *    free(b);
     * </code></pre>
     * ... but rather:
     * <pre><code>
     *    char *b=(char*)malloc(...);
     *    xNode.addText_WOSD(b);
     * </code></pre>
     *    ('free(b)' is performed by the XMLNode class)
     */
    XMLClear *updateClear_WOSD(XMLSTR lpszNewValue, XMLCSTR lpszOldValue);

    // These are some useful functions when you want to insert a childNode, a text or a XMLClearTag in the
    // middle (at a specified position) of a XMLNode tree already constructed. The value returned by these
    // methods is to be used as last parameter (parameter 'pos') of addChild, addText or addClear.
    
    /**
     * \brief Returns the position of a certain text node.
     * 
     * The value returned by this method is to be used as last parameter 
     * (parameter 'pos') of addChild, addText or addClear. This is usefull when 
     * you want to insert a childNode, a text or a XMLClearTag in the middle 
     * (at a specified position) of a XMLNode tree already constructed.
     */
    XMLElementPosition positionOfText(int i=0) const;

    /**
     * \brief Returns the position of a certain text node.
     * 
     * The value returned by this method is to be used as last parameter 
     * (parameter 'pos') of addChild, addText or addClear. This is usefull when 
     * you want to insert a childNode, a text or a XMLClearTag in the middle 
     * (at a specified position) of a XMLNode tree already constructed.
     */
    XMLElementPosition positionOfText(XMLCSTR lpszValue) const;

    /**
     * \brief Returns the position of a certain XMLClearTag.
     * 
     * The value returned by this method is to be used as last parameter 
     * (parameter 'pos') of addChild, addText or addClear. This is usefull when 
     * you want to insert a childNode, a text or a XMLClearTag in the middle 
     * (at a specified position) of a XMLNode tree already constructed.
     */
    XMLElementPosition positionOfClear(int i=0) const;

    /**
     * \brief Returns the position of a certain XMLClearTag.
     * 
     * The value returned by this method is to be used as last parameter 
     * (parameter 'pos') of addChild, addText or addClear. This is usefull when 
     * you want to insert a childNode, a text or a XMLClearTag in the middle 
     * (at a specified position) of a XMLNode tree already constructed.
     */
    XMLElementPosition positionOfClear(XMLCSTR lpszValue) const;

    /**
     * \brief Returns the position of a certain XMLClearTag.
     * 
     * The value returned by this method is to be used as last parameter 
     * (parameter 'pos') of addChild, addText or addClear. This is usefull when 
     * you want to insert a childNode, a text or a XMLClearTag in the middle 
     * (at a specified position) of a XMLNode tree already constructed.
     */
    XMLElementPosition positionOfClear(XMLClear *a) const;

    /**
     * \brief Returns the position of a certain child node.
     * 
     * The value returned by this method is to be used as last parameter 
     * (parameter 'pos') of addChild, addText or addClear. This is usefull when 
     * you want to insert a childNode, a text or a XMLClearTag in the middle 
     * (at a specified position) of a XMLNode tree already constructed.
     */
    XMLElementPosition positionOfChildNode(int i=0) const;

    /**
     * \brief Returns the position of a certain child node.
     * 
     * The value returned by this method is to be used as last parameter 
     * (parameter 'pos') of addChild, addText or addClear. This is usefull when 
     * you want to insert a childNode, a text or a XMLClearTag in the middle 
     * (at a specified position) of a XMLNode tree already constructed.
     */
    XMLElementPosition positionOfChildNode(XMLNode x) const;

    /**
     * \brief Returns the position of the i<i>th</i> childNode with the 
     * specified name.
     * 
     * The value returned by this method is to be used as last parameter 
     * (parameter 'pos') of addChild, addText or addClear. This is usefull when 
     * you want to insert a childNode, a text or a XMLClearTag in the middle 
     * (at a specified position) of a XMLNode tree already constructed.<br>
     * <br>
     * If (name==NULL) the position of the i<i>th</i> childNode will be 
     * returned.
     */
    XMLElementPosition positionOfChildNode(XMLCSTR name, int i=0) const; // return the position of the ith childNode with the specified name
                                                                      // if (name==NULL) return the position of the ith childNode

    /**
     * \brief Enumeration for %XML character encoding.
     */
    typedef enum XMLCharEncoding { encoding_UTF8=1, encoding_ascii=2, encoding_ShiftJIS=3 } XMLCharEncoding;

    /**
     * \brief Sets global parameters affecting the string and file parsing.
     * 
     * The setGlobalOptions function allows you to change four global parameters 
     * that affect string and file parsing. First of all, you most-probably will 
     * never have to change these 4 global parameters.<br>
     * The return value of the setGlobalOptions function is "0" when there are 
     * no errors. If you try to set an unrecognized encoding then the return 
     * value will be "1" to signal an error.
     * 
     * \param guessWideCharChars If "guessWideCharChars=1" and if this library 
     * is compiled in WideChar mode, then the "parseFile" and "openFileHelper" 
     * functions will test if the file contains ASCII characters. If this is the 
     * case, then the file will be loaded and converted in memory to WideChar 
     * before being parsed. If "guessWideCharChars=0", no conversion will be 
     * performed.<br>
     * <br>
     * If "guessWideCharChars=1" and if this library is compiled in 
     * ASCII/UTF8/char* mode, then the "parseFile" and "openFileHelper" 
     * functions will test if the file contains WideChar characters. If this is 
     * the case, then the file will be loaded and converted in memory to
     * ASCII/UTF8/char* before being parsed. If "guessWideCharChars=0", no 
     * conversion will be performed.<br>
     * <br>
     * Sometimes, it's useful to set "guessWideCharChars=0" to disable any 
     * conversion because the test to detect the file-type (ASCII/UTF8/char* or 
     * WideChar) may fail (rarely).
     *
     * \param characterEncoding This parameter is only meaningful when compiling 
     * in char* mode (multibyte character mode). In wchar_t* (wide char mode), 
     * this parameter is ignored. This parameter should be one of the three 
     * currently recognized encodings: XMLNode::encoding_UTF8, 
     * XMLNode::encoding_ascii, XMLNode::encoding_ShiftJIS.
     *
     * \param dropWhiteSpace In most situations, text fields containing only 
     * white spaces (and carriage returns) are useless. Even more, these "empty" 
     * text fields are annoying because they increase the complexity of the 
     * user's code for parsing. So, 99% of the time, it's better to drop the 
     * "empty" text fields. However The %XML specification indicates that no 
     * white spaces should be lost when parsing the file. So to be perfectly 
     * XML-compliant, you should set dropWhiteSpace=0. A note of caution: if 
     * you set "dropWhiteSpace=0", the parser will be slower and your code will 
     * be more complex.
     *
     * \param removeCommentsInMiddleOfText Let's consider this code:
     * <pre><code>       XMLNode x=XMLNode::parseString("<a>foo<!-- hello -->bar<!DOCTYPE world >chu</a>","a");</code></pre>
     * If removeCommentsInMiddleOfText=0, then we will have:
     * <pre><code>        x.getText(0) -> "foo"
     *        x.getText(1) -> "bar"
     *        x.getText(2) -> "chu"
     *        x.getClear(0) --> "<!-- hello -->"
     *        x.getClear(1) --> "<!DOCTYPE world >"</code></pre>
     * If removeCommentsInMiddleOfText=1, then we will have:
     * <pre><code>        x.getText(0) -> "foobar"
     *        x.getText(1) -> "chu"
     *        x.getClear(0) --> "<!DOCTYPE world >"</code></pre>
     */
    static char setGlobalOptions(XMLCharEncoding characterEncoding=XMLNode::encoding_UTF8, char guessWideCharChars=1,
                                 char dropWhiteSpace=1, char removeCommentsInMiddleOfText=1);

    /**
     * \brief This function tries to guess the character encoding.
     * 
     * You most-probably will never have to use this function. It then returns 
     * the appropriate value of the global parameter "characterEncoding". The 
     * guess is based on the content of a buffer of length "bufLen" bytes that 
     * contains the first bytes (minimum 25 bytes; 200 bytes is a good value) of 
     * the file to be parsed. The "openFileHelper" function is using this 
     * function to automatically compute the value of the "characterEncoding" 
     * global parameter. There are several heuristics used to do the guess. One 
     * of the heuristic is based on the "encoding" attribute. The original %XML 
     * specifications forbids to use this attribute to do the guess but you can 
     * still use it if you set "useXMLEncodingAttribute" to 1 (this is the 
     * default behavior and the behavior of most parsers). If an inconsistency 
     * in the encoding is detected, then the return value is "0".
     * 
     * @see setGlobalOptions()
     */
    static XMLCharEncoding guessCharEncoding(void *buffer, int bufLen, char useXMLEncodingAttribute=1);

  private:

// these are functions and structures used internally by the XMLNode class (don't bother about them):

      typedef struct XMLNodeDataTag // to allow shallow copy and "intelligent/smart" pointers (automatic delete):
      {
          XMLCSTR                lpszName;        // Element name (=NULL if root)
          int                    nChild,          // Number of child nodes
                                 nText,           // Number of text fields
                                 nClear,          // Number of Clear fields (comments)
                                 nAttribute;      // Number of attributes
          char                   isDeclaration;   // Whether node is an XML declaration - '<?xml ?>'
          struct XMLNodeDataTag  *pParent;        // Pointer to parent element (=NULL if root)
          XMLNode                *pChild;         // Array of child nodes
          XMLCSTR                *pText;          // Array of text fields
          XMLClear               *pClear;         // Array of clear fields
          XMLAttribute           *pAttribute;     // Array of attributes
          int                    *pOrder;         // order of the child_nodes,text_fields,clear_fields
          int                    ref_count;       // for garbage collection (smart pointers)
      } XMLNodeData;
      XMLNodeData *d;

      char parseClearTag(void *px, void *pa);
      char maybeAddTxT(void *pa, XMLCSTR tokenPStr);
      int ParseXMLElement(void *pXML);
      void *addToOrder(int memInc, int *_pos, int nc, void *p, int size, XMLElementType xtype);
      int indexText(XMLCSTR lpszValue) const;
      int indexClear(XMLCSTR lpszValue) const;
      XMLNode addChild_priv(int,XMLSTR,char,int);
      XMLAttribute *addAttribute_priv(int,XMLSTR,XMLSTR);
      XMLCSTR addText_priv(int,XMLSTR,int);
      XMLClear *addClear_priv(int,XMLSTR,XMLCSTR,XMLCSTR,int);
      void emptyTheNode(char force);
      static inline XMLElementPosition findPosition(XMLNodeData *d, int index, XMLElementType xtype);
      static int CreateXMLStringR(XMLNodeData *pEntry, XMLSTR lpszMarker, int nFormat);
      static int removeOrderElement(XMLNodeData *d, XMLElementType t, int index);
      static void exactMemory(XMLNodeData *d);
      static int detachFromParent(XMLNodeData *d);
} XMLNode;

/// This structure is given by the function "enumContents".
typedef struct XMLNodeContents
{
    /// This dictates what's the content of the XMLNodeContent
    enum XMLElementType etype;
    // should be an union to access the appropriate data.
    // compiler does not allow union of object with constructor... too bad.
    XMLNode child;
    XMLAttribute attrib;
    XMLCSTR text;
    XMLClear clear;

} XMLNodeContents;

XMLDLLENTRY void freeXMLString(XMLSTR t); // {free(t);}

/**
 * \brief Duplicates (copies in a new allocated buffer) the source string. 
 * 
 * This is a very handy function when used with all the "XMLNode::*_WOSD" 
 * functions. (If (cbData!=0) then cbData is the number of chars to duplicate)
 */
XMLDLLENTRY XMLSTR stringDup(XMLCSTR source, int cbData=0);

// The next 4 functions are equivalents to the atoi, atol, atof functions.
// The only difference is: If the variable "xmlString" is NULL, than the return value
// is "defautValue". These 4 functions are only here as "convenience" functions for the
// user (they are not used inside the XMLparser). If you don't need them, you can
// delete them without any trouble.

/**
 * \brief Like C's atoi(), this converts a xmlString into a character.
 * 
 * If the variable "xmlString" is NULL, than the return value is "defautValue".
 * <br><br>
 * This function is only here as "convenience" functions for the user (it is not 
 * used inside the XMLparser). If you don't need it, you can delete iw without 
 * any trouble.
 */
XMLDLLENTRY char xmltoc(XMLCSTR xmlString, char defautValue=0);

/**
 * \brief Like C's atoi(), this converts a xmlString into an integer.
 * 
 * If the variable "xmlString" is NULL, than the return value is "defautValue".
 * <br><br>
 * This function is only here as "convenience" functions for the user (it is not 
 * used inside the XMLparser). If you don't need it, you can delete iw without 
 * any trouble.
 */
XMLDLLENTRY int xmltoi(XMLCSTR xmlString, int defautValue=0);

/**
 * \brief Like C's atol(), this converts a xmlString into an long value.
 * 
 * If the variable "xmlString" is NULL, than the return value is "defautValue".
 * <br><br>
 * This function is only here as "convenience" functions for the user (it is not 
 * used inside the XMLparser). If you don't need it, you can delete iw without 
 * any trouble.
 */
XMLDLLENTRY long xmltol(XMLCSTR xmlString, long defautValue=0);

/**
 * \brief Like C's atof(), this converts a xmlString into an double value.
 * 
 * If the variable "xmlString" is NULL, than the return value is "defautValue".
 * <br><br>
 * This function is only here as "convenience" functions for the user (it is not 
 * used inside the XMLparser). If you don't need it, you can delete iw without 
 * any trouble.
 */
XMLDLLENTRY double  xmltof(XMLCSTR xmlString,double defautValue=.0);

/**
 * \brief Returns the given string or a default value if the string is NULL. 
 * 
 * This function is only here as "convenience" functions for the user (it is not 
 * used inside the XMLparser). If you don't need it, you can delete iw without 
 * any trouble.
 */
XMLDLLENTRY XMLCSTR xmltoa(XMLCSTR xmlString,XMLCSTR defautValue=_CXML(""));

/**
 * \brief Processes strings to replace special characters by their %XML 
 * equivalent.
 * 
 * This class is processing strings so that all the characters &,",',<,> are 
 * replaced by their %XML equivalent: &amp;, &quot;, &apos;, &lt;, &gt;.<br> 
 * This  class is useful when creating from scratch an %XML file using the 
 * "printf", "fprintf", "cout",... functions. If you are creating from scratch 
 * an %XML file using the provided XMLNode class you must not use the 
 * "ToXMLStringTool" class (the "XMLNode" class does the processing job for you 
 * during rendering). Using the "ToXMLStringTool class" and the "fprintf 
 * function" is THE most efficient way to produce VERY large %XML documents VERY 
 * fast.
 */
typedef struct XMLDLLENTRY ToXMLStringTool
{
public:
    ToXMLStringTool(): buf(NULL),buflen(0){}
    ~ToXMLStringTool();
    void freeBuffer();

    XMLSTR toXML(XMLCSTR source);

    /**
     * \brief Converts string "source" to string "dest" (deprecated).
     * \deprecated
     * 
     * This function is deprecated because there is a possibility of 
     * "destination-buffer-overflow".
     */
    static XMLSTR toXMLUnSafe(XMLSTR dest,XMLCSTR source);
    
    /**
     * \brief Returns the length of a string (deprecated). 
     * \deprecated
     * 
     * This function is deprecated. Use "toXML" instead.
     */
    static int lengthXMLString(XMLCSTR source);

private:
    XMLSTR buf;
    int buflen;
}ToXMLStringTool;

/**
 * \brief Allows you to include any binary data (images, sounds,...) into an 
 * %XML document using "Base64 encoding".
 * 
 * This class is completely separated from the rest of the xmlParser library and 
 * can be removed without any problem. To include some binary data into an %XML 
 * file, you must convert the binary data into standard text (using "encode"). 
 * To retrieve the original binary data from the b64-encoded text included 
 * inside the %XML file use "decode". Alternatively, these functions can also be 
 * used to "encrypt/decrypt" some critical data contained inside the %XML (it's 
 * not a strong encryption at all, but sometimes it can be useful).
 */
typedef struct XMLDLLENTRY XMLParserBase64Tool {
public:
    XMLParserBase64Tool(): buf(NULL),buflen(0){}
    ~XMLParserBase64Tool();
    void freeBuffer();

	/**
	 * \brief Returns the length of the base64 string that encodes a data buffer 
	 * of size inBufLen bytes.
	 * 
	 * If "formatted" parameter is true, some space will be reserved for a 
	 * carriage-return every 72 chars.
	 */
	static int encodeLength(int inBufLen, char formatted=0);

	/**
	 * \brief Returns a string containing the base64 encoding of "inByteLen" 
	 * bytes from "inByteBuf".
	 * 
	 * If "formatted" parameter is true, then there will be a carriage-return 
	 * every 72 chars. The string will be free'd when the XMLParserBase64Tool 
	 * object is deleted. All returned strings are sharing the same memory 
	 * space.
	 */
	XMLSTR encode(unsigned char *inByteBuf, unsigned int inByteLen, char formatted=0);

	/**
	 * \brief Returns the number of bytes which will be decoded from "inString".
	 */
	static unsigned int decodeSize(XMLCSTR inString, XMLError *xe=NULL);

	/**
	 * \brief Returns a pointer to a buffer containing the binary data decoded from "inString"
	 * 
	 * If "inString" is malformed NULL will be returned.<br>
	 * The output buffer will be free'd when the XMLParserBase64Tool object is 
	 * deleted.<br>
	 * All output buffer are sharing the same memory space.
	 */
	unsigned char* decode(XMLCSTR inString, int *outByteLen=NULL, XMLError *xe=NULL);

	/**
	 * \brief Decodes data from "inString" to "outByteBuf".
	 * \deprecated
	 * 
	 * You need to provide the size (in byte) of "outByteBuf" in 
	 * "inMaxByteOutBuflen". If "outByteBuf" is not large enough or if data is 
	 * malformed, then "FALSE" will be returned; otherwise "TRUE".
	 */
	static unsigned char decode(XMLCSTR inString, unsigned char *outByteBuf, int inMaxByteOutBuflen, XMLError *xe=NULL);

private:
    void *buf;
    int buflen;
    void alloc(int newsize);
}XMLParserBase64Tool;

#undef XMLDLLENTRY

#endif

/* Aladdin Free Public License
(Version 8, November 18, 1999)

Copyright (C) 1994, 1995, 1997, 1998, 1999 Aladdin Enterprises,
Menlo Park, California, U.S.A. All rights reserved.

    *NOTE:* This License is not the same as any of the GNU Licenses
    <http://www.gnu.org/copyleft/gpl.html> published by the Free
    Software Foundation <http://www.gnu.org/>. Its terms are
    substantially different from those of the GNU Licenses. If you are
    familiar with the GNU Licenses, please read this license with extra
    care. 

Aladdin Enterprises hereby grants to anyone the permission to apply this
License to their own work, as long as the entire License (including the
above notices and this paragraph) is copied with no changes, additions,
or deletions except for changing the first paragraph of Section 0 to
include a suitable description of the work to which the license is being
applied and of the person or entity that holds the copyright in the
work, and, if the License is being applied to a work created in a
country other than the United States, replacing the first paragraph of
Section 6 with an appropriate reference to the laws of the appropriate
country.


    0. Subject Matter

This License applies to the computer program known as "XMLParser library". 
The "Program", below, refers to such program. The Program
is a copyrighted work whose copyright is held by Frank Vanden Berghen
(the "Licensor"). 

A "work based on the Program" means either the Program or any derivative
work of the Program, as defined in the United States Copyright Act of
1976, such as a translation or a modification.

* BY MODIFYING OR DISTRIBUTING THE PROGRAM (OR ANY WORK BASED ON THE
PROGRAM), YOU INDICATE YOUR ACCEPTANCE OF THIS LICENSE TO DO SO, AND ALL
ITS TERMS AND CONDITIONS FOR COPYING, DISTRIBUTING OR MODIFYING THE
PROGRAM OR WORKS BASED ON IT. NOTHING OTHER THAN THIS LICENSE GRANTS YOU
PERMISSION TO MODIFY OR DISTRIBUTE THE PROGRAM OR ITS DERIVATIVE WORKS.
THESE ACTIONS ARE PROHIBITED BY LAW. IF YOU DO NOT ACCEPT THESE TERMS
AND CONDITIONS, DO NOT MODIFY OR DISTRIBUTE THE PROGRAM. *


    1. Licenses.

Licensor hereby grants you the following rights, provided that you
comply with all of the restrictions set forth in this License and
provided, further, that you distribute an unmodified copy of this
License with the Program:

(a)
    You may copy and distribute literal (i.e., verbatim) copies of the
    Program's source code as you receive it throughout the world, in any
    medium. 
(b)
    You may modify the Program, create works based on the Program and
    distribute copies of such throughout the world, in any medium. 


    2. Restrictions.

This license is subject to the following restrictions:

(a)
    Distribution of the Program or any work based on the Program by a
    commercial organization to any third party is prohibited if any
    payment is made in connection with such distribution, whether
    directly (as in payment for a copy of the Program) or indirectly (as
    in payment for some service related to the Program, or payment for
    some product or service that includes a copy of the Program "without
    charge"; these are only examples, and not an exhaustive enumeration
    of prohibited activities). The following methods of distribution
    involving payment shall not in and of themselves be a violation of
    this restriction:

    (i)
        Posting the Program on a public access information storage and
        retrieval service for which a fee is received for retrieving
        information (such as an on-line service), provided that the fee
        is not content-dependent (i.e., the fee would be the same for
        retrieving the same volume of information consisting of random
        data) and that access to the service and to the Program is
        available independent of any other product or service. An
        example of a service that does not fall under this section is an
        on-line service that is operated by a company and that is only
        available to customers of that company. (This is not an
        exhaustive enumeration.) 
    (ii)
        Distributing the Program on removable computer-readable media,
        provided that the files containing the Program are reproduced
        entirely and verbatim on such media, that all information on
        such media be redistributable for non-commercial purposes
        without charge, and that such media are distributed by
        themselves (except for accompanying documentation) independent
        of any other product or service. Examples of such media include
        CD-ROM, magnetic tape, and optical storage media. (This is not
        intended to be an exhaustive list.) An example of a distribution
        that does not fall under this section is a CD-ROM included in a
        book or magazine. (This is not an exhaustive enumeration.) 

(b)
    Activities other than copying, distribution and modification of the
    Program are not subject to this License and they are outside its
    scope. Functional use (running) of the Program is not restricted,
    and any output produced through the use of the Program is subject to
    this license only if its contents constitute a work based on the
    Program (independent of having been made by running the Program). 
(c)
    You must meet all of the following conditions with respect to any
    work that you distribute or publish that in whole or in part
    contains or is derived from the Program or any part thereof ("the
    Work"):

    (i)
        If you have modified the Program, you must cause the Work to
        carry prominent notices stating that you have modified the
        Program's files and the date of any change. In each source file
        that you have modified, you must include a prominent notice that
        you have modified the file, including your name, your e-mail
        address (if any), and the date and purpose of the change; 
    (ii)
        You must cause the Work to be licensed as a whole and at no
        charge to all third parties under the terms of this License; 
    (iii)
        If the Work normally reads commands interactively when run, you
        must cause it, at each time the Work commences operation, to
        print or display an announcement including an appropriate
        copyright notice and a notice that there is no warranty (or
        else, saying that you provide a warranty). Such notice must also
        state that users may redistribute the Work only under the
        conditions of this License and tell the user how to view the
        copy of this License included with the Work. (Exceptions: if the
        Program is interactive but normally prints or displays such an
        announcement only at the request of a user, such as in an "About
        box", the Work is required to print or display the notice only
        under the same circumstances; if the Program itself is
        interactive but does not normally print such an announcement,
        the Work is not required to print an announcement.); 
    (iv)
        You must accompany the Work with the complete corresponding
        machine-readable source code, delivered on a medium customarily
        used for software interchange. The source code for a work means
        the preferred form of the work for making modifications to it.
        For an executable work, complete source code means all the
        source code for all modules it contains, plus any associated
        interface definition files, plus the scripts used to control
        compilation and installation of the executable code. If you
        distribute with the Work any component that is normally
        distributed (in either source or binary form) with the major
        components (compiler, kernel, and so on) of the operating system
        on which the executable runs, you must also distribute the
        source code of that component if you have it and are allowed to
        do so; 
    (v)
        If you distribute any written or printed material at all with
        the Work, such material must include either a written copy of
        this License, or a prominent written indication that the Work is
        covered by this License and written instructions for printing
        and/or displaying the copy of the License on the distribution
        medium; 
    (vi)
        You may not impose any further restrictions on the recipient's
        exercise of the rights granted herein. 

If distribution of executable or object code is made by offering the
equivalent ability to copy from a designated place, then offering
equivalent ability to copy the source code from the same place counts as
distribution of the source code, even though third parties are not
compelled to copy the source code along with the object code.


      3. Reservation of Rights.

No rights are granted to the Program except as expressly set forth
herein. You may not copy, modify, sublicense, or distribute the Program
except as expressly provided under this License. Any attempt otherwise
to copy, modify, sublicense or distribute the Program is void, and will
automatically terminate your rights under this License. However, parties
who have received copies, or rights, from you under this License will
not have their licenses terminated so long as such parties remain in
full compliance.


      4. Other Restrictions.

If the distribution and/or use of the Program is restricted in certain
countries for any reason, Licensor may add an explicit geographical
distribution limitation excluding those countries, so that distribution
is permitted only in or among countries not thus excluded. In such case,
this License incorporates the limitation as if written in the body of
this License.


      5. Limitations.

* THE PROGRAM IS PROVIDED TO YOU "AS IS," WITHOUT WARRANTY. THERE IS NO
WARRANTY FOR THE PROGRAM, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS. THE
ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH
YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL
NECESSARY SERVICING, REPAIR OR CORRECTION. *

* IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL LICENSOR, OR ANY OTHER PARTY WHO MAY MODIFY AND/OR REDISTRIBUTE THE
PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS
OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR
THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER
PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES. *


      6. General.

This License is governed by the laws of Belgium., excluding choice of 
law rules.

If any part of this License is found to be in conflict with the law,
that part shall be interpreted in its broadest meaning consistent with
the law, and no other parts of the License shall be affected.

For United States Government users, the Program is provided with
*RESTRICTED RIGHTS*. If you are a unit or agency of the United States
Government or are acquiring the Program for any such unit or agency, the
following apply:

    If the unit or agency is the Department of Defense ("DOD"), the
    Program and its documentation are classified as "commercial computer
    software" and "commercial computer software documentation"
    respectively and, pursuant to DFAR Section 227.7202, the Government
    is acquiring the Program and its documentation in accordance with
    the terms of this License. If the unit or agency is other than DOD,
    the Program and its documentation are classified as "commercial
    computer software" and "commercial computer software documentation"
    respectively and, pursuant to FAR Section 12.212, the Government is
    acquiring the Program and its documentation in accordance with the
    terms of this License. 
*/
