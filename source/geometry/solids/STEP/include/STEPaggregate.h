#ifndef STEPAGGREGATE_H
#define STEPAGGREGATE_H

/*
* NIST STEP Core Class Library
* clstepcore/STEPaggregate.h
* April 1997
* K. C. Morris
* David Sauder

* Development of this software was funded by the United States Government,
* and is not subject to copyright.
*/

/* $Id: STEPaggregate.h,v 1.4 2000/01/21 13:42:31 gcosmo Exp $ */

#ifdef __OSTORE__
#include <ostore/ostore.hh>    // Required to access ObjectStore Class Library
#endif

#ifdef __O3DB__
#include <OpenOODB.h>
#endif

class InstMgr;
class STEPaggregate;
#ifndef _ODI_OSSG_
class LOGICAL;
class BOOL;
#endif
class TypeDescriptor;

#include <errordesc.h>
#include <SingleLinkList.h>
#include <baseType.h>
#include <sdai.h>
#include <STEPundefined.h>
#include <scl_string.h>
#include "G4ios.hh"

//class SCLP23(Application_instance);
//#define 	S_ENTITY_NULL	&NilSTEPentity
//extern SCLP23(Application_instance) NilSTEPentity;


#define 	AGGR_NULL	&NilSTEPaggregate
extern STEPaggregate NilSTEPaggregate;

//typedef unsigned short BOOLEAN;
class SingleLinkNode;

/******************************************************************************
 **
 *****************************************************************************/

typedef STEPaggregate * STEPaggregateH;
typedef STEPaggregate * STEPaggregate_ptr;
typedef STEPaggregate_ptr STEPaggregate_var;

class STEPaggregate :  public SingleLinkList 
{
  protected:
    int _null;

  protected:

    virtual Severity ReadValue(G4std::istream &in, ErrorDescriptor *err,
			       const TypeDescriptor *elem_type,
			       InstMgr *insts, int addFileId =0,
			       int assignVal =1, int ExchangeFileFormat =1,
			       const char *currSch =0);
  public:

    int is_null() { return _null; }

    virtual Severity AggrValidLevel(const char *value, ErrorDescriptor *err,
			    const TypeDescriptor *elem_type, InstMgr *insts,
			    int optional, char *tokenList, int addFileId =0,
			    int clearError =0);

    virtual Severity AggrValidLevel(G4std::istream &in, ErrorDescriptor *err,
			    const TypeDescriptor *elem_type, InstMgr *insts,
			    int optional, char *tokenList, int addFileId =0,
			    int clearError =0);

// INPUT
    virtual Severity StrToVal(const char *s, ErrorDescriptor *err =0,
			      const TypeDescriptor *elem_type =0,
			      InstMgr *insts =0, int addFileId = 0);
    virtual Severity STEPread (G4std::istream& in, ErrorDescriptor *err,
			       const TypeDescriptor *elem_type =0,
			       InstMgr *insts =0, int addFileId =0,
			       const char *currSch =0);
// OUTPUT
    virtual const char *asStr(SCLstring & s) const;
    virtual void STEPwrite (G4std::ostream& out =G4cout, const char * =0) const;

//    SingleLinkNode * GetHead () const
//	{ return (STEPnode *) SingleLinkList::GetHead(); }

    virtual SingleLinkNode * NewNode ();  
    void AddNode (SingleLinkNode *);
    void Empty ();
    
    STEPaggregate ();
    virtual ~STEPaggregate ();

// COPY - defined in subtypes
    virtual STEPaggregate& ShallowCopy (const STEPaggregate&);

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif

};									      

/******************************************************************
 ** Class:  GenericAggregate
 ** Description:  This class supports LIST OF: 
 **	SELECT_TYPE, BINARY_TYPE, GENERIC_TYPE, ENUM_TYPE, UNKNOWN_TYPE type
 ******************************************************************/

class GenericAggregate  :  public STEPaggregate 
{
  public:
    virtual SingleLinkNode *NewNode();
    virtual STEPaggregate& ShallowCopy (const STEPaggregate&);

    GenericAggregate();
    virtual ~GenericAggregate();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};
typedef  GenericAggregate * GenericAggregateH;
typedef  GenericAggregate * GenericAggregate_ptr;
typedef  GenericAggregate_ptr GenericAggregate_var;

/******************************************************************************
 **
 *****************************************************************************/

class EntityAggregate  :  public  STEPaggregate 
{
  public:
    virtual Severity ReadValue(G4std::istream &in, ErrorDescriptor *err, 
			       const TypeDescriptor *elem_type, 
			       InstMgr *insts, int addFileId =0, 
			       int assignVal =1, int ExchangeFileFormat =1,
			       const char *currSch =0);

    virtual  SingleLinkNode *NewNode();
    virtual  STEPaggregate&  ShallowCopy (const STEPaggregate&);

    EntityAggregate ();
    virtual ~EntityAggregate ();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};
typedef   EntityAggregate * EntityAggregateH;
typedef   EntityAggregate * EntityAggregate_ptr;
typedef   EntityAggregate_ptr EntityAggregate_var;

/******************************************************************
 ** Class:  SelectAggregate
 ** Description:  this is a minimal representions for a collection of 
 **	SCLP23(Select)
 ******************************************************************/
class SelectAggregate  :  public STEPaggregate 
{
  public:
    virtual Severity ReadValue(G4std::istream &in, ErrorDescriptor *err, 
			       const TypeDescriptor *elem_type, 
			       InstMgr *insts, int addFileId =0, 
			       int assignVal =1, int ExchangeFileFormat =1,
			       const char *currSch =0);

    virtual SingleLinkNode *NewNode ();
    virtual STEPaggregate& ShallowCopy  (const STEPaggregate&);

    SelectAggregate ();
    virtual ~SelectAggregate ();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};
typedef  SelectAggregate *  SelectAggregateH;
typedef  SelectAggregate *  SelectAggregate_ptr;
typedef  SelectAggregate_ptr SelectAggregate_var;

/******************************************************************
 ** Class:  StringAggregate
 ** Description:  This class supports LIST OF STRING type
 ******************************************************************/
class StringAggregate  :  public STEPaggregate 
{
  public:
    virtual SingleLinkNode *NewNode();
    virtual STEPaggregate& ShallowCopy (const STEPaggregate&);

    StringAggregate();
    virtual ~StringAggregate();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};
typedef  StringAggregate * StringAggregateH;
typedef  StringAggregate * StringAggregate_ptr;
typedef  StringAggregate_ptr StringAggregate_var;


/******************************************************************
 ** Class:  BinaryAggregate
 ** Description:  This class supports LIST OF BINARY type
 ******************************************************************/
class BinaryAggregate  :  public STEPaggregate 
{
  public:
    virtual SingleLinkNode *NewNode();
    virtual STEPaggregate& ShallowCopy (const STEPaggregate&);

    BinaryAggregate();
    virtual ~BinaryAggregate();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};
typedef  BinaryAggregate * BinaryAggregateH;
typedef  BinaryAggregate * BinaryAggregate_ptr;
typedef  BinaryAggregate_ptr BinaryAggregate_var;

/******************************************************************
 ** Class:  EnumAggregate
 ** Description:  this is a minimal representions for a collection of 
 **	SCLP23(Enum)
 ******************************************************************/
class EnumAggregate  :  public STEPaggregate 
{
  public:
    virtual SingleLinkNode *NewNode ();
    virtual STEPaggregate& ShallowCopy  (const STEPaggregate&);

    EnumAggregate ();
  virtual ~EnumAggregate ();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};
typedef  EnumAggregate *  EnumAggregateH;
typedef  EnumAggregate *  EnumAggregate_ptr;
typedef  EnumAggregate_ptr EnumAggregate_var;

class LOGICALS  : public EnumAggregate  
{
  public:
    virtual SingleLinkNode * NewNode ();  

    LOGICALS();
    virtual ~LOGICALS();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};
typedef  LOGICALS *  LogicalsH;
typedef  LOGICALS *  LOGICALS_ptr;
typedef  LOGICALS_ptr LOGICALS_var;
#ifdef __OSTORE__
LOGICALS * create_LOGICALS(os_database *db);
#else
LOGICALS * create_LOGICALS();
#endif


class BOOLS  : public EnumAggregate  
{
  public:
    virtual SingleLinkNode * NewNode ();  

    BOOLS();
    virtual ~BOOLS();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};

//typedef  BOOLS *  BooleansH;
typedef  BOOLS *  BOOLS_ptr;
typedef  BOOLS_ptr BOOLS_var;

#ifdef __OSTORE__
BOOLS * create_BOOLS(os_database *db);
#else
BOOLS * create_BOOLS();
#endif

class RealAggregate  : public STEPaggregate  {

  public:
    virtual SingleLinkNode *NewNode();
    virtual STEPaggregate& ShallowCopy  (const STEPaggregate&);

    RealAggregate();
    virtual ~RealAggregate();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};
typedef  RealAggregate *  RealAggregateH;
typedef  RealAggregate *  RealAggregate_ptr;
typedef  RealAggregate_ptr RealAggregate_var;

class IntAggregate  : public STEPaggregate  {

  public:
    virtual SingleLinkNode *NewNode();
    virtual STEPaggregate& ShallowCopy  (const STEPaggregate&);

    IntAggregate();
    virtual ~IntAggregate();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};
typedef  IntAggregate *  IntAggregateH;
typedef  IntAggregate *  IntAggregate_ptr;
typedef  IntAggregate_ptr IntAggregate_var;

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

class STEPnode :  public SingleLinkNode  {
  protected:
    int _null;

  public:
    int is_null() { return _null; } 
    void set_null() { _null = 1; } 

//	INPUT
    virtual Severity StrToVal(const char *s, ErrorDescriptor *err);
    virtual Severity StrToVal(G4std::istream &in, ErrorDescriptor *err);

    virtual Severity STEPread(const char *s, ErrorDescriptor *err);
    virtual Severity STEPread(G4std::istream &in, ErrorDescriptor *err);

//	OUTPUT
    virtual const char *asStr(SCLstring & s);
    virtual const char *STEPwrite(SCLstring &s, const char * =0);
    virtual void STEPwrite (G4std::ostream& out =G4cout);
#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};
typedef  STEPnode *  STEPnodeH;

/******************************************************************
 ** Class:  GenericNode
 ** Description:  This class is for the Nodes of GenericAggregates
 ******************************************************************/

class GenericAggrNode  : public STEPnode {
  public:

    SCLundefined value;

  public:
//	INPUT
    virtual Severity StrToVal(const char *s, ErrorDescriptor *err);
    virtual Severity StrToVal(G4std::istream &in, ErrorDescriptor *err);

    virtual Severity STEPread(const char *s, ErrorDescriptor *err);
    virtual Severity STEPread(G4std::istream &in, ErrorDescriptor *err);

//	OUTPUT
    virtual const char *asStr(SCLstring & s);
    virtual const char *STEPwrite(SCLstring &s, const char * =0);
    virtual void 	STEPwrite (G4std::ostream& out =G4cout);

//	CONSTRUCTORS
    GenericAggrNode (const char *str);
    GenericAggrNode (GenericAggrNode& gan);
    GenericAggrNode ();
    ~GenericAggrNode ();

    virtual SingleLinkNode *	NewNode ();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif

};

/////////////////////////////////////////////////////////////////////////////

class EntityNode  : public STEPnode {
  public:
    SCLP23(Application_instance) * node;

// INPUT
    virtual Severity StrToVal(const char *s, ErrorDescriptor *err, 
			      const TypeDescriptor *elem_type,
			      InstMgr *insts, int addFileId = 0);
    virtual Severity StrToVal(G4std::istream &in, ErrorDescriptor *err, 
			      const TypeDescriptor *elem_type,
			      InstMgr *insts, int addFileId = 0);

    virtual Severity STEPread(const char *s, ErrorDescriptor *err, 
			      const TypeDescriptor *elem_type,
			      InstMgr *insts, int addFileId = 0);
    virtual Severity STEPread(G4std::istream &in, ErrorDescriptor *err, 
			      const TypeDescriptor *elem_type,
			      InstMgr *insts, int addFileId = 0);
//	OUTPUT
    virtual const char *asStr(SCLstring & s);
    virtual const char *STEPwrite (SCLstring &s, const char * =0);
    virtual void 	STEPwrite (G4std::ostream& out =G4cout);

//	CONSTRUCTORS
    EntityNode (SCLP23(Application_instance) * e);
    EntityNode();
    ~EntityNode();

    virtual SingleLinkNode *	NewNode ();

    // Calling these funtions is an error.
    Severity StrToVal(const char *s, ErrorDescriptor *err)
    {
	G4cerr << "Internal error:  " << __FILE__ <<  __LINE__
	    << "\n" << _POC_ "\n";
	return StrToVal(s, err, 0, 0, 0);
    }
    Severity StrToVal(G4std::istream &in, ErrorDescriptor *err)
    {
	G4cerr << "Internal error:  " << __FILE__ <<  __LINE__
	    << "\n" << _POC_ "\n";
	return StrToVal(in, err, 0, 0, 0);
    }

    Severity STEPread(const char *s, ErrorDescriptor *err)
    {
	G4cerr << "Internal error:  " << __FILE__ <<  __LINE__
	    << "\n" << _POC_ "\n";
	return STEPread(s, err, 0, 0, 0);
    }
    Severity STEPread(G4std::istream &in, ErrorDescriptor *err)
    {
	G4cerr << "Internal error:  " << __FILE__ <<  __LINE__
	    << "\n" << _POC_ "\n";
	return STEPread(in, err, 0, 0, 0);
    }

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};

///////////////////////////////////////////////////////////////////////////

/******************************************************************
 ** Class:  SelectNode
 ** Description:  this is a minimal representions for node in lists of
 **	SCLP23(Select)
 ******************************************************************/

class SelectNode  : public STEPnode {
  public:

    SCLP23(Select) * node;

  public:
//	INPUT
    virtual Severity StrToVal(const char *s, ErrorDescriptor *err, 
			      const TypeDescriptor *elem_type,
			      InstMgr *insts, int addFileId = 0);
    virtual Severity StrToVal(G4std::istream &in, ErrorDescriptor *err, 
			      const TypeDescriptor *elem_type,
			      InstMgr *insts, int addFileId = 0,
			      const char *currSch =0);

    virtual Severity STEPread(const char *s, ErrorDescriptor *err, 
			      const TypeDescriptor *elem_type,
			      InstMgr *insts, int addFileId = 0);
    virtual Severity STEPread(G4std::istream &in, ErrorDescriptor *err, 
			      const TypeDescriptor *elem_type,
			      InstMgr *insts, int addFileId = 0,
			      const char *currSch =0);
//	OUTPUT
    virtual const char *asStr(SCLstring & s);
    virtual const char *STEPwrite (SCLstring &s, const char * =0);
    virtual void 	STEPwrite (G4std::ostream& out =G4cout);

//	CONSTRUCTORS
    SelectNode(SCLP23(Select) * s);
    SelectNode();
    ~SelectNode();

    virtual SingleLinkNode *	NewNode ();

    // Calling these funtions is an error.
    Severity StrToVal(const char *s, ErrorDescriptor *err)
    {
	G4cerr << "Internal error:  " << __FILE__ <<  __LINE__
	     << "\n" << _POC_ "\n";
	return StrToVal(s, err, 0, 0, 0);
    }
    Severity StrToVal(G4std::istream &in, ErrorDescriptor *err)
    {
	G4cerr << "Internal error:  " << __FILE__ <<  __LINE__
	     << "\n" << _POC_ "\n";
	return StrToVal(in, err, 0, 0, 0);
    }

    Severity STEPread(const char *s, ErrorDescriptor *err)
    {
	G4cerr << "Internal error:  " << __FILE__ <<  __LINE__
	     << "\n" << _POC_ "\n";
	return STEPread(s, err, 0, 0, 0);
    }
    Severity STEPread(G4std::istream &in, ErrorDescriptor *err)
    {
	G4cerr << "Internal error:  " << __FILE__ <<  __LINE__
	     << "\n" << _POC_ "\n";
	return STEPread(in, err, 0, 0, 0);
    }

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};

///////////////////////////////////////////////////////////////////////////

/******************************************************************
 ** Class:  StringNode
 ** Description:  This class is for the Nodes of StringAggregates
 ******************************************************************/

class StringNode  : public STEPnode {
  public:

    SCLP23(String) value;

  public:
//	INPUT
    virtual Severity StrToVal(const char *s, ErrorDescriptor *err);
    virtual Severity StrToVal(G4std::istream &in, ErrorDescriptor *err);

    virtual Severity STEPread(const char *s, ErrorDescriptor *err);
    virtual Severity STEPread(G4std::istream &in, ErrorDescriptor *err);

//	OUTPUT
    virtual const char *asStr(SCLstring & s);
    virtual const char *STEPwrite (SCLstring &s, const char * =0);
    virtual void 	STEPwrite (G4std::ostream& out =G4cout);

//	CONSTRUCTORS
    StringNode(StringNode& sn);
    StringNode (const char * sStr);
    StringNode();
    ~StringNode();

    virtual SingleLinkNode *	NewNode ();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};

///////////////////////////////////////////////////////////////////////////

/******************************************************************
 ** Class:  BinaryNode
 ** Description:  This class is for the Nodes of BinaryAggregates
 ******************************************************************/

class BinaryNode  : public STEPnode {
  public:

    SCLP23(Binary) value;

  public:
//	INPUT
    virtual Severity StrToVal(const char *s, ErrorDescriptor *err);
    virtual Severity StrToVal(G4std::istream &in, ErrorDescriptor *err);

    virtual Severity STEPread(const char *s, ErrorDescriptor *err);
    virtual Severity STEPread(G4std::istream &in, ErrorDescriptor *err);

//	OUTPUT
    virtual const char *asStr(SCLstring & s);
    virtual const char *STEPwrite (SCLstring &s, const char * =0);
    virtual void 	STEPwrite (G4std::ostream& out =G4cout);

//	CONSTRUCTORS
    BinaryNode(BinaryNode& bn);
    BinaryNode (const char * sStr);
    BinaryNode();
    ~BinaryNode();

    virtual SingleLinkNode *	NewNode ();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};

/******************************************************************
 ** Class:  EnumNode
 ** Description:  this is a minimal representions for node in lists of
 **	SCLP23(Enum)
 ******************************************************************/

class EnumNode  : public STEPnode {
  public:

    SCLP23(Enum) * node;

  public:
//	INPUT
    virtual Severity StrToVal(const char *s, ErrorDescriptor *err);
    virtual Severity StrToVal(G4std::istream &in, ErrorDescriptor *err);

    virtual Severity STEPread(const char *s, ErrorDescriptor *err);
    virtual Severity STEPread(G4std::istream &in, ErrorDescriptor *err);

//	OUTPUT
    virtual const char *asStr(SCLstring & s);
    virtual const char *STEPwrite (SCLstring &s, const char * =0);
    virtual void 	STEPwrite (G4std::ostream& out =G4cout);

//	CONSTRUCTORS
    EnumNode(SCLP23(Enum) * e);
    EnumNode();
    ~EnumNode();

    virtual SingleLinkNode *	NewNode ();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};

class RealNode  : public STEPnode {
  public:
    SCLP23(Real) value; // double

  public:
//	INPUT
    virtual Severity StrToVal(const char *s, ErrorDescriptor *err);
    virtual Severity StrToVal(G4std::istream &in, ErrorDescriptor *err);

    virtual Severity STEPread(const char *s, ErrorDescriptor *err);
    virtual Severity STEPread(G4std::istream &in, ErrorDescriptor *err);

//	OUTPUT
    virtual const char *asStr(SCLstring & s);
    virtual const char *STEPwrite (SCLstring &s, const char * =0);
    virtual void 	STEPwrite (G4std::ostream& out =G4cout);

//	CONSTRUCTORS
    RealNode();
    ~RealNode();

    virtual SingleLinkNode *	NewNode ();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};

class IntNode  : public STEPnode {
  public:
    SCLP23(Integer) value; // long int

  public:
//	INPUT
    virtual Severity StrToVal(const char *s, ErrorDescriptor *err);
    virtual Severity StrToVal(G4std::istream &in, ErrorDescriptor *err);

    virtual Severity STEPread(const char *s, ErrorDescriptor *err);
    virtual Severity STEPread(G4std::istream &in, ErrorDescriptor *err);

//	OUTPUT
    virtual const char *asStr(SCLstring & s);
    virtual const char *STEPwrite (SCLstring &s, const char * =0);
    virtual void 	STEPwrite (G4std::ostream& out =G4cout);

//	CONSTRUCTORS
    IntNode();
    ~IntNode();

    virtual SingleLinkNode *	NewNode ();

#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
};

/******************************************************************
 **	The following classes are currently stubs
 **							
**/
/*
class Array  :  public STEPaggregate  {	
  public:								      
    int lowerBound;							      
    int upperBound;							      
};
									      
class Bag  :  public STEPaggregate  {								      
  public:								      
    int min_ele;							      
    int max_ele;							      
};
									      
class List :  public STEPaggregate  {								      
    int min_ele;							      
    int max_ele;							      
    List *prev;								      
};
									      
class Set :								      
public STEPaggregate  {								      
    int cnt;								      
};
*/
#endif 
