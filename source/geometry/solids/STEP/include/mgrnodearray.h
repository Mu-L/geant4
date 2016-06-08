
#ifndef mgrnodearray_h
#define mgrnodearray_h

/*
* NIST STEP Editor Class Library
* cleditor/mgrnodearray.h
* April 1997
* David Sauder
* K. C. Morris

* Development of this software was funded by the United States Government,
* and is not subject to copyright.
*/

/* $Id: mgrnodearray.h,v 1.4 2000/01/21 13:42:40 gcosmo Exp $ */ 

/*
#ifdef __OSTORE__
#include <ostore/ostore.hh>    // Required to access ObjectStore Class Library
#endif
*/

#ifdef __O3DB__
#include <OpenOODB.h>
#endif

#include <string.h>

#include <gennode.h>
#include <gennodelist.h>
//#include <gennode.inline.h>

#include <mgrnode.h>
#include <mgrnodelist.h>

#include <gennodearray.h>

#define ARRAY_DEFAULT_SIZE (1024)

//////////////////////////////////////////////////////////////////////////////
// class MgrNodeArray
// This class implements the array of MgrNodes representing the
// master list for the seInstMgr which will have a one to one
// mapping by index to the display list of corresponding STEPentities.
// If you delete this object it deletes all of the entries it points to.
//////////////////////////////////////////////////////////////////////////////

class MgrNodeArray : public GenNodeArray
{
public:
//#ifdef __OSTORE__
//    MgrNodeArray(os_database *db, int defaultSize = ARRAY_DEFAULT_SIZE);
//#else
    MgrNodeArray(int defaultSize = ARRAY_DEFAULT_SIZE);
//#endif
    ~MgrNodeArray();

// REDEFINED functions
	// need to redefine Append() & Insert(GenericNode *) so they call 
	//	MgrNodeArray::Insert(GenericNode *, int index);
    virtual int Insert(GenericNode* gn, int index);
    virtual void Append(GenericNode* gn)	{ Insert(gn, _count); }
    virtual int Insert(GenericNode* gn)		{ return Insert(gn, _count); }
    virtual void Remove(int index);
    virtual void ClearEntries();
    virtual void DeleteEntries();

// ADDED functions
    virtual int MgrNodeIndex(int fileId);
    void AssignIndexAddress(int index);
/*
#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
*/
};

//////////////////////////////////////////////////////////////////////////////
// class MgrNodeArraySorted
// This class implements the array of MgrNodes representing the
// sorted master list for the seInstMgr.  This list is sorted by
// STEPentity::fileId.
// If you delete this object it won't delete the entries it points to.
//////////////////////////////////////////////////////////////////////////////

class MgrNodeArraySorted : public GenNodeArray {
public:
//#ifdef __OSTORE__
//    MgrNodeArraySorted(os_database *db, int defaultSize = ARRAY_DEFAULT_SIZE);
//#else
    MgrNodeArraySorted(int defaultSize = ARRAY_DEFAULT_SIZE);
//#endif
    ~MgrNodeArraySorted() { }

// REDEFINED functions
    virtual int Index(GenericNode* gn);
    virtual int Index(GenericNode** gn);

    virtual int Insert(GenericNode* gn);
    virtual int Insert(GenericNode* gn, int index) 
	{ G4cerr << 
	  "Call MgrNodeArraySorted::Insert() without index argument instead.\n"
		<< "index argument: " << index << " being ignored.\n";
	  return Insert(gn); }
    virtual void Append(GenericNode* gn)		{ Insert(gn); }
    virtual void ClearEntries();
    virtual void DeleteEntries();

// ADDED functions
    virtual int MgrNodeIndex(int fileId);
    int FindInsertPosition (const int fileId);
/*
#ifdef __OSTORE__
    static os_typespec* get_os_typespec();
#endif
*/
};


//////////////////////////////////////////////////////////////////////////////
// class MgrNodeArray inline public functions
//////////////////////////////////////////////////////////////////////////////

#endif
