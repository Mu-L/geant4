
/*
* NIST STEP Core Class Library
* clstepcore/STEPattributeList.cc
* April 1997
* K. C. Morris
* David Sauder

* Development of this software was funded by the United States Government,
* and is not subject to copyright.
*/

/* $Id: STEPattributeList.cc,v 1.3 2000/01/21 13:42:56 gcosmo Exp $ */

#include <STEPattributeList.h>
#include <STEPattribute.h>

//#include <stdlib.h>

AttrListNode::AttrListNode(STEPattribute *a)
{
    attr = a;
}

AttrListNode::~AttrListNode()
{
}


STEPattributeList::STEPattributeList()
{
}

STEPattributeList::~STEPattributeList()
{
}

STEPattribute& STEPattributeList::operator [] (int n)
{
    int x = 0;
    AttrListNode* a = (AttrListNode *)head;
    int cnt =  EntryCount();
    if (n < cnt)
	while (a && (x < n))
	{
	    a = (AttrListNode *)(a->next);
	    x++;
	}
    // Fixed return value - GC
    if (!a)
      G4cerr << "\nERROR in STEP Core library:  " << __FILE__ <<  ":"
             << __LINE__ << "\n" << _POC_ << "\n\n";
    return *(a->attr);
}

int STEPattributeList::list_length()
{
    return EntryCount();
}

void STEPattributeList::push(STEPattribute *a)
{
#ifdef __OSTORE__
    AttrListNode *saln = new (os_database::of(this), 
			      AttrListNode::get_os_typespec()) AttrListNode(a);
#else
    AttrListNode *saln = new AttrListNode(a);
#endif
    AppendNode (saln);
}










/*
STEPattributeListNode NilSTEPattributeListNode;

class init_NilSTEPattributeListNode
{
public:
  inline init_NilSTEPattributeListNode() 
  {
    NilSTEPattributeListNode.tl = &NilSTEPattributeListNode;
    NilSTEPattributeListNode.ref = -1;
  }
};

static init_NilSTEPattributeListNode NilSTEPattributeListNode_initializer;

*/
