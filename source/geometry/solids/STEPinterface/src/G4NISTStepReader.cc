// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4NISTStepReader.cc,v 1.3 2000/01/21 13:46:03 gcosmo Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
// 
// ----------------------------------------------------------------------
// Class G4NISTStepReader
//
// Authors: J.Sulkimo, P.Urban.
// Revisions by: L.Broglia, G.Cosmo.
//
// History:
//   18-Nov-1999: First step of re-engineering - G.Cosmo
// ----------------------------------------------------------------------

#include "G4NISTStepReader.hh"

#include "instmgr.h"
#include "Registry.h"
#include "STEPaggregate.h"
#include "STEPfile.h"   /*  STEPfile class and others used by SCL */

#ifdef __O3DB__
  #include <OpenOODB.h>
#endif

void G4NISTStepReader::ReadSTEPFile(G4String STEPfilename)
{
  // Reads the Step file into the nist toolkit class library 
    
  // if the comments below look canned, it's because they're taken
  // straight from the treg source... ;-)

  // This has to be Done before anything else.  This initializes
  // all of the registry information for the schema you are using.
  // The SchemaInit() function is generated by fedex_plus... see
  // extern statement above.
  Registry *registry = new Registry(SchemaInit);
    
  // The nifty thing about the Registry is that it basically keeps a List
  // of everything in your schema.  What this means is that we can go
  // through the Registry and instantiate, say, one of everything, without
  // knowing at coding-time what entities there are to instantiate.  So,
  // this test could be linked with other class libraries produced from
  // other schema, rather than the example, and run happily.

  //    InstMgr InstanceList;
  STEPfile *sfile = new STEPfile(*registry, InstanceList);

  // The STEPfile is actually an object that manages the relationship
  // between what's instantiated in the instance manager, and how that
  // information gets passed to the outside, e.g., a file on disk.
  
  // First we're going to read in the data from the input STEPfile,
  // and Print it to stdout
  
  // Reading the STEPfile instantiates all the objects in it.
  // The instances get pointers into the InstMgr, and each type
  // and entity gets a pointer into the registry.
  G4cout << "\n### Reading exchange file from " << STEPfilename << G4endl;
  sfile->ReadExchangeFile(STEPfilename);    

  // Just checking... ;-)
  G4cout << "\n### The InstMgr says there are ";
  G4cout << InstanceList.InstanceCount() << " instantiated objects" << G4endl;
    
  // G4cout << "\n### Here is the exchange file:" << G4endl << G4endl;
  // sfile->WriteExchangeFile(G4cout);

  delete sfile;
  delete registry; 
}


void G4NISTStepReader::SaveSTEPFile() {}

void G4NISTStepReader::UpdateSTEPFile() {}
