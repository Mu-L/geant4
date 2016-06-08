// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4KaonZeroSInelasticProcess.hh,v 1.2 1999/12/15 14:53:34 gunter Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
 // G4 Process: KaonZeroS InelasticProcess Process
 // J.L. Chuma, TRIUMF, 11-Feb-1997
 // Last modified: 03-Apr-1997

#ifndef G4KaonZeroSInelasticProcess_h
#define G4KaonZeroSInelasticProcess_h 1

//#include "G4HadronicInelasticProcess.hh"
#include "G4HadronInelasticProcess.hh"
 
// class G4KaonZeroSInelasticProcess : public G4HadronicInelasticProcess
 class G4KaonZeroSInelasticProcess : public G4HadronInelasticProcess
 {
 public:
    
    G4KaonZeroSInelasticProcess(
     const G4String& processName = "KaonZeroSInelastic" ) :
      //      G4HadronicInelasticProcess( processName, G4KaonZeroShort::KaonZeroShort() )
      G4HadronInelasticProcess( processName, G4KaonZeroShort::KaonZeroShort() )
    { }
    
    ~G4KaonZeroSInelasticProcess()
    { }
 };
 
#endif
