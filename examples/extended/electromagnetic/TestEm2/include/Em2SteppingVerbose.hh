// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: Em2SteppingVerbose.hh,v 1.3 2000/02/29 12:16:59 maire Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
//  
//---------------------------------------------------------------
//
// Em2SteppingVerbose.hh
//
// Description:
//   This class manages the vervose outputs in G4SteppingManager. 
//   
//---------------------------------------------------------------

class Em2SteppingVerbose;

#ifndef Em2SteppingVerbose_h
#define Em2SteppingVerbose_h 1

#include "G4SteppingVerbose.hh"

class Em2SteppingVerbose : public G4SteppingVerbose {
public:   
// Constructor/Destructor
  Em2SteppingVerbose();
 ~Em2SteppingVerbose();
//
  void StepInfo();
  void TrackingStarted();
//


};

#endif
