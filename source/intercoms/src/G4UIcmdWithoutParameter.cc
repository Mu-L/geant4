// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4UIcmdWithoutParameter.cc,v 1.2 1999/12/15 14:50:41 gunter Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
//

#include "G4UIcmdWithoutParameter.hh"

G4UIcmdWithoutParameter::G4UIcmdWithoutParameter
(const char * theCommandPath,G4UImessenger * theMessenger)
:G4UIcommand(theCommandPath,theMessenger)
{;}

