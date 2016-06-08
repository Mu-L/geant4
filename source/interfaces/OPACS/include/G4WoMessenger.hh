// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4WoMessenger.hh,v 1.3 1999/12/15 14:50:44 gunter Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
#ifndef G4WoMessenger_h
#define G4WoMessenger_h 1

#ifdef G4UI_BUILD_WO_SESSION

#include "globals.hh"
#include "G4UImessenger.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;

#include <OShell.h>

class G4WoMessenger: public G4UImessenger
{
  public:
    G4WoMessenger(OShell);
    ~G4WoMessenger();
  public:
    void     SetNewValue    (G4UIcommand * command,G4String newValues);
    G4String GetCurrentValue(G4UIcommand * command);
  private:
    OShell shell;
  private: //commands
    G4UIdirectory*           WoDirectory;
    G4UIcmdWithoutParameter* sendExitCmd;
    G4UIcmdWithAString*      oshCmd;
};

#endif

#endif





