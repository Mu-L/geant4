// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4VisCommandsViewerSet.hh,v 1.3 2000/05/19 09:17:47 johna Exp $
// GEANT4 tag $Name: geant4-02-00 $

// /vis/viewer/set commands - John Allison  16th May 2000

#ifndef G4VISCOMMANDSVIEWERSET_HH
#define G4VISCOMMANDSVIEWERSET_HH

#include "G4VVisCommand.hh"

#include "g4std/vector"

class G4UIcommand;
class G4UIcmdWithAString;
class G4UIcmdWithABool;

class G4VisCommandsViewerSet: public G4VVisCommand {
public:
  // Uses compiler defaults for copy constructor and assignment.
  G4VisCommandsViewerSet ();
  ~G4VisCommandsViewerSet ();
  G4String GetCurrentValue (G4UIcommand* command);
  void SetNewValue (G4UIcommand* command, G4String newValue);
private:
  G4String ConvertToString(G4bool blValue);
  G4bool GetNewBoolValue(const G4String& paramString);
  G4UIcmdWithAString* fpCommandAll;
  G4UIcmdWithAString* fpCommandStyle;
  G4UIcmdWithABool* fpCommandEdge;
  G4UIcmdWithABool* fpCommandHiddenEdge;
  G4UIcommand* fpCommandCulling;
};

#endif