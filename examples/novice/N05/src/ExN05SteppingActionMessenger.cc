// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: ExN05SteppingActionMessenger.cc,v 1.2 1999/12/15 14:49:31 gunter Exp $
// GEANT4 tag $Name: geant4-02-00 $
//

#include "ExN05SteppingActionMessenger.hh"
#include "ExN05SteppingAction.hh"

#include "globals.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithABool.hh"

ExN05SteppingActionMessenger::ExN05SteppingActionMessenger(ExN05SteppingAction* SA)
:SteppingAction(SA)
{
  stepDirectory = new G4UIdirectory("/step/");
  stepDirectory->SetGuidance("Step draw control command.");

  drawStepCmd = new G4UIcmdWithABool("/step/draw",this);
  drawStepCmd->SetGuidance("Draw each step on the fly. (default = true)");
  drawStepCmd->SetParameterName("draw", true);
  drawStepCmd->SetDefaultValue(true);
}

void ExN05SteppingActionMessenger::SetNewValue(G4UIcommand* command, G4String newValues)
{
  if( command->GetCommandName() == "draw" )
  {
    G4int vl;
    const char* t = newValues;
    G4std::istrstream is((char*)t);
    is >> vl;
    SteppingAction->SetDrawFlag(vl!=0);
  }
}

