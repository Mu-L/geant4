// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: Em4SteppingAction.hh,v 1.2 1999/12/15 14:49:06 gunter Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
// 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#ifndef Em4SteppingAction_h
#define Em4SteppingAction_h 1

#include "G4UserSteppingAction.hh"

class Em4EventAction;
class G4SteppingVerbose2;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class Em4SteppingAction : public G4UserSteppingAction
{
  public:
    Em4SteppingAction(Em4EventAction*);
   ~Em4SteppingAction();

    void UserSteppingAction(const G4Step*);
    
  private:
    Em4EventAction* eventAction;
    G4SteppingVerbose2* myVerbose;    
};

#endif
