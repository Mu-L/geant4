//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************

// File name:     G4NistMessenger
//
// Author:        Vladimir Ivanchenko
//
// Creation date: 23.12.2004
//
// Modifications:
//
//
// -------------------------------------------------------------------

#include "G4NistMessenger.hh"

#include "G4ApplicationState.hh"
#include "G4DensityEffectData.hh"
#include "G4IonisParamMat.hh"
#include "G4NistManager.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIdirectory.hh"
#include "G4UIcommand.hh"
#include "G4UIparameter.hh"

#include <sstream>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4NistMessenger::G4NistMessenger(G4NistManager* man) : manager(man)
{
  matDir = new G4UIdirectory("/material/");
  matDir->SetGuidance("Commands for materials");

  verCmd = new G4UIcmdWithAnInteger("/material/verbose", this);
  verCmd->SetGuidance("Set verbose level.");
  verCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
  verCmd->SetToBeBroadcasted(false);

  nistDir = new G4UIdirectory("/material/nist/");
  nistDir->SetGuidance("Commands for the nist dataBase");

  prtElmCmd = new G4UIcmdWithAString("/material/nist/printElement", this);
  prtElmCmd->SetGuidance("print element(s) in dataBase.");
  prtElmCmd->SetGuidance("symbol = element.");
  prtElmCmd->SetGuidance("all    = all elements.");
  prtElmCmd->SetParameterName("symbol", true);
  prtElmCmd->SetDefaultValue("all");
  prtElmCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
  prtElmCmd->SetToBeBroadcasted(false);

  przElmCmd = new G4UIcmdWithAnInteger("/material/nist/printElementZ", this);
  przElmCmd->SetGuidance("print element Z in dataBase.");
  przElmCmd->SetGuidance("0 = all elements.");
  przElmCmd->SetParameterName("Z", true);
  przElmCmd->SetDefaultValue(0);
  przElmCmd->SetRange("0<=Z && Z<108");
  przElmCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
  przElmCmd->SetToBeBroadcasted(false);

  lisMatCmd = new G4UIcmdWithAString("/material/nist/listMaterials", this);
  lisMatCmd->SetGuidance("Materials in Geant4 dataBase.");
  lisMatCmd->SetGuidance("simple - simple NIST materials.");
  lisMatCmd->SetGuidance("compound - compound NIST materials.");
  lisMatCmd->SetGuidance("hep - HEP materials.");
  lisMatCmd->SetGuidance("bio - biomedical materials.");
  lisMatCmd->SetGuidance("all - list of all Geant4 materials.");
  lisMatCmd->SetParameterName("matlist", true);
  // lisMatCmd->SetCandidates("simple compound hep bio all");
  lisMatCmd->SetDefaultValue("all");
  lisMatCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
  lisMatCmd->SetToBeBroadcasted(false);

  g4Dir = new G4UIdirectory("/material/g4/");
  g4Dir->SetGuidance("Commands for G4MaterialTable");

  g4ElmCmd = new G4UIcmdWithAString("/material/g4/printElement", this);
  g4ElmCmd->SetGuidance("print Element from G4ElementTable.");
  g4ElmCmd->SetGuidance("all - all elements.");
  g4ElmCmd->SetParameterName("elm", true);
  g4ElmCmd->SetDefaultValue("all");

  g4MatCmd = new G4UIcmdWithAString("/material/g4/printMaterial", this);
  g4MatCmd->SetGuidance("print Material from G4MaterialTable.");
  g4MatCmd->SetGuidance("all - all materials");
  g4MatCmd->SetParameterName("pmat", true);
  g4MatCmd->SetDefaultValue("all");
  g4MatCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
  g4MatCmd->SetToBeBroadcasted(false);

  g4DensCmd = new G4UIcmdWithAString("/material/g4/printDensityEffParam", this);
  g4DensCmd->SetGuidance("print Material from G4DensityEffectData.");
  g4DensCmd->SetGuidance("all - all materials");
  g4DensCmd->SetParameterName("dmat", true);
  g4DensCmd->SetDefaultValue("all");
  g4DensCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
  g4DensCmd->SetToBeBroadcasted(false);

  densCmd = new G4UIcmdWithAString("/material/g4/enableDensityEffOnFly", this);
  densCmd->SetGuidance("enable accurate computation of density effect.");
  densCmd->SetGuidance("all - all materials.");
  densCmd->SetParameterName("dens", true);
  densCmd->SetDefaultValue("all");
  densCmd->AvailableForStates(G4State_PreInit);
  lisMatCmd->SetToBeBroadcasted(false);

  adensCmd = new G4UIcmdWithAString("/material/g4/disableDensityEffOnFly", this);
  adensCmd->SetGuidance("disable accurate computation of density effect.");
  adensCmd->SetGuidance("all - all materials.");
  adensCmd->SetParameterName("dens", true);
  adensCmd->SetDefaultValue("all");
  adensCmd->AvailableForStates(G4State_PreInit);
  adensCmd->SetToBeBroadcasted(false);

  fPosiCmd = new G4UIcommand("/material/g4/ortoPositroniumFraction", this);
  fPosiCmd->SetGuidance("defined orto-positronium fraction for positron annihilation AtRest.");
  fPosiCmd->SetGuidance("via material name, all - all materials.");
  fPosiCmd->AvailableForStates(G4State_PreInit);
  fPosiCmd->SetToBeBroadcasted(false);

  auto p1 = new G4UIparameter("matname",'s', false);
  fPosiCmd->SetParameter(p1);

  auto p2 = new G4UIparameter("fraction", 'd', false);
  p2->SetParameterRange("fraction>=0. && fraction <=1.");
  fPosiCmd->SetParameter(p2);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4NistMessenger::~G4NistMessenger()
{
  delete verCmd;
  delete prtElmCmd;
  delete przElmCmd;
  delete lisMatCmd;
  delete nistDir;

  delete g4ElmCmd;
  delete g4MatCmd;
  delete g4DensCmd;
  delete densCmd;
  delete adensCmd;
  delete fPosiCmd;
  
  delete g4Dir;
  delete matDir;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4NistMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  // G4cout << "G4NistMessenger::SetNewValue <" << newValue << ">" << G4endl;
  if (command == verCmd) {
    manager->SetVerbose(verCmd->GetNewIntValue(newValue));
  }
  else if (command == prtElmCmd) {
    manager->PrintElement(newValue);
  }
  else if (command == przElmCmd) {
    G4int Z = przElmCmd->GetNewIntValue(newValue);
    if (Z >= 0 && Z < 108) {
      manager->PrintElement(Z);
    }
  }
  else if (command == lisMatCmd) {
    manager->ListMaterials(newValue);
  }
  else if (command == g4ElmCmd) {
    manager->PrintG4Element(newValue);
  }
  else if (command == g4MatCmd) {
    manager->PrintG4Material(newValue);
  }
  else if (command == g4DensCmd) {
    G4IonisParamMat::GetDensityEffectData()->PrintData(newValue);
  }
  else if (command == densCmd) {
    manager->SetDensityEffectCalculatorFlag(newValue, true);
  }
  else if (command == adensCmd) {
    manager->SetDensityEffectCalculatorFlag(newValue, false);
  }
  else if (command == fPosiCmd) {
    G4String mnam{""};
    G4double f{0.0};
    std::istringstream ss(newValue);
    ss >> mnam >> f;
    // set fraction for all materials 
    if (mnam == "all" || mnam == "none") {
      if (mnam == "none" || f < 0.0) { f = 0.0; }
      auto mtable = G4Material::GetMaterialTable();
      for ( auto const & mat : *mtable ) {
	mat->GetIonisation()->SetOrtoPositroniumFraction(f);
      }
    } else {
      // set fraction for one material 
      auto mat = manager->FindOrBuildMaterial(mnam, true);
      if (nullptr != mat) { 
	mat->GetIonisation()->SetOrtoPositroniumFraction(f);
      }
    }
  }
}
