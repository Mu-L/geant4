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
//
// This example is provided by the Geant4-DNA collaboration
// Any report or published results obtained using the Geant4-DNA software
// shall cite the following Geant4-DNA collaboration publications:
// Med. Phys. 45 (2018) e722-e739
// Phys. Med. 31 (2015) 861-874
// Med. Phys. 37 (2010) 4692-4708
// Int. J. Model. Simul. Sci. Comput. 1 (2010) 157–178
//
// The Geant4-DNA web site is available at http://geant4-dna.org
//
/// \file medical/dna/svalue/src/PhysicsList.cc
/// \brief Implementation of the PhysicsList class

#include "PhysicsList.hh"

#include "PhysicsListMessenger.hh"

#include "G4EmDNAPhysics.hh"
#include "G4EmDNAPhysics_option1.hh"
#include "G4EmDNAPhysics_option2.hh"
#include "G4EmDNAPhysics_option3.hh"
#include "G4EmDNAPhysics_option4.hh"
#include "G4EmDNAPhysics_option5.hh"
#include "G4EmDNAPhysics_option6.hh"
#include "G4EmDNAPhysics_option7.hh"
#include "G4EmDNAPhysics_option8.hh"
#include "G4EmStandardPhysics_option3.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4UserSpecialCuts.hh"

// Particles

#include "G4BaryonConstructor.hh"
#include "G4BosonConstructor.hh"
#include "G4DNAGenericIonsManager.hh"
#include "G4IonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"

// Decay

#include "G4Electron.hh"
#include "G4NuclideTable.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysicsList::PhysicsList() : G4VModularPhysicsList(), fEmPhysicsList(0), fMessenger(0)
{
  fMessenger = new PhysicsListMessenger(this);

  SetVerboseLevel(1);

  // Physics

  fEmPhysicsList = new G4EmDNAPhysics();
  defaultCutValue = 1. * CLHEP::nm;

  G4double lowLimit = 10. * CLHEP::eV;
  G4double highLimit = 100. * CLHEP::GeV;

  G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(lowLimit, highLimit);

  // Change time and other limits in G4NuclideTable
  G4NuclideTable::GetInstance()->SetThresholdOfHalfLife(0.1 * picosecond);
  G4NuclideTable::GetInstance()->SetLevelTolerance(1.0 * eV);

  fRadDecay = new G4RadioactiveDecayPhysics();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysicsList::~PhysicsList()
{
  delete fMessenger;
  delete fEmPhysicsList;
  delete fRadDecay;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysicsList::ConstructParticle()
{
  G4BosonConstructor pBosonConstructor;
  pBosonConstructor.ConstructParticle();

  G4LeptonConstructor pLeptonConstructor;
  pLeptonConstructor.ConstructParticle();

  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  G4IonConstructor pIonConstructor;
  pIonConstructor.ConstructParticle();

  G4DNAGenericIonsManager* genericIonsManager;
  genericIonsManager = G4DNAGenericIonsManager::Instance();
  genericIonsManager->GetIon("alpha++");
  genericIonsManager->GetIon("alpha+");
  genericIonsManager->GetIon("helium");
  genericIonsManager->GetIon("hydrogen");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysicsList::ConstructProcess()
{
  // Transportation

  AddTransportation();

  // Electromagnetic physics list

  fEmPhysicsList->ConstructProcess();

  // Tracking cut

  AddTrackingCut();

  // Radioactive decay

  fRadDecay->ConstructProcess();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysicsList::AddPhysicsList(const G4String& name)
{
  if (verboseLevel > 0) {
    G4cout << "PhysicsList::AddPhysicsList: <" << name << ">" << G4endl;
  }

  if (name == fEmName) return;

  if (name == "dna") {
    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics();
  }
  else if (name == "dna_opt1") {
    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option1();
  }
  else if (name == "dna_opt2") {
    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option2();
  }
  else if (name == "dna_opt3") {
    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option3();
  }
  else if (name == "dna_opt4") {
    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option4();
  }
  else if (name == "dna_opt5") {
    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option5();
  }
  else if (name == "dna_opt6") {
    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option6();
  }
  else if (name == "dna_opt7") {
    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option7();
  }
  else if (name == "dna_opt8") {
    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option8();
  }
  else if (name == "std_opt3") {
    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmStandardPhysics_option3();
  }
  else if (name == "std_opt4") {
    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmStandardPhysics_option4();
  }
  else {
    G4cout << "PhysicsList::AddPhysicsList: <" << name << ">"
           << " is not defined" << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysicsList::AddTrackingCut()
{
  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();
  ph->RegisterProcess(new G4UserSpecialCuts(), G4Electron::Electron());
}
