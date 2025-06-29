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
// shall cite the following Geant4-DNA collaboration publication:
// Med. Phys. 37 (2010) 4692-4708
// J. Comput. Phys. 274 (2014) 841-882
// Phys. Med. Biol. 63(10) (2018) 105014-12pp
// The Geant4-DNA web site is available at http://geant4-dna.org
//
//
#include "PrimaryKiller.hh"

#include <G4Event.hh>
#include <G4EventManager.hh>
#include <G4RunManager.hh>
#include <G4UIcmdWithADoubleAndUnit.hh>
#include <G4UnitsTable.hh>
#include <globals.hh>

/** \file PrimaryKiller.cc
    \class PrimaryKiller

    Kill the primary particle:
    - either after a given energy loss
    - or after the primary particle has reached a given energy
 */

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryKiller::PrimaryKiller(G4String name, G4int depth)
  : G4VPrimitiveScorer(name, depth), G4UImessenger()
{
  fpELossUI = new G4UIcmdWithADoubleAndUnit("/primaryKiller/eLossMin", this);
  fpAbortEventIfELossUpperThan = new G4UIcmdWithADoubleAndUnit("/primaryKiller/eLossMax", this);
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PrimaryKiller::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == fpELossUI) {
    fELossRange_Min = fpELossUI->GetNewDoubleValue(newValue);
  }
  else if (command == fpAbortEventIfELossUpperThan) {
    fELossRange_Max = fpAbortEventIfELossUpperThan->GetNewDoubleValue(newValue);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4bool PrimaryKiller::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
  const G4Track* track = aStep->GetTrack();

  if (track->GetTrackID() != 1) return FALSE;

  //-------------------

  auto kineticE = aStep->GetPostStepPoint()->GetKineticEnergy();

  auto eLoss = aStep->GetPreStepPoint()->GetKineticEnergy() - kineticE;

  if (eLoss == 0.) return FALSE;

  //-------------------
  fELoss += eLoss;

  if (fELoss > fELossRange_Max) {
    G4RunManager::GetRunManager()->AbortEvent();
    int eventID = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
    G4cout << "Aborts event " << eventID
           << ", energy loss "
              "is too large. \n"
           << " * Energy loss by primary is: " << G4BestUnit(fELoss, "Energy")
           << ". Event is aborted if the Eloss is > " << G4BestUnit(fELossRange_Max, "Energy")
           << G4endl;
  }

  if (fELoss >= fELossRange_Min) {  //|| kineticE <= fKineticE_Min){
    ((G4Track*)track)->SetTrackStatus(fStopAndKill);
    G4cout << "Kill track at : " << G4BestUnit(kineticE, "Energy")
           << ", Energy loss by primary is: " << G4BestUnit(fELoss, "Energy")
           << ", primary is terminated as Eloss is >: " << G4BestUnit(fELossRange_Min, "Energy")
           << G4endl;  //", EThreshold is: "
  }

  return true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PrimaryKiller::Initialize(G4HCofThisEvent* /*HCE*/)
{
  Clear();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
