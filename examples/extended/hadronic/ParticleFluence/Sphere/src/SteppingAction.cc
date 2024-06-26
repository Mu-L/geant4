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
/// \file SteppingAction.cc
/// \brief Implementation of the SteppingAction class
//
//

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "SteppingAction.hh"

#include "Run.hh"

#include "G4IonTable.hh"
#include "G4LossTableManager.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTypes.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4TouchableHistory.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VSolid.hh"
#include "G4VTouchable.hh"

const std::array<G4String, SteppingAction::fkNumberKinematicRegions>
  SteppingAction::fkArrayKinematicRegionNames = {"", "below 20 MeV", "above 20 MeV"};

const std::array<G4String, SteppingAction::fkNumberScoringPositions>
  SteppingAction::fkArrayScoringPositionNames = {"forward", "backward"};

const std::array<G4String, SteppingAction::fkNumberParticleTypes>
  SteppingAction::fkArrayParticleTypeNames = {"all",      "electron",   "gamma",      "muon",
                                              "neutrino", "pion",       "neutron",    "proton",
                                              "ion",      "otherMeson", "otherBaryon"};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4int SteppingAction::GetIndex(const G4int iKinematicRegion, const G4int iScoringPosition,
                               const G4int iParticleType)
{
  G4int index = -1;
  if (iKinematicRegion >= 0 && iKinematicRegion < fkNumberKinematicRegions && iScoringPosition >= 0
      && iScoringPosition < fkNumberScoringPositions && iParticleType >= 0
      && iParticleType < fkNumberParticleTypes)
  {
    index = iKinematicRegion * fkNumberScoringPositions * fkNumberParticleTypes
            + iScoringPosition * fkNumberParticleTypes + iParticleType;
  }
  if (index < 0 || index >= fkNumberCombinations) {
    G4cerr << "SteppingAction::GetIndex : WRONG index=" << index << "  set it to 0 !" << G4endl;
    index = 0;
  }
  return index;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::SteppingAction() : G4UserSteppingAction()
{
  Initialize();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SteppingAction::Initialize()
{
  // Initialization needed at the beginning of each Run
  fPrimaryParticleId = 0;
  fPrimaryParticleEnergy = 0.0;
  fPrimaryParticleDirection = G4ThreeVector(0.0, 0.0, 1.0);
  fTargetMaterialName = "";
  fIsFirstStepOfTheEvent = true;
  fIsFirstStepInTarget = true;
  fIsFirstStepInScoringShell = true;
  fCubicVolumeScoringShell = 1.0;
  for (G4int i = 0; i < fkNumberCombinations; ++i) {
    fArraySumStepLengths[i] = 0.0;
  }
  /*
  for ( G4int i = 0; i < fkNumberCombinations; ++i ) fArraySumStepLengths[i] = 999.9;
  G4cout << " fkNumberCombinations=" << fkNumberCombinations << G4endl;
  for ( G4int i = 0; i < fkNumberKinematicRegions; ++i ) {
    for ( G4int j = 0; j < fkNumberScoringPositions; ++j ) {
      for ( G4int k = 0; k < fkNumberParticleTypes; ++k ) {
        G4int index = GetIndex( i, j, k );
        G4cout << "(i, j, k)=(" << i << ", " << j << ", " << k << ")  ->" << index;
        if ( fArraySumStepLengths[ index ] < 1.0 ) G4cout << " <=== REPEATED!";
        else                                       fArraySumStepLengths[ index ] = 0.0;
        G4cout << G4endl;
      }
    }
  }
  for ( G4int i = 0; i < fkNumberCombinations; ++i ) {
    if ( fArraySumStepLengths[i] > 999.0 ) G4cout << " i=" << i << " NOT COVERED !" << G4endl;
  }
  */
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SteppingAction::UserSteppingAction(const G4Step* theStep)
{
  // Get information on the primary particle
  if (fIsFirstStepOfTheEvent) {
    if (theStep->GetTrack()->GetParentID() == 0) {
      fPrimaryParticleId = theStep->GetTrack()->GetDefinition()->GetPDGEncoding();
      fPrimaryParticleEnergy = theStep->GetPreStepPoint()->GetKineticEnergy();
      fPrimaryParticleDirection = theStep->GetPreStepPoint()->GetMomentumDirection();
      if (fRunPtr) {
        fRunPtr->SetPrimaryParticleId(fPrimaryParticleId);
        fRunPtr->SetPrimaryParticleEnergy(fPrimaryParticleEnergy);
        fRunPtr->SetPrimaryParticleDirection(fPrimaryParticleDirection);
      }
      fIsFirstStepOfTheEvent = false;
    }
  }
  // Get information on the target material
  if (fIsFirstStepInTarget
      && theStep->GetPreStepPoint()->GetPhysicalVolume()->GetName() == "physiSphere")
  {
    fTargetMaterialName = theStep->GetPreStepPoint()->GetMaterial()->GetName();
    if (fRunPtr) fRunPtr->SetTargetMaterialName(fTargetMaterialName);
    fIsFirstStepInTarget = false;
  }
  // Get information on step lengths in the scoring shell
  if (theStep->GetPreStepPoint()->GetPhysicalVolume()->GetName() == "physiScoringShell") {
    if (fIsFirstStepInScoringShell) {
      fCubicVolumeScoringShell =
        theStep->GetTrack()->GetVolume()->GetLogicalVolume()->GetSolid()->GetCubicVolume();
      if (fRunPtr) fRunPtr->SetCubicVolumeScoringShell(fCubicVolumeScoringShell);
      fIsFirstStepInScoringShell = false;
    }
    G4double stepLength = theStep->GetTrack()->GetStepLength() * theStep->GetTrack()->GetWeight();
    G4int absPdg = theStep->GetTrack()->GetDefinition() == nullptr
                     ? 0
                     : std::abs(theStep->GetTrack()->GetDefinition()->GetPDGEncoding());
    /*
    G4cout << theStep->GetTrack()->GetDefinition()->GetParticleName() << "  absPdg=" << absPdg
           << "  Ekin[MeV]=" << theStep->GetPreStepPoint()->GetKineticEnergy()
           << "  r[mm]=" << theStep->GetTrack()->GetPosition().mag()
           << "  z[mm]=" << theStep->GetTrack()->GetPosition().z()
           << "  " << theStep->GetTrack()->GetVolume()->GetName()
           << "  " << theStep->GetTrack()->GetMaterial()->GetName()
           << "  L[mm]=" << stepLength << "  "
           << ( fPrimaryParticleDirection.dot( theStep->GetTrack()->GetPosition().unit() ) > 0.0
                ? "forward" : "backward" ) << G4endl;
    */
    // Three kinematical regions:  [0] : any value ;  [1] : below 20 MeV ;  [2] : above 20 MeV
    G4int iKinematicRegion = theStep->GetPreStepPoint()->GetKineticEnergy() < 20.0 ? 1 : 2;
    // Two scoring positions:  [0] : forward hemisphere ;  [1] : backward hemisphere
    // (with respect to the primary particle initial direction)
    G4int iScoringPosition =
      fPrimaryParticleDirection.dot(theStep->GetTrack()->GetPosition().unit()) > 0.0 ? 0 : 1;
    G4int iParticleType = -1;
    if (absPdg == 11)
      iParticleType = 1;  // electron (and positron)
    else if (absPdg == 22)
      iParticleType = 2;  // gamma
    else if (absPdg == 13)
      iParticleType = 3;  // muons (mu- and mu+)
    else if (absPdg == 12 || absPdg == 14 || absPdg == 16)
      iParticleType = 4;  // neutrinos (and
                          // anti-neutrinos), all flavors
    else if (absPdg == 111 || absPdg == 211)
      iParticleType = 5;  // (charged) pions
    else if (absPdg == 2112)
      iParticleType = 6;  // neutron (and anti-neutron)
    else if (absPdg == 2212)
      iParticleType = 7;  // proton  (and anti-proton)
    else if (G4IonTable::IsIon(theStep->GetTrack()->GetDefinition()) ||  // ions (and anti-ions)
             G4IonTable::IsAntiIon(theStep->GetTrack()->GetDefinition()))
      iParticleType = 8;
    else if (absPdg < 1000)
      iParticleType = 9;  // other mesons (e.g. kaons) (Note: this works
                          // in most cases, but not always!)
    else if (absPdg > 1000)
      iParticleType = 10;  // other baryons (e.g. hyperons, anti-hyperons,
                           // etc.)
    // Consider the specific case : kinematic region, scoring position, and particle type
    G4int index = GetIndex(iKinematicRegion, iScoringPosition, iParticleType);
    fArraySumStepLengths[index] += stepLength;
    // Consider the "all" particle case, with the same kinematic region and scoring position
    index = GetIndex(iKinematicRegion, iScoringPosition, 0);
    fArraySumStepLengths[index] += stepLength;
    // Consider the "any" kinematic region case, with the same scoring position and particle type
    index = GetIndex(0, iScoringPosition, iParticleType);
    fArraySumStepLengths[index] += stepLength;
    // Consider the "any" kinematic region and "all" particle, with the same scoring position
    index = GetIndex(0, iScoringPosition, 0);
    fArraySumStepLengths[index] += stepLength;
    if (fRunPtr) fRunPtr->SetSteppingArray(fArraySumStepLengths);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
