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
//
/// \file GB04DetectorConstruction.cc
/// \brief Implementation of the GB04DetectorConstruction class

#include "GB04DetectorConstruction.hh"

#include "GB04BOptrBremSplitting.hh"

#include "G4Box.hh"
#include "G4Colour.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4UniformMagField.hh"
#include "G4VisAttributes.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

GB04DetectorConstruction::GB04DetectorConstruction(G4bool bf) : fBiasingFlag(bf) {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

GB04DetectorConstruction::~GB04DetectorConstruction() = default;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* GB04DetectorConstruction::Construct()
{
  G4Material* worldMaterial = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
  G4Material* defaultMaterial = G4NistManager::Instance()->FindOrBuildMaterial("G4_Al");

  G4VSolid* solidWorld = new G4Box("World", 10 * m, 10 * m, 10 * m);

  auto logicWorld = new G4LogicalVolume(solidWorld,  // its solid
                                        worldMaterial,  // its material
                                        "World");  // its name

  auto physiWorld = new G4PVPlacement(nullptr,  // no rotation
                                      G4ThreeVector(),  // at (0,0,0)
                                      logicWorld,  // its logical volume
                                      "World",  // its name
                                      nullptr,  // its mother  volume
                                      false,  // no boolean operation
                                      0);  // copy number

  // -----------------------------------
  // -- volume where biasing is applied:
  // -----------------------------------
  G4double halfZ = 5 * mm;
  G4VSolid* solidTest = new G4Box("test.solid", 1 * m, 1 * m, halfZ);

  auto logicTest = new G4LogicalVolume(solidTest,  // its solid
                                       defaultMaterial,  // its material
                                       "test.logical");  // its name

  new G4PVPlacement(nullptr,  // no rotation
                    G4ThreeVector(0, 0, halfZ),  // volume entrance at (0,0,0)
                    logicTest,  // its logical volume
                    "test.phys",  // its name
                    logicWorld,  // its mother  volume
                    false,  // no boolean operation
                    0);  // copy number

  return physiWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void GB04DetectorConstruction::ConstructSDandField()
{
  if (fBiasingFlag) {
    // -- Fetch volume for biasing:
    G4LogicalVolume* logicTest = G4LogicalVolumeStore::GetInstance()->GetVolume("test.logical");

    // ----------------------------------------------
    // -- operator creation and attachment to volume:
    // ----------------------------------------------
    auto bremSplittingOperator = new GB04BOptrBremSplitting();
    bremSplittingOperator->AttachTo(logicTest);
    G4cout << " Attaching biasing operator " << bremSplittingOperator->GetName()
           << " to logical volume " << logicTest->GetName() << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
