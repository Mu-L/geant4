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

#include "DetectorConstruction.hh"

#include "ChemistryWorld.hh"
#include "PeriodicBoundaryBuilder.hh"
#include "Scorer.hh"

#include "G4Box.hh"
#include "G4Electron_aq.hh"
#include "G4LogicalVolume.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SDManager.hh"
#include "G4UnitsTable.hh"
#include "G4VPrimitiveScorer.hh"
#include "G4VisAttributes.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::DetectorConstruction() : G4VUserDetectorConstruction()
{
  fpChemistryWorld = std::make_unique<ChemistryWorld>();
  fpChemistryWorld->ConstructChemistryBoundary();
  DefineCommands();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  auto water = G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
  auto boudingBox = fpChemistryWorld->GetChemistryBoundary();
  G4double world_sizeXYZ = 2 * boudingBox->halfSideLengthInY();

  G4cout << "Volume size : " << G4BestUnit(world_sizeXYZ, "Length") << G4endl;

  G4double buffer;
  if (fPBC) {
    G4cout << "PeriodicBoundaryCondition is applied " << G4endl;
    buffer = 0.01 * nanometer;
  }
  else {
    buffer = 0;
    G4cout << "PeriodicBoundaryCondition is not applied " << G4endl;
  }

  auto solidWorld = new G4Box("World", 0.5 * (world_sizeXYZ - buffer),
                              0.5 * (world_sizeXYZ - buffer), 0.5 * (world_sizeXYZ - buffer));

  auto logicWorld = new G4LogicalVolume(solidWorld, water, "World");

  if (fPBC) {
    auto pbb = std::make_unique<PeriodicBoundaryBuilder>();
    fpPBCLogicVolume = pbb->Construct(logicWorld);
  }

  fpPhysWorld = new G4PVPlacement(nullptr,  // no rotation
                                  G4ThreeVector(),  // its position at (0,0,0)
                                  logicWorld,  // its logical volume
                                  "World",  // its name
                                  nullptr,  // its mother  volume
                                  false,  // no boolean operation
                                  0,  // copy number
                                  true);  // checking overlaps

  return fpPhysWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void DetectorConstruction::ConstructSDandField()
{
  G4SDManager::GetSDMpointer()->SetVerboseLevel(1);
  auto mfDetector = new G4MultiFunctionalDetector("mfDetector");
  // the order of G4VPrimitiveScorer to define ID number
  G4VPrimitiveScorer* pDose = new Scorer<Dose>();
  dynamic_cast<Scorer<Dose>*>(pDose)->SetChemistryWorld(fpChemistryWorld.get());
  mfDetector->RegisterPrimitive(pDose);
  G4VPrimitiveScorer* gValues = new Scorer<Gvalues>();
  dynamic_cast<Scorer<Gvalues>*>(gValues)->SetChemistryWorld(fpChemistryWorld.get());
  mfDetector->RegisterPrimitive(gValues);
  G4SDManager::GetSDMpointer()->AddNewDetector(mfDetector);
  G4String SDName;
  if (fPBC) {
    SDName = fpPBCLogicVolume->GetName();
  }
  else {
    SDName = fpPhysWorld->GetName();
  }
  SetSensitiveDetector(SDName, mfDetector);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::DefineCommands()
{
  fPBCMessenger =
    std::make_unique<G4GenericMessenger>(this, "/UHDR/Detector/", "Periodic Boundary Condition");
  auto& PBC = fPBCMessenger->DeclareProperty("PBC", fPBC);
  PBC.SetParameterName("PeriodicBoundaryCondition", true);
  PBC.SetDefaultValue("false");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....
