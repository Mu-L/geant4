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

#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4DNABoundingBox.hh"
#include "G4GenericMessenger.hh"
#include "G4VChemistryWorld.hh"
#include "G4VUserDetectorConstruction.hh"

#include <memory>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

class G4VPhysicalVolume;
class G4VPrimitiveScorer;

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction();
    ~DetectorConstruction() override = default;

    G4VPhysicalVolume* Construct() final;
    void ConstructSDandField() final;
    G4VChemistryWorld* GetChemistryWorld() const { return fpChemistryWorld.get(); }

  private:
    std::unique_ptr<G4VChemistryWorld> fpChemistryWorld;
    G4bool fPBC = false;
    G4VPhysicalVolume* fpPhysWorld = nullptr;
    G4LogicalVolume* fpPBCLogicVolume = nullptr;
    std::unique_ptr<G4GenericMessenger> fPBCMessenger;
    void DefineCommands();
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
