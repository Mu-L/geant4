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
/// \file DetectorConstruction.hh
/// \brief Definition of the DetectorConstruction class
//
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4Cache.hh"
#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

class G4Box;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Material;
class DetectorMessenger;

const G4int kMaxAbsor = 10;  // 0 + 9

class G4GlobalMagFieldMessenger;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction();
    ~DetectorConstruction() override;

  public:
    G4Material* MaterialWithSingleIsotope(G4String, G4String, G4double, G4int, G4int);

    void SetNbOfAbsor(G4int);
    void SetAbsorMaterial(G4int, const G4String&);
    void SetAbsorThickness(G4int, G4double);

    void SetAbsorSizeYZ(G4double);

    G4VPhysicalVolume* Construct() override;
    void ConstructSDandField() override;

  public:
    G4int GetNbOfAbsor() { return fNbOfAbsor; }
    G4Material* GetAbsorMaterial(G4int i) { return fAbsorMaterial[i]; };
    G4double GetAbsorThickness(G4int i) { return fAbsorThickness[i]; };
    G4double GetXfront(G4int i) { return fXfront[i]; };

    G4double GetAbsorSizeX() { return fAbsorSizeX; };
    G4double GetAbsorSizeYZ() { return fAbsorSizeYZ; };

    G4double GetWorldSizeX() { return fWorldSizeX; };
    G4double GetWorldSizeYZ() { return fWorldSizeYZ; };

    void PrintParameters();

  private:
    G4int fNbOfAbsor = 0;
    G4Material* fAbsorMaterial[kMaxAbsor];
    G4double fAbsorThickness[kMaxAbsor];
    G4double fXfront[kMaxAbsor];

    G4double fAbsorSizeX = 0.;
    G4double fAbsorSizeYZ = 0.;

    G4double fWorldSizeX = 0.;
    G4double fWorldSizeYZ = 0.;
    G4Material* fDefaultMaterial = nullptr;

    G4VPhysicalVolume* fPhysiWorld = nullptr;

    DetectorMessenger* fDetectorMessenger = nullptr;
    G4Cache<G4GlobalMagFieldMessenger*> fFieldMessenger = nullptr;

  private:
    void DefineMaterials();
    void ComputeParameters();
    G4VPhysicalVolume* ConstructVolumes();
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
