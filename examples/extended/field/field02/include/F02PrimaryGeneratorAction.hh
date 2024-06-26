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
/// \file field/field02/include/F02PrimaryGeneratorAction.hh
/// \brief Definition of the F02PrimaryGeneratorAction class
//
//
//
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef F02PrimaryGeneratorAction_h
#define F02PrimaryGeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4ParticleGun;
class G4Event;
class F02DetectorConstruction;
class F02PrimaryGeneratorMessenger;
class G4ParticleDefinition;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class F02PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    F02PrimaryGeneratorAction(F02DetectorConstruction*);
    ~F02PrimaryGeneratorAction() override;

  public:
    void GeneratePrimaries(G4Event*) override;
    void SetRndmFlag(G4String val) { fRndmFlag = val; }
    void SetXVertex(G4double x);
    void SetYVertex(G4double y);
    void SetZVertex(G4double z);

    static G4String GetPrimaryName();

  private:
    G4ParticleGun* fParticleGun = nullptr;  // pointer a to G4 service class
    F02DetectorConstruction* fDetector = nullptr;  // pointer to the geometry

    F02PrimaryGeneratorMessenger* fGunMessenger = nullptr;  // messenger of this class
    G4String fRndmFlag = "off";  // flag for random impact point

    static G4ParticleDefinition* fgPrimaryParticle;
    G4double fXVertex = 0.;
    G4double fYVertex = 0.;
    G4double fZVertex = 0.;
    G4bool fVertexDefined = false;
};

#endif
