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
/// \file field/field04/include/F04PrimaryGeneratorMessenger.hh
/// \brief Definition of the F04PrimaryGeneratorMessenger class
//

#ifndef F04PrimaryGeneratorMessenger_h
#define F04PrimaryGeneratorMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

class G4UIcmdWithAString;
class G4UIcmdWithADoubleAndUnit;

class F04PrimaryGeneratorAction;

class F04PrimaryGeneratorMessenger : public G4UImessenger
{
  public:
    F04PrimaryGeneratorMessenger(F04PrimaryGeneratorAction*);
    ~F04PrimaryGeneratorMessenger() override;

    void SetNewValue(G4UIcommand*, G4String) override;

  private:
    F04PrimaryGeneratorAction* fAction = nullptr;

    G4UIcmdWithAString* fRndmCmd = nullptr;
    G4UIcmdWithADoubleAndUnit* fSetXvertexCmd = nullptr;
    G4UIcmdWithADoubleAndUnit* fSetYvertexCmd = nullptr;
    G4UIcmdWithADoubleAndUnit* fSetZvertexCmd = nullptr;
};

#endif
