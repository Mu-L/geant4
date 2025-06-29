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
//---------------------------------------------------------------------------
// ClassName: G4HadronicParametersMessenger
// Author: Alberto Ribon
// Date: May 2020
// Description: messenger class for the class G4HadronicParameters
//---------------------------------------------------------------------------

#ifndef G4HadronicParametersMessenger_h
#define G4HadronicParametersMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

class G4UIdirectory;
class G4UIcommand;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithABool;
class G4HadronicParameters;


class G4HadronicParametersMessenger : public G4UImessenger {
  public:
    explicit G4HadronicParametersMessenger( G4HadronicParameters* inputHadParam );
    ~G4HadronicParametersMessenger();
    void SetNewValue ( G4UIcommand *command, G4String newValues ) override;
  private:
    G4HadronicParameters* theHadronicParameters;
    G4UIdirectory* theDirectory;
    G4UIcmdWithAnInteger* theVerboseCmd;
    G4UIcmdWithADoubleAndUnit* theMaxEnergyCmd;
    G4UIcmdWithABool* theCRCoalescenceCmd;
    G4UIcommand* theDumpCmd;
};

#endif
