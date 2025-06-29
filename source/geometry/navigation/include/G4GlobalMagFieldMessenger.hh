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
// G4GlobalMagFieldMessenger
//
// Class description:
//
// Global uniform magnetic field messenger class. 
//
// It defines UI commands:
// - /globalField/setValue vx vy vz unit
// - /globalField/verbose level
//
// It creates/deletes the global uniform magnetic field and
// activates/inactivates it according to the set field value.
// The field value can be changed either interactively via 
// the UI command or via SetFieldValue() function.

// Author: Ivana Hrivnacova (IN2P3/IJCLab Orsay), 28 August 2013
// --------------------------------------------------------------------
#ifndef G4GlobalMagFieldMessenger_hh
#define G4GlobalMagFieldMessenger_hh 1

#include "globals.hh"
#include "G4UImessenger.hh"

class G4UniformMagField;
class G4UIdirectory;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithAnInteger;

/**
 * @brief G4GlobalMagFieldMessenger, a global uniform magnetic field messenger
 * class. It creates/deletes the global uniform magnetic field and
 * activates/inactivates it according to the set field value.
 * The field value can be changed either interactively via the UI command or
 * via the SetFieldValue() function.
 */

class G4GlobalMagFieldMessenger : public G4UImessenger
{
  public:

    /**
     * Constructor and Destructor.
     */
    G4GlobalMagFieldMessenger(const G4ThreeVector& value = G4ThreeVector());
    ~G4GlobalMagFieldMessenger() override;
    
    /**
     * Setter for UI command.
     */
    void SetNewValue(G4UIcommand*, G4String) override;

    /**
     * Setter and accessor for the field value.
     */
    void SetFieldValue(const G4ThreeVector& value);
    G4ThreeVector GetFieldValue() const;
    
    /**
     * Verbosity control.
     */
    inline void SetVerboseLevel(G4int verboseLevel);
    inline G4int GetVerboseLevel() const;
    
  private:

    void SetField(const G4ThreeVector& value, const G4String& inFunction);
    
    G4UniformMagField* fMagField = nullptr;
    G4int fVerboseLevel = 0;

    G4UIdirectory* fDirectory = nullptr;
    G4UIcmdWith3VectorAndUnit* fSetValueCmd = nullptr;
    G4UIcmdWithAnInteger* fSetVerboseCmd = nullptr;
};

// --------------------------------------------------------------------
// inline functions
// --------------------------------------------------------------------

inline void G4GlobalMagFieldMessenger::SetVerboseLevel(G4int verboseLevel)
{
  fVerboseLevel = verboseLevel;
}

inline G4int G4GlobalMagFieldMessenger::GetVerboseLevel() const
{
  return fVerboseLevel;
}
    
#endif
