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
// G4GeometryMessenger
//
// Class description:
//
// A messenger defining commands for debugging, verifying
// and controlling the detector geometry and navigation.

// Author: Gabriele Cosmo (CERN), 24 October 2001.
// --------------------------------------------------------------------
#ifndef G4GeometryMessenger_hh
#define G4GeometryMessenger_hh 1

#include "G4Types.hh"
#include "G4UImessenger.hh"
#include "G4ThreeVector.hh"

#include <vector>

class G4UIdirectory;
class G4UIcommand;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithAString;
class G4TransportationManager;
class G4GeomTestVolume;

/**
 * @brief G4GeometryMessenger is a messenger defining commands for debugging,
 * verifying and controlling the detector geometry and navigation.
 */

class G4GeometryMessenger : public G4UImessenger
{
  public:

    /**
     * Constructor and Destructor.
     */
    G4GeometryMessenger(G4TransportationManager* tman);
    ~G4GeometryMessenger() override;

    /**
     * Sets/gets values for UI command.
     */
    void SetNewValue( G4UIcommand* command, G4String newValues ) override;
    G4String GetCurrentValue( G4UIcommand* command ) override;
  
  private:

    void Init();
    void CheckGeometry();
    void ResetNavigator();
    void SetVerbosity(const G4String& newValue);
    void SetCheckMode(const G4String& newValue);
    void SetPushFlag(const G4String& newValue);
    void RecursiveOverlapTest();
    void TreeOverlapTest();

    struct OverlapMode
    {
      inline static const G4String placed = "placed";
      inline static const G4String logical = "logical";
    };

    G4UIdirectory             *geodir, *navdir, *testdir;
    G4UIcmdWithABool          *chkCmd, *pchkCmd, *verCmd, *parCmd;
    G4UIcmdWithoutParameter   *resCmd;
    G4UIcmdWithAString        *recCmd;
    G4UIcmdWithADoubleAndUnit *tolCmd;
    G4UIcmdWithAnInteger      *verbCmd, *rslCmd, *rcsCmd, *rcdCmd, *errCmd;

    G4double tol = 0.0;
    G4int recLevel = 0, recDepth = -1;
    G4bool checkParallelWorlds = false;

    G4TransportationManager* tmanager;
    std::vector<G4GeomTestVolume*> tvolumes{};
};

#endif
