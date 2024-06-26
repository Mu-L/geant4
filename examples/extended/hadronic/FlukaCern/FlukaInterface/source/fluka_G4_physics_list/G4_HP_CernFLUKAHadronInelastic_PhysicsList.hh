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
// A physics list based on FTFP_BERT_HP LIV,
// but with full hadron inelastic physics replaced by the one from FLUKA.CERN.
//
// Author: G.Hugo, 01 August 2022
//
// ***************************************************************************
#ifdef G4_USE_FLUKA
#  ifndef G4_HP_CERNFLUKAHADRONINELASTIC_PHYSICSLIST_HH
#    define G4_HP_CERNFLUKAHADRONINELASTIC_PHYSICSLIST_HH

// G4
#    include "G4VModularPhysicsList.hh"
#    include "globals.hh"

class G4_HP_CernFLUKAHadronInelastic_PhysicsList final : public G4VModularPhysicsList
{
  public:
    G4_HP_CernFLUKAHadronInelastic_PhysicsList(G4int verbose = 1);

    G4_HP_CernFLUKAHadronInelastic_PhysicsList(const G4_HP_CernFLUKAHadronInelastic_PhysicsList&) =
      delete;
    G4_HP_CernFLUKAHadronInelastic_PhysicsList&
    operator=(const G4_HP_CernFLUKAHadronInelastic_PhysicsList&) = delete;

    virtual void SetCuts();
};

#  endif
#endif  // G4_USE_FLUKA
