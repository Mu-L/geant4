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
/*
 * G4EmDNAChemistry_option3.hh
 *
 *  Created on: Jul 23, 2019
 *      Author: W. G. Shin
 *              J. Ramos-Mendez and B. Faddegon
*/

#ifndef G4EmDNAChemistry_option3_hh
#define G4EmDNAChemistry_option3_hh 1

#include "G4VPhysicsConstructor.hh"
#include "G4VUserChemistryList.hh"
#include "globals.hh"
#include "G4ChemTimeStepModel.hh"
class G4DNAMolecularReactionTable;
class G4DNAMolecularReactionData;

class G4EmDNAChemistry_option3 : public G4VUserChemistryList,
                         public G4VPhysicsConstructor
{

public:

  G4EmDNAChemistry_option3();
  ~G4EmDNAChemistry_option3() override = default;

  void ConstructParticle() override
  {
    ConstructMolecule();
  }
  void ConstructMolecule() override;
  void ConstructProcess() override;

  void ConstructDissociationChannels() override;
  void ConstructReactionTable(G4DNAMolecularReactionTable* reactionTable) override;
  void ConstructTimeStepModel(G4DNAMolecularReactionTable* reactionTable) override;
 private:
   static void SetReactionType(G4DNAMolecularReactionData* pData, G4ChemTimeStepModel model);
};

#endif
