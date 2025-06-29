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
/// \file GB04BOptnBremSplitting.hh
/// \brief Definition of the GB04BOptnBremSplitting class
//
//---------------------------------------------------------------
//
// GB04BOptnBremSplitting
//
// Class Description:
//    A G4VBiasingOperation to change a process cross-section.
//
//
//---------------------------------------------------------------
//   Initial version                         Nov. 2014 M. Verderi

#ifndef GB04BOptnBremSplitting_hh
#define GB04BOptnBremSplitting_hh 1

#include "G4ParticleChange.hh"
#include "G4VBiasingOperation.hh"

class GB04BOptnBremSplitting : public G4VBiasingOperation
{
  public:
    // -- Constructor :
    GB04BOptnBremSplitting(G4String name);
    // -- destructor:
    ~GB04BOptnBremSplitting() override;

  public:
    // ----------------------------------------------
    // -- Methods from G4VBiasingOperation interface:
    // ----------------------------------------------
    // -- Unused:
    const G4VBiasingInteractionLaw*
    ProvideOccurenceBiasingInteractionLaw(const G4BiasingProcessInterface*,
                                          G4ForceCondition&) override
    {
      return nullptr;
    }

    // --Used:
    G4VParticleChange* ApplyFinalStateBiasing(const G4BiasingProcessInterface*, const G4Track*,
                                              const G4Step*, G4bool&) override;

    // -- Unsued:
    G4double DistanceToApplyOperation(const G4Track*, G4double, G4ForceCondition*) override
    {
      return DBL_MAX;
    }
    G4VParticleChange* GenerateBiasingFinalState(const G4Track*, const G4Step*) override
    {
      return nullptr;
    }

  public:
    // ----------------------------------------------
    // -- Additional methods, specific to this class:
    // ----------------------------------------------
    // -- Splitting factor:
    void SetSplittingFactor(G4int splittingFactor) { fSplittingFactor = splittingFactor; }
    G4int GetSplittingFactor() const { return fSplittingFactor; }

  private:
    G4int fSplittingFactor;
    G4ParticleChange fParticleChange;
};

#endif
