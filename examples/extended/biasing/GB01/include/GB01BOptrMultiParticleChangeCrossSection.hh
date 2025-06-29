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
/// \file GB01/include/GB01BOptrMultiParticleChangeCrossSection.hh
/// \brief Definition of the GB01BOptrMultiParticleChangeCrossSection class
//
//-----------------------------------------------------------------
//
// GB01BOptrMultiParticleChangeCrossSection
//
// Class Description:
//        A G4VBiasingOperator concrete implementation example to
//    illustrate how an operator can make use of an other operator.
//        In the present case, the GB01BOptrChangeCrossSection
//    operator, that is made to bias processes of one particle
//    type is instancied several times, one for each particle type
//    to bias.
//
//-----------------------------------------------------------------
//

#ifndef GB01BOptrMultiParticleChangeCrossSection_hh
#define GB01BOptrMultiParticleChangeCrossSection_hh 1

#include "G4VBiasingOperator.hh"
class GB01BOptrChangeCrossSection;
class G4ParticleDefinition;

#include <map>

class GB01BOptrMultiParticleChangeCrossSection : public G4VBiasingOperator
{
  public:
    GB01BOptrMultiParticleChangeCrossSection();
    ~GB01BOptrMultiParticleChangeCrossSection() override = default;

    // ---------------------------------
    // -- Method specific to this class:
    // ---------------------------------
    // -- Each particle type for which its name is passed will be biased; *provided*
    // -- that the proper calls to biasingPhysics->Bias(particleName) have been done
    // -- in the main program.
    void AddParticle(G4String particleName);

  private:
    // -----------------------------
    // -- Mandatory from base class:
    // -----------------------------
    // -- This method returns a biasing operation that will bias the physics process occurence:
    G4VBiasingOperation*
    ProposeOccurenceBiasingOperation(const G4Track* track,
                                     const G4BiasingProcessInterface* callingProcess) override;
    // -- Methods not used:
    G4VBiasingOperation*
    ProposeFinalStateBiasingOperation(const G4Track*, const G4BiasingProcessInterface*) override
    {
      return nullptr;
    }
    G4VBiasingOperation*
    ProposeNonPhysicsBiasingOperation(const G4Track*, const G4BiasingProcessInterface*) override
    {
      return nullptr;
    }

  private:
    // -- ("using" is to avoid compiler complaining against (false) method shadowing.)
    using G4VBiasingOperator::OperationApplied;

    // -- Optionnal base class method implementation.
    // -- This method is called to inform the operator that a proposed operation has been applied.
    // -- In the present case, it means that a physical interaction occured (interaction at
    // -- PostStepDoIt level):
    void OperationApplied(const G4BiasingProcessInterface* callingProcess,
                          G4BiasingAppliedCase biasingCase,
                          G4VBiasingOperation* occurenceOperationApplied,
                          G4double weightForOccurenceInteraction,
                          G4VBiasingOperation* finalStateOperationApplied,
                          const G4VParticleChange* particleChangeProduced) override;

  public:
    // -- Optionnal base class method. It is called at the time a tracking of a particle starts:
    void StartTracking(const G4Track* track) override;

  private:
    // -- List of associations between particle types and biasing operators:
    std::map<const G4ParticleDefinition*, GB01BOptrChangeCrossSection*> fBOptrForParticle;
    std::vector<const G4ParticleDefinition*> fParticlesToBias;
    GB01BOptrChangeCrossSection* fCurrentOperator;

    // -- count number of biased interations for current track:
    G4int fnInteractions;
};

#endif
