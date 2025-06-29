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
// G4FermiBreakUpAN is an alternative realisation of Fermi Break Up
// de-excitation by A. Novikov, Yandex and MIPT (January 2025)
// under supervision of Aleksandr Svetlichnyi, INR RAS and MIPT
//
// The model originally developed in J.P. Bondorf, A.S. Botvina, A.S. Iljinov,
// I.N. Mishustin, and K. Sneppen, "Statistical multifragmentation of nuclei."
// Physics Reports, 257(3):133–221, Jun 1995.
// https://doi.org/10.1016/0370-1573(94)00097-M, doi:10.1016/0370-1573(94)00097-m.
//

#ifndef G4FERMIBREAKUPAN_HH
#define G4FERMIBREAKUPAN_HH

#include "G4FermiDataTypes.hh"
#include "G4FermiParticle.hh"
#include "G4FermiSplitter.hh"
#include "G4VFermiBreakUp.hh"
#include "globals.hh"

#include <memory>

class G4FermiBreakUpAN : public G4VFermiBreakUp
{
  private:
    class PossibleSplits
    {
      private:
        using NucleiSplits = std::vector<G4FermiFragmentVector>;

      public:
        PossibleSplits() = default;
        PossibleSplits& operator=(PossibleSplits&&) noexcept = default;

        PossibleSplits(const G4FermiAtomicMass maxAtomicMass);

        const NucleiSplits& GetSplits(const G4FermiAtomicMass atomicMass,
                                      const G4FermiChargeNumber chargeNumber) const;

        void InsertSplits(const G4FermiAtomicMass atomicMass,
                          const G4FermiChargeNumber chargeNumber,
			  NucleiSplits&& splits);

      private:
        std::vector<NucleiSplits> splits_;
    };

  public:
    explicit G4FermiBreakUpAN(G4int verbosity = 0);
    ~G4FermiBreakUpAN() override = default;

    void Initialise() override;

    // check if the Fermi Break Up model can be used
    G4bool IsApplicable(G4int Z, G4int A, G4double eexc) const override;

    // vector of products is added to the provided vector
    // if no decay channel is found out for the primary fragment
    // then it is added to the results vector
    // if primary decays then it is deleted
    void BreakFragment(G4FragmentVector* results, G4Fragment* theNucleus) override;

    std::vector<G4FermiParticle> BreakItUp(const G4FermiParticle& nucleus) const;

  private:
    std::vector<G4FermiParticle> SplitToParticles(const G4FermiParticle& sourceParticle,
                                                  const G4FermiFragmentVector& split) const;

    // improve performance, reusing allocated memory
    mutable std::vector<G4double> weights_;
    PossibleSplits splits_;

    G4int secID_;
    G4int verbosity_ = 0;
};

#endif  // G4FERMIBREAKUP_HH
