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
// Author: Mathieu Karamitros
// Modified by Christian Velten (2024)

// The code is developed in the framework of the ESA AO7146
//
// We would be very happy hearing from you, send us your feedback! :)
//
// In order for Geant4-DNA to be maintained and still open-source,
// article citations are crucial.
// If you use Geant4-DNA chemistry and you publish papers about your software,
// in addition to the general paper on Geant4-DNA:
//
// Int. J. Model. Simul. Sci. Comput. 1 (2010) 157–178
//
// we would be very happy if you could please also cite the following
// reference papers on chemistry:
//
// J. Comput. Phys. 274 (2014) 841-882
// Prog. Nucl. Sci. Tec. 2 (2011) 503-508

#pragma once

#include "G4MolecularConfiguration.hh"
#include "G4VUserMoleculeCounter.hh"

#include <map>
#include <vector>

//------------------------------------------------------------------------------

struct G4MoleculeCounterIndex : public G4VMoleculeCounter::G4VMoleculeCounterIndex
{
    const G4MolecularConfiguration* Molecule;
    G4MoleculeCounterIndex() : Molecule(nullptr) {}
    explicit G4MoleculeCounterIndex(const G4MolecularConfiguration* molecule) : Molecule(molecule)
    {}
    ~G4MoleculeCounterIndex() override = default;
    G4bool operator<(G4VMoleculeCounterIndex const& other) const override
    {
      return std::less{}(Molecule, static_cast<const G4MoleculeCounterIndex&>(other).Molecule);
    }
    G4bool operator==(G4VMoleculeCounterIndex const& other) const override
    {
      return std::equal_to{}(Molecule, static_cast<const G4MoleculeCounterIndex&>(other).Molecule);
    }
    G4String GetInfo() const override
    {
      G4String null = "null";
      if (Molecule == nullptr) {
        return null;
      }
      else {
        G4String name = Molecule->GetName();
        G4String info = "Molecule: " + name;
        return info;
      }
    }
    const G4MolecularConfiguration* GetMolecule() const override { return Molecule; }
};

class G4MoleculeCounter : public G4VUserMoleculeCounter<G4MoleculeCounterIndex>
{
    //----------------------------------------------------------------------------
  public:
    G4MoleculeCounter();
    G4MoleculeCounter(G4String);
    ~G4MoleculeCounter() override = default;

    void InitializeUser() override;

  public:
    std::unique_ptr<G4VMoleculeCounterIndex> BuildIndex(const G4Track*) const override;
    std::unique_ptr<G4VMoleculeCounterIndex> BuildIndex(const G4Track*,
                                                        const G4StepPoint*) const override;
    std::unique_ptr<G4VMoleculeCounterIndex>
    BuildSimpleIndex(const G4MolecularConfiguration*) const override;
};
