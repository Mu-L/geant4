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
// P. Arce, June-2014 Conversion neutron_hp to particle_hp
//
#ifndef G4ParticleHPThermalScatteringNames_h
#define G4ParticleHPThermalScatteringNames_h 1

// Class Description
// Name list of Elements for a high precision (based on evaluated data
// libraries) description of themal neutron scattering below 4 eV;
// Based on Thermal neutron scattering files
// from the evaluated nuclear data files ENDF/B-VI, Release2
// To be used in your physics list in case you need this physics.
// In this case you want to register an object of this class with
// the corresponding process.
// Class Description - End

// 15-Nov-06 First implementation is done by T. Koi (SLAC/SCCS)

#include "globals.hh"

#include <map>

class G4ParticleHPThermalScatteringNames
{
  public:

    G4ParticleHPThermalScatteringNames();

    ~G4ParticleHPThermalScatteringNames() = default;

    G4bool IsThisThermalElement(const G4String&) const;
    G4bool IsThisThermalElement(const G4String&, const G4String&) const;
    inline std::size_t GetSize() const { return names.size(); }
    inline const G4String& GetTS_NDL_Name(const G4String& nameG4Element) const
    {
      auto p = names.find(nameG4Element);
      return (p != names.end()) ? p->second : sss;
    }
    inline const G4String& GetTS_NDL_Name(const G4String& material,
                                          const G4String& element) const
    {
      auto p = nist_names.find(std::pair<G4String, G4String>(material, element));
      return (p != nist_names.end()) ? p->second : sss;
    }

    // For user prepared thermal files
    // Name of G4Element , Name of NDL file
    void AddThermalElement(const G4String&, const G4String&);

  private:

    G4String sss{""};
    //              G4Element  NDL name
    std::map<G4String, G4String> names;

    //                         G4Material G4Element    NDL name
    std::map<std::pair<G4String, G4String>, G4String> nist_names;
};

#endif
