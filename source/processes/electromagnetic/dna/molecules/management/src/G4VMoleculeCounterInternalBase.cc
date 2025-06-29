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
// Author: Christian Velten (2025)

#include "G4VMoleculeCounterInternalBase.hh"

namespace G4
{
namespace MoleculeCounter
{

G4bool FixedTimeComparer::operator()(const G4double& a, const G4double& b) const
{
  if (std::fabs(a - b) < fPrecision) {
    return false;
  }
  return a < b;
}

G4ThreadLocal G4double FixedTimeComparer::fPrecision = 0.1 * picosecond;
}  // namespace MoleculeCounter
}  // namespace G4

G4ThreadLocal G4int G4VMoleculeCounterInternalBase::_createdCounters = 0;

//------------------------------------------------------------------------------

G4VMoleculeCounterInternalBase::G4VMoleculeCounterInternalBase() : fId(_createdCounters++) {}

G4VMoleculeCounterInternalBase::G4VMoleculeCounterInternalBase(const G4String& name)
  : fId(_createdCounters++), fName(name)
{}

//------------------------------------------------------------------------------

void G4VMoleculeCounterInternalBase::SetFixedTimePrecision(G4double precision)
{
  G4::MoleculeCounter::FixedTimeComparer::fPrecision = precision;
}
