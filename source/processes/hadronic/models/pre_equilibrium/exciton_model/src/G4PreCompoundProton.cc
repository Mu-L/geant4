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
// -------------------------------------------------------------------
//
// GEANT4 Class file
//
//
// File name:     G4PreCompoundProton
//
// Author:         V.Lara
//
// Modified:  
// 21.08.2008 J. M. Quesada added external choice of inverse cross section 
// 21.08.2008 J. M. Quesada added external choice for superimposed Coulomb 
//                          barrier (if useSICB=true) 
// 20.08.2010 V.Ivanchenko added G4Pow and G4PreCompoundParameters pointers
//                         use int Z and A and cleanup
//

#include "G4PreCompoundProton.hh"
#include "G4CoulombBarrier.hh"
#include "G4Proton.hh"
#include "G4DeexPrecoUtility.hh"

G4PreCompoundProton::G4PreCompoundProton()
  : G4PreCompoundNucleon(G4Proton::Proton(), new G4CoulombBarrier(1, 1))
{}

G4double G4PreCompoundProton::GetRj(G4int nParticles, G4int nCharged) const
{
  G4double rj = 0.0;
  if(nParticles > 0) { 
    rj = static_cast<G4double>(nCharged)/static_cast<G4double>(nParticles);
  }
  return rj;
}

G4double G4PreCompoundProton::GetAlpha() const
{
  return 1.0 + G4DeexPrecoUtility::ProtonCValue(theResZ);
}
  
