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
//  J. M. Quesada (August 2008).  
//  Based  on previous work by V. Lara
//
// Modified:
// 03.09.2008 by J. M. Quesada for external choice of inverse 
//                    cross section option (default OPTxs=2)
// 06.09.2008 by JMQ Also external choice has been added for
//     superimposed Coulomb barrier (if useSICB=true, default false) 
// 20.08.2010 V.Ivanchenko added int Z and A and cleanup; added 
//                        G4ParticleDefinition to constructor

#ifndef G4PreCompoundFragment_h
#define G4PreCompoundFragment_h 1

#include "G4VPreCompoundFragment.hh"

class G4PreCompoundFragment : public G4VPreCompoundFragment
{
public:  

  G4PreCompoundFragment(const G4ParticleDefinition*,
			G4VCoulombBarrier * aCoulombBarrier);
  
  ~G4PreCompoundFragment() override = default;

  // inverse cross section for a channel
  G4double CrossSection(G4double ekin);

  // the value of the recent inverse cross section for a channel
  G4double RecentXS() const { return recentXS; };

  G4PreCompoundFragment(const G4PreCompoundFragment &right) = delete;
  const G4PreCompoundFragment& 
  operator= (const G4PreCompoundFragment &right) = delete;
  G4bool operator==(const G4PreCompoundFragment &right) const = delete;
  G4bool operator!=(const G4PreCompoundFragment &right) const = delete;

private:	

  G4double GetOpt0(G4double ekin) const;
  
  G4int lastA{0};
  G4double muu{0.0};
  G4double recentXS{0.0};
};

#endif
