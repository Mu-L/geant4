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
// GEANT4 Class file
//
// File name:     G4PreCompoundTransitionsInt
//
// Author:        V.Ivantchenko, 25 January 2025
//

#include "G4PreCompoundTransitionsInt.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include "G4NuclearLevelData.hh"
#include "G4DeexPrecoParameters.hh"
#include "G4Fragment.hh"
#include "G4Proton.hh"
#include "G4Exp.hh"
#include "G4Log.hh"

G4PreCompoundTransitionsInt::G4PreCompoundTransitionsInt(G4int verb)
  : fVerbose(verb)
{
  proton = G4Proton::Proton();
  fNuclData = G4NuclearLevelData::GetInstance();
  G4DeexPrecoParameters* param = fNuclData->GetParameters();
  FermiEnergy = param->GetFermiEnergy();
  r0 = param->GetTransitionsR0();
}

G4double G4PreCompoundTransitionsInt::
CalculateProbability(const G4Fragment & aFragment)
{
  // Number of Particles 
  G4int P = aFragment.GetNumberOfParticles();
  // Number of Excitons 
  //G4int N = P+H;
  // Nucleus 
  G4int A = aFragment.GetA_asInt();
  G4int Z = aFragment.GetZ_asInt();
  G4double U = aFragment.GetExcitationEnergy();
  TransitionProb2 = 0.0;
  TransitionProb3 = 0.0;
  /*
  G4cout << "G4PreCompoundTransitions::CalculateProbability H/P/N/Z/A= " 
	 << H << " " << P << " " << N << " " << Z << " " << A <<G4endl;
  G4cout << aFragment << G4endl;
  */
  if(U < 10*eV || 0 == P) { return 0.0; }
  
  //J. M. Quesada (Feb. 08) new physics
  // OPT=1 Transitions are calculated according to Gudima's paper 
  //       (original in G4PreCompound from VL) 
  // OPT=2 Transitions are calculated according to Gupta's formulae
  //
  static const G4double sixdpi2 = 6.0/CLHEP::pi2;
  G4double GE = sixdpi2*U*fNuclData->GetLevelDensity(Z, A, U);
  if (useCEMtr) {
    // Relative Energy (T_{rel})
    G4double RelativeEnergy = 1.6*FermiEnergy + U/G4double(2*P);
    
    // Sample kind of nucleon-projectile 
    G4bool ChargedNucleon(false);
    if(G4lrint(P*G4UniformRand()) <= aFragment.GetNumberOfCharged()) {
      ChargedNucleon = true; 
    }
    
    // Relative Velocity: 
    // <V_{rel}>^2
    G4double RelativeVelocitySqr;
    if (ChargedNucleon) { 
      RelativeVelocitySqr = 2*RelativeEnergy/CLHEP::proton_mass_c2; 
    } else { 
      RelativeVelocitySqr = 2*RelativeEnergy/CLHEP::neutron_mass_c2; 
    }
    // <V_{rel}>
    G4double RelativeVelocity = std::sqrt(RelativeVelocitySqr);
    
    // Proton-Proton Cross Section
    G4double ppXSection = 
      (10.63/RelativeVelocitySqr - 29.92/RelativeVelocity + 42.9)
      * CLHEP::millibarn;
    // Proton-Neutron Cross Section
    G4double npXSection = 
      (34.10/RelativeVelocitySqr - 82.20/RelativeVelocity + 82.2)
      * CLHEP::millibarn;
    
    // Averaged Cross Section: \sigma(V_{rel})
    G4double xs = (ChargedNucleon) ?
      ((Z-1)*ppXSection + (A-Z)*npXSection)/G4double(A-1) :
      ((A-Z-1)*ppXSection + Z*npXSection)/G4double(A-1);
    
    // Fermi relative energy ratio
    G4double FermiRelRatio = FermiEnergy/RelativeEnergy;
    
    // This factor is introduced to take into account the Pauli principle
    G4double PauliFactor = 1.0 - 1.4*FermiRelRatio;
    if (FermiRelRatio > 0.5) {
      G4double x = 2.0 - 1.0/FermiRelRatio;
      PauliFactor += 0.4*FermiRelRatio*x*x*std::sqrt(x);
    }
    // Interaction volume 
    G4double xx = 2*r0 + CLHEP::hbarc/(CLHEP::proton_mass_c2*RelativeVelocity);
    G4double Vint = CLHEP::pi*xx*xx*xx/0.75;
    
    // Transition probability for \Delta n = +2    
    TransitionProb1 = std::max(0.0, xs*PauliFactor
      *std::sqrt(2.0*RelativeEnergy/CLHEP::proton_mass_c2)/Vint);

    //JMQ 281009  phenomenological factor in order to increase 
    //   equilibrium contribution
    //   G4double factor=5.0;
    //   TransitionProb1 *= factor;
    
    // GE = g*E where E is Excitation Energy
    G4double Fph = G4double(P*(P - 1))*0.5;
    
    if(!useNGB) { 
        
      // F(p+1,h+1)
      G4double Fph1 = Fph + P;

      static const G4double plimit = 100;

      //JMQ/AH  bug fixed: if (U-Fph < 0.0) 
      if (GE > Fph1) { 
        G4double x0 = GE-Fph;
	G4double x1 = (2*P + 1)*G4Log(x0/(GE-Fph1));
	if(x1 < plimit) {
	  x1 = G4Exp(x1)*TransitionProb1/x0;
    
	  // Transition probability for \Delta n = -2 (at F(p,h) = 0)
	  TransitionProb2 = std::max(0.0, (2*P*P*(2*P + 1)*(P - 1))*x1/x0);
        
	  // Transition probability for \Delta n = 0 (at F(p,h) = 0)
	  TransitionProb3 = std::max(0.0, ((2*P+1)*(3*P-1))*x1);
	}
      }
    }

  } else {
    //JMQ: Transition probabilities from Gupta's work    
    // GE = g*E where E is Excitation Energy
    TransitionProb1 = std::max(0.0, U*(4.2e+12 - 3.6e+10*U/G4double(2*P+1)))
      /(16*CLHEP::c_light); 

    if (!useNGB) { 
      TransitionProb2 = (2*(2*P-1)*(P-1)*P*P)*TransitionProb1/(GE*GE);  
    }
  }
  return TransitionProb1 + TransitionProb2 + TransitionProb3;
}

void G4PreCompoundTransitionsInt::PerformTransition(G4Fragment& frag)
{
  G4int A = frag.GetA_asInt();
  G4int Z = frag.GetZ_asInt();
  G4int Npart = frag.GetNumberOfParticles();
  G4int Ncharged = frag.GetNumberOfCharged();

  G4double ChosenTransition = 
    G4UniformRand()*(TransitionProb1 + TransitionProb2 + TransitionProb3);

  G4int deltaN = 0;
  // Number of excited particles inside the fragment
  if (ChosenTransition <= TransitionProb1) {
    if (Npart < A) { deltaN = 1; }
  } 
  else if (ChosenTransition <= TransitionProb1+TransitionProb2) {
    if (Npart > 0) { deltaN = -1; }
  }
  if (deltaN != 0) { 
    if (G4lrint(Npart*G4UniformRand()) <= Ncharged) {
      Ncharged += deltaN;
      Ncharged = std::min(std::max(Ncharged, 0), Z);
    }
  }
  Npart += deltaN;
  Ncharged = std::min(Ncharged, Npart);

  // update fragment 
  frag.SetNumberOfExcitedParticle(Npart, Ncharged);
  if (2 < fVerbose) { 
    G4cout << "           # After transition Npart=" << Npart
	   << " Ncharged=" << Ncharged << " deltaN=" << deltaN << G4endl;
  }
}

