// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4LightMedia.hh,v 1.2 1999/12/15 14:53:40 gunter Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
 // Hadronic Process: Light Media Charge and/or Strangeness Exchange
 // J.L. Chuma, TRIUMF, 21-Feb-1997
 // Last modified: 21-Feb-1997

#ifndef G4LightMedia_h
#define G4LightMedia_h 1
 
#include "globals.hh"
#include "G4DynamicParticle.hh"
#include "G4Nucleus.hh"
#include "G4ParticleTypes.hh" 
 
 class G4LightMedia
 {
 public:
    
    G4LightMedia() { }

    G4LightMedia( const G4LightMedia &right )
    { *this = right; }
    
    ~G4LightMedia() { }
    
    G4LightMedia & operator=( const G4LightMedia &right )
    { return *this; }
    
    G4bool operator==( const G4LightMedia &right ) const
    { return ( this == (G4LightMedia *) &right ); }
    
    G4bool operator!=( const G4LightMedia &right ) const
    { return ( this != (G4LightMedia *) &right ); }
    
    G4DynamicParticle * PionPlusExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * PionMinusExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * KaonPlusExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * KaonZeroShortExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * KaonZeroLongExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * KaonMinusExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * ProtonExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * AntiProtonExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * NeutronExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * AntiNeutronExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * LambdaExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * AntiLambdaExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * SigmaPlusExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * SigmaMinusExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * AntiSigmaPlusExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * AntiSigmaMinusExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * XiZeroExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * XiMinusExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * AntiXiZeroExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * AntiXiMinusExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * OmegaMinusExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
    G4DynamicParticle * AntiOmegaMinusExchange(
     const G4DynamicParticle *incidentParticle,
     const  G4Nucleus &aNucleus );
    
 private:
    
 };
 
#endif
 
