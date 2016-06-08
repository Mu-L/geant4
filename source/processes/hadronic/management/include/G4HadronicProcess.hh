// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4HadronicProcess.hh,v 1.5 1999/12/15 14:52:07 gunter Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
 // This is the top level Hadronic Process class
 // The inelastic, elastic, capture, and fission processes
 // should derive from this class
 // This is an abstract base class, since the pure virtual function
 // PostStepDoIt has not been defined yet.
 // Note:  there is no .cc file
 //
 // original by H.P.Wellisch
 // J.L. Chuma, TRIUMF, 10-Mar-1997
 // Last modified: 04-Apr-1997
 
#ifndef G4HadronicProcess_h
#define G4HadronicProcess_h 1
 
#include "globals.hh"
#include "G4VDiscreteProcess.hh"
#include "G4EnergyRangeManager.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4Element.hh"
#include "G4ElementVector.hh"
#include "G4ElementTable.hh"
#include "G4PhysicsTable.hh"
#include "G4PhysicsVector.hh"
#include "G4Nucleus.hh" 
#include "G4ReactionProduct.hh"
#include "g4rw/tpordvec.h"
#include "G4VIsotopeProduction.hh"
#include "G4IsoParticleChange.hh"
 
 class G4HadronicProcess : public G4VDiscreteProcess
 {
 public:
    
    G4HadronicProcess( const G4String &processName = "Hadronic" ) :
      G4VDiscreteProcess( processName )
    { isoIsOnAnyway = 0; }
    
    virtual ~G4HadronicProcess()
    { theProductionModels.clearAndDestroy(); }
    
    inline void RegisterMe( G4HadronicInteraction *a )
    { GetManagerPointer()->RegisterMe( a ); }
    
    inline G4HadronicInteraction *GetHadronicInteraction()
    { return theInteraction; }
    
    inline G4HadronicInteraction *ChooseHadronicInteraction(
     G4double kineticEnergy, G4Material *aMaterial, G4Element *anElement )
    { return GetManagerPointer()->
        GetHadronicInteraction( kineticEnergy, aMaterial, anElement ); }
    
    inline const G4EnergyRangeManager &GetEnergyRangeManager() const
    { return theEnergyRangeManager; }
    
    inline void SetEnergyRangeManager( const G4EnergyRangeManager &value )
    { theEnergyRangeManager = value; }

    G4Element * ChooseAandZ( const G4DynamicParticle *aParticle,
                             const G4Material *aMaterial );

    virtual G4VParticleChange *PostStepDoIt( const G4Track &aTrack, 
                                            const G4Step &aStep ) = 0;

    G4VParticleChange *GeneralPostStepDoIt( const G4Track &aTrack, 
                                           const G4Step &aStep );
    
    void SetDispatch( G4HadronicProcess *value )
    { dispatch=value; }
    
    virtual G4double GetMicroscopicCrossSection( const G4DynamicParticle *aParticle, 
                                                 const G4Element *anElement ) = 0;
    
    G4double GetCurrentZ()
    { return currentZ; }
    
    G4double GetCurrentN()
    { return currentN; }
    
    // Set methods for isotope production
    
    static void EnableIsotopeProductionGlobally();
    static void DisableIsotopeProductionGlobally();
    
    void EnableIsotopeCounting()  {isoIsOnAnyway = 1;}
    void DisableIsotopeCounting() {isoIsOnAnyway = -1;}
    
    void RegisterIsotopeProductionModel(G4VIsotopeProduction * aModel)
    { theProductionModels.insert(aModel); }

    static G4IsoParticleChange * GetIsotopeProductionInfo() 
    { 
      G4IsoParticleChange * anIsoResult = theIsoResult;
      if(theIsoResult) theOldIsoResult = theIsoResult;
      theIsoResult = NULL;
      return anIsoResult;
    }

 protected:
    
    inline G4EnergyRangeManager *GetManagerPointer()
    { return &theEnergyRangeManager; }
    
    G4EnergyRangeManager theEnergyRangeManager;
    
    G4HadronicInteraction *theInteraction;

    G4Nucleus targetNucleus;
    
 private:
    
    G4VParticleChange * DoIsotopeCounting(G4VParticleChange * aResult,
                                          const G4Track & aTrack,
                                          const G4Nucleus & aNucleus);
                                          
    G4IsoResult * ExtractResidualNucleus(const G4Track & aTrack,
                                         const G4Nucleus & aNucleus,
                                         G4VParticleChange * aResult);
    
 private:
    
    G4double currentZ;
    G4double currentN;
    G4HadronicProcess *dispatch;

// swiches for isotope production
    
    static G4bool isoIsEnabled; // true or false; local swich overrides
    G4int isoIsOnAnyway; // true(1), false(-1) or default(0)
    
    G4IsoParticleChange theIsoPC;
    G4RWTPtrOrderedVector<G4VIsotopeProduction> theProductionModels;

    static G4IsoParticleChange * theIsoResult;
    static G4IsoParticleChange * theOldIsoResult;
    
 };
 
#endif
 
