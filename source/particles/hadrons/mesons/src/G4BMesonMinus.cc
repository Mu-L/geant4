// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4BMesonMinus.cc,v 1.4 2000/02/27 05:57:42 kurasige Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
// 
// ----------------------------------------------------------------------
//      GEANT 4 class implementation file
//
//      For information related to this code contact:
//      CERN, CN Division, ASD Group
//
//      Created                 Hisaya Kurashige, 16 June 1997
// **********************************************************************
//  Change both methods to get the pointer into non-inlined H.Kurashige 4 Aug. 1998
// ----------------------------------------------------------------

#include "g4std/fstream"
#include "g4std/iomanip"

#include "G4BMesonMinus.hh"

#include "G4DecayTable.hh"

// ######################################################################
// ###                          BMesonMinus                            ###
// ######################################################################

G4BMesonMinus::G4BMesonMinus(
       const G4String&     aName,        G4double            mass,
       G4double            width,        G4double            charge,   
       G4int               iSpin,        G4int               iParity,    
       G4int               iConjugation, G4int               iIsospin,   
       G4int               iIsospin3,    G4int               gParity,
       const G4String&     pType,        G4int               lepton,      
       G4int               baryon,       G4int               encoding,
       G4bool              stable,       G4double            lifetime,
       G4DecayTable        *decaytable )
 : G4VMeson( aName,mass,width,charge,iSpin,iParity,
             iConjugation,iIsospin,iIsospin3,gParity,pType,
             lepton,baryon,encoding,stable,lifetime,decaytable )
{
   SetParticleSubType("B");
}

// ......................................................................
// ...                 static member definitions                      ...
// ......................................................................
//     
//    Arguments for constructor are as follows
//               name             mass          width         charge
//             2*spin           parity  C-conjugation
//          2*Isospin       2*Isospin3       G-parity
//               type    lepton number  baryon number   PDG encoding
//             stable         lifetime    decay table 

// In this version, charged pions are set to stable
G4BMesonMinus G4BMesonMinus::theBMesonMinus
(
	         "B-",      5.2789*GeV,   3.99e-10*MeV,    -1.*eplus, 
		    0,              -1,             0,          
		    1,              -1,             0,             
	      "meson",               0,             0,         -521,
		false,      1.62e-3*ns,          NULL
);

G4BMesonMinus*  G4BMesonMinus::BMesonMinusDefinition(){return &theBMesonMinus;}
G4BMesonMinus*  G4BMesonMinus::BMesonMinus(){return &theBMesonMinus;}
// initialization for static cut values
G4double   G4BMesonMinus::theBMesonMinusLengthCut = -1.0;
G4double*  G4BMesonMinus::theBMesonMinusKineticEnergyCuts = NULL;





