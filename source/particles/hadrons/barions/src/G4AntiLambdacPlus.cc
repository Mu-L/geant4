// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4AntiLambdacPlus.cc,v 1.5 2000/02/27 06:17:02 kurasige Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
// 
// ----------------------------------------------------------------------
//      GEANT 4 class implementation file
//
//      For information related to this code contact:
//      CERN, CN Division, ASD Group
//      History: first implementation, based on object model of
//      4th April 1996, G.Cosmo
// **********************************************************************
//  Added particle definitions, H.Kurashige, 14 June 1997
//  Change both methods to get the pointer into non-inlined H.Kurashige 4 Aug. 1998
// ----------------------------------------------------------------------

#include "g4std/fstream"
#include "g4std/iomanip"

#include "G4AntiLambdacPlus.hh"

#include "G4PhaseSpaceDecayChannel.hh"
#include "G4DecayTable.hh"

// ######################################################################
// ###                           AntiLambdacPlus                           ###
// ######################################################################

G4AntiLambdacPlus::G4AntiLambdacPlus(
       const G4String&     aName,        G4double            mass,
       G4double            width,        G4double            charge,   
       G4int               iSpin,        G4int               iParity,    
       G4int               iConjugation, G4int               iIsospin,   
       G4int               iIsospin3,    G4int               gParity,
       const G4String&     pType,        G4int               lepton,      
       G4int               baryon,       G4int               encoding,
       G4bool              stable,       G4double            lifetime,
       G4DecayTable        *decaytable )
 : G4VBaryon( aName,mass,width,charge,iSpin,iParity,
              iConjugation,iIsospin,iIsospin3,gParity,pType,
              lepton,baryon,encoding,stable,lifetime,decaytable )
{
  SetParticleSubType("lambda_c");
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

G4AntiLambdacPlus G4AntiLambdacPlus::theAntiLambdacPlus(
     "anti_lambda_c+",      2.2849*GeV,   3.19e-9*MeV,   -1.*eplus,
		    1,              +1,             0,          
		    0,               0,             0,             
	     "baryon",               0,            +1,       -4122,
	        false,     0.206e-3*ns,          NULL
);

G4AntiLambdacPlus* G4AntiLambdacPlus::AntiLambdacPlusDefinition(){return &theAntiLambdacPlus;}
G4AntiLambdacPlus* G4AntiLambdacPlus::AntiLambdacPlus(){return &theAntiLambdacPlus;}
// initialization for static cut values
G4double   G4AntiLambdacPlus::theAntiLambdacPlusLengthCut = -1.0;
G4double*  G4AntiLambdacPlus::theAntiLambdacPlusKineticEnergyCuts = NULL;

