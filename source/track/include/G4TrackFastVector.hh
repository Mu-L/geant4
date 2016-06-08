// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4TrackFastVector.hh,v 1.3 1999/11/07 16:32:04 kurasige Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
// 
// ------------------------------------------------------------
//	GEANT 4 class header file 
//
//	For information related to this code contact:
//	CERN, CN Division, ASD group
//	History: first implementation, based on object model of
//	8 Mar 1997, H.Kurashige
// ------------------------------------------------------------

#ifndef G4TrackFastVector_h
#define G4TrackFastVector_h 1

class G4Track;
#include "G4FastVector.hh"

const G4int G4TrackFastVectorSize = 512;
typedef G4FastVector<G4Track,G4TrackFastVectorSize> G4TrackFastVector;
//  Contains pointers to G4Track objects which are
//  generated by either primary or secondary interaction.

#endif

