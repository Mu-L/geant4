// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4VIon.cc,v 1.2 1999/12/15 14:51:02 gunter Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
// 
// --------------------------------------------------------------
//      GEANT 4 class implementation file 
//
//      For information related to this code contact:
//      CERN, CN Division, ASD Group
//      History: first implementation, based on object model of
//      2nd December 1995, G.Cosmo
// --------------------------------------------------------------

#include "G4VIon.hh"

const G4VIon & G4VIon::operator=(const G4VIon &right)
{
  if (this != &right) {
  } return right;
}
