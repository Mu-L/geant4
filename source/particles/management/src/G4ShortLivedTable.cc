// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4ShortLivedTable.cc,v 1.9 2000/02/25 07:36:24 kurasige Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
// 
// --------------------------------------------------------------
//	GEANT 4 class implementation file 
//
//	For information related to this code contact:
//	CERN, IT Division, ASD Group
//	History: first implementation, based on object model of
//	27 June 1998  H.Kurashige
// ------------------------------------------------------------
//      added Remove()                  06 Nov.,98 H.Kurashige


#include "G4ShortLivedTable.hh"
#include "G4ParticleTable.hh"

#include "G4ios.hh"

#include "g4std/strstream"


G4ShortLivedTable::G4ShortLivedTable()
{
  fShortLivedList = new G4ShortLivedList();
}

G4ShortLivedTable::~G4ShortLivedTable()
{
  if (fShortLivedList ==0) return;
  // remove all contents in the short lived List and delete all particles  
  G4ShortLivedList::iterator i;
  for (i = fShortLivedList->begin(); i!= fShortLivedList->end(); ++i) {
    delete (*i);
  }
  fShortLivedList->clear();
  delete fShortLivedList;
  fShortLivedList =0;
}

G4int G4ShortLivedTable::GetVerboseLevel() const
{
  return G4ParticleTable::GetParticleTable()->GetVerboseLevel();
}

G4bool G4ShortLivedTable::IsShortLived(G4ParticleDefinition* particle) const
{
  return particle->IsShortLived();
}

void G4ShortLivedTable::Insert(G4ParticleDefinition* particle)
{
  if (IsShortLived(particle)) {
    fShortLivedList->push_back(particle);
  } 
}

void G4ShortLivedTable::Remove(G4ParticleDefinition* particle)
{
  if (IsShortLived(particle)) {
    G4ShortLivedList::iterator idx;
    for (idx = fShortLivedList->begin(); idx!= fShortLivedList->end(); ++idx) {
      if ( particle == *idx) {
        fShortLivedList->erase(idx);
      }
    }
  } else {
#ifdef G4VERBOSE
    if (GetVerboseLevel()>0) {
      G4cout << "G4ShortLivedTable::Remove :" << particle->GetParticleName() ;
      G4cout << " is not short lived" << G4endl; 
    }
#endif
  }
}


void G4ShortLivedTable::DumpTable(const G4String &particle_name) const
{
  G4ParticleDefinition* particle;

  G4ShortLivedList::iterator idx;
  for (idx = fShortLivedList->begin(); idx!= fShortLivedList->end(); ++idx) {
    particle = *idx;
    if (( particle_name == "ALL" ) || (particle_name == "all")){
      particle->DumpTable();
    } else if ( particle_name == particle->GetParticleName() ) {
      particle->DumpTable();
    }
  }
}

