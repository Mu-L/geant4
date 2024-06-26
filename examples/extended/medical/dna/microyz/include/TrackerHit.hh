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
// This example is provided by the Geant4-DNA collaboration
// Any report or published results obtained using the Geant4-DNA software
// shall cite the following Geant4-DNA collaboration publications:
// Med. Phys. 45 (2018) e722-e739
// Phys. Med. 31 (2015) 861-874
// Med. Phys. 37 (2010) 4692-4708
// Int. J. Model. Simul. Sci. Comput. 1 (2010) 157–178
// The Geant4-DNA web site is available at http://geant4-dna.org
//
/// \file TrackerHit.hh
/// \brief Definition of the TrackerHit class

#ifndef TrackerHit_h
#define TrackerHit_h 1

#include "G4THitsCollection.hh"
#include "G4ThreeVector.hh"
#include "G4VHit.hh"

class TrackerHit : public G4VHit
{
  public:
    TrackerHit();
    TrackerHit(const TrackerHit&);
    ~TrackerHit() override;

    // Operators
    const TrackerHit& operator=(const TrackerHit&);
    G4bool operator==(const TrackerHit&) const;

    inline void* operator new(size_t);
    inline void operator delete(void*);

    // Methods from base class
    void Draw() override;
    void Print() override;

    // Set methods
    inline void SetTrackID(const G4int& track) { fTrackID = track; };
    inline void SetEdep(const G4double& de) { fEdep = de; };
    inline void SetPos(const G4ThreeVector& xyz) { fPos = xyz; };
    inline void SetIncidentEnergy(const G4double& e) { fIncidentEnergy = e; };

    // Get methods
    inline G4int GetTrackID() const { return fTrackID; };
    inline G4double GetEdep() const { return fEdep; };
    inline G4ThreeVector GetPos() const { return fPos; };
    inline G4double GetIncidentEnergy() const { return fIncidentEnergy; };

  private:
    G4int fTrackID = -1;
    G4double fEdep = 0;
    G4ThreeVector fPos = G4ThreeVector();
    G4double fIncidentEnergy = 0.;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

using TrackerHitsCollection = G4THitsCollection<TrackerHit>;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

extern G4ThreadLocal G4Allocator<TrackerHit>* TrackerHitAllocator;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline void* TrackerHit::operator new(size_t)
{
  if (!TrackerHitAllocator) {
    TrackerHitAllocator = new G4Allocator<TrackerHit>;
  }
  return (void*)TrackerHitAllocator->MallocSingle();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline void TrackerHit::operator delete(void* hit)
{
  TrackerHitAllocator->FreeSingle((TrackerHit*)hit);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
