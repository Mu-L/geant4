// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G3VolTable.hh,v 1.14 2000/03/02 17:54:03 gcosmo Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
// modified by I.Hrivnacova, 13.10.99

#ifndef G3VOLTABLE_HH
#define G3VOLTABLE_HH 1

#include "g4std/map"
#include "G3VolTableEntry.hh"

class G4LogicalVolume;
class G4Material;
class G4VSolid;

class G3VolTable{
private:
  G3VolTableEntry* G3toG4TopVTE;
  G4String _FirstKey;
  G4std::map<G4String, G3VolTableEntry*, G4std::less<G4String> > VTD;
  G4int _NG3Pos;

public:
  G3VolTableEntry* PutVTE(G3VolTableEntry* aVTE);  
  G3VolTableEntry* GetVTE(const G4String& Vname);
  void PrintAll();
  G3VolTable();
  virtual ~G3VolTable();
  G4LogicalVolume* GetG3toG4Mother();
  G3VolTableEntry* GetFirstVTE();
  void SetFirstVTE();
  void VTEStat();
  void CountG3Pos();
  void Clear();
};
extern G3VolTable G3Vol;
#endif
