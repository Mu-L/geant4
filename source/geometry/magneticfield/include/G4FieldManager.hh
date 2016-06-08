// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4FieldManager.hh,v 1.3 2000/04/27 09:14:05 gcosmo Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
//  
// class G4FieldManager
//
// Class description:
//
// A class to manage (Store) a pointer to the Field subclass that
// describes the field of a detector (magnetic, electric or other).
// Also stores a reference to the chord finder.
//
// The G4FieldManager class exists to allow the user program to specify 
// the electric, magnetic and/or other field(s) of the detector.
// (OR, in the future, of a part of it - planned to be a logical volume).
// It also stores a pointer to the ChordFinder object that can do the
// propagation in this field. All geometrical track "advancement" 
// in the field is handled by this ChordFinder object.
//
// G4FieldManager allows the other classes/object (of the MagneticField 
// & other class categories) to find out whether a detector field object 
// exists and what that object is.
//
// The Chord Finder must be created either by calling CreateChordFinder
// for a Magnetic Field or by the user creating a  a Chord Finder object
// "manually" and setting this pointer.
//
// A default FieldManager is created by the singleton class
// G4NavigatorForTracking and exists before main is called.
// However a new one can be created and given to G4NavigatorForTracking.

// History:
// - 10.03.97 John Apostolakis, design and implementation.

#ifndef G4FIELDMANAGER_HH
#define G4FIELDMANAGER_HH 1

#include "G4Field.hh"
#include "G4MagneticField.hh"
#include "G4ChordFinder.hh"

class G4FieldManager
{
  public:  // with description

     G4FieldManager();
     G4FieldManager(G4MagneticField *detectorField);
    ~G4FieldManager();

     G4bool          SetDetectorField(G4Field *detectorField);
     G4Field*        GetDetectorField();
     G4bool          DoesFieldExist();

     void            CreateChordFinder(G4MagneticField *detectorMagField);
     void            SetChordFinder(G4ChordFinder *aChordFinder);
     G4ChordFinder*  GetChordFinder();

  private:

     G4Field*        fDetectorField;
     G4ChordFinder*  fChordFinder;

     G4bool          fAllocatedChordFinder; // Did we used "new" to
					    // create fChordFinder ?
};

// Our current design envisions that one Field manager is 
// valid for a detector.
// (eg a detector with electric E and magnetic B field will now treat
// them as one field - and could treat any other field of importance 
// as additional components of a single field.)
// Does it make sense to have several instead ?
// Is the lack of elegance of the design (of G4Field) made up 
// for by the simplification it allows ?

// Implementation of inline functions

#include "G4FieldManager.icc"

#endif   /*  G4FIELDMANAGER_HH */
