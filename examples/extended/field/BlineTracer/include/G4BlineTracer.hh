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
/// \file field/BlineTracer/include/G4BlineTracer.hh
/// \brief Definition of the G4BlineTracer class
//
//
//
//
// --------------------------------------------------------------------
//
// G4BlineTracer
//
// Class description:
//
// Defines a tool to trace and visualise magnetic field lines
// To use this tool in a Geant4 application the user should
// create an instance of this class in the code as a run action.
// It will only work if a G4MagneticField field object is declared.

// --------------------------------------------------------------------
// Author: Laurent Desorgher (desorgher@phim.unibe.ch)
//         Created - 2003-10-06
// --------------------------------------------------------------------
#ifndef G4BlineTracer_h
#define G4BlineTracer_h 1

#include "CLHEP/Units/SystemOfUnits.h"

#include "G4Types.hh"
#include "G4UserRunAction.hh"

#include <vector>

class G4VUserPrimaryGeneratorAction;
class G4MagneticField;
class G4FieldManager;
class G4ChordFinder;

class G4BlineTracerMessenger;
class G4BlineSteppingAction;
class G4BlineEventAction;
class G4BlinePrimaryGeneratorAction;
class G4BlineEquation;

class G4BlineTracer : public G4UserRunAction
{
  public:  // with description
    G4BlineTracer();
    ~G4BlineTracer() override;

    void BeginOfRunAction(const G4Run* aRun) override;
    void EndOfRunAction(const G4Run* aRun) override;

    void ComputeBlines(G4int nlines);

    inline void SetMaxTrackingStep(G4double max_step) { fMaxTrackingStep = max_step; }
    inline G4BlineEventAction* GetEventAction() { return fEventAction; }

  private:
    void ResetChordFinders();

  private:
    G4BlineTracerMessenger* fMessenger = nullptr;
    G4BlineSteppingAction* fSteppingAction = nullptr;
    G4BlineEventAction* fEventAction = nullptr;
    G4BlinePrimaryGeneratorAction* fPrimaryGeneratorAction = nullptr;
    G4double fMaxTrackingStep = 1000. * CLHEP::m;
    G4bool fWas_ResetChordFinders_already_called = false;

    // G4VUserPrimaryGeneratorAction* fUserPrimaryAction;
    //  User defined primary generator action

    std::vector<G4ChordFinder*> fVecChordFinders;
    std::vector<G4FieldManager*> fVecFieldManagers;
    std::vector<G4MagneticField*> fVecMagneticFields;
    std::vector<G4BlineEquation*> fVecEquationOfMotion;
    // ChordFinders, detector fields, equation of motions, and field
    // manager for the different local and global magnetic fields.
};

#endif
