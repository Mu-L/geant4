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
//
//
// 
// John Allison  31st December 1997.
//
// Class Description:
//
// G4VModel is a base class for visualization models.  A model is a
// graphics-system-indepedent description of a Geant4 component.
// The key fuctionality of a model is to know how to describe itself
// to a scene handler.  A scene is a collection of models.
//
// The static data member const fpCurrentMP is provided so that it
// can be access statically: G4VModel::GetCurrentModelingParameters().
// Its value is expected to be identical to fpMP set in the scene
// handler, but to avoid linking problems, it must *not* be set on
// the vis side; it must be set in modeling by any model that
// requires it, G4TrajectoriesModel.

#ifndef G4VMODEL_HH
#define G4VMODEL_HH

#include "globals.hh"
#include "G4VisExtent.hh"

class G4VGraphicsScene;
class G4ModelingParameters;

class G4VModel {

public: // With description

  friend std::ostream& operator << (std::ostream& os, const G4VModel&);

  G4VModel(const G4ModelingParameters* = 0);
   
  virtual ~G4VModel ();

  virtual void DescribeYourselfTo (G4VGraphicsScene&) = 0;
  // The main task of a model is to describe itself to the graphics scene.

  const G4ModelingParameters* GetModelingParameters () const;
  static const G4ModelingParameters* GetCurrentModelingParameters ();
  // The latter for static access

  const G4String& GetType() const;
  // The sub-class should set its type, which could be the class
  // name, or, in the case of G4CallBackModel, it could be the type of
  // object it is coding.

  virtual G4String GetCurrentDescription () const;
  // A description which depends on the current state of the model.

  virtual G4String GetCurrentTag () const;
  // A tag which depends on the current state of the model.

  const G4VisExtent& GetExtent () const;
  // Extent of visible objects in local coordinate system.

  const G4String& GetGlobalDescription () const;
  // A description which does not change and lasts the life of the model.

  const G4String& GetGlobalTag () const;
  // A tag which does not change and lasts the life of the model.

  // Set methods for above...
  void SetModelingParameters (const G4ModelingParameters*);
  static void SetCurrentModelingParameters (const G4ModelingParameters*);
  void SetExtent (const G4VisExtent&);
  void SetType (const G4String&);
  void SetGlobalDescription (const G4String&);
  void SetGlobalTag (const G4String&);

  virtual G4bool Validate (G4bool warn = true);
  // Validate, but allow internal changes (hence non-const function).

protected:

  G4String                    fType;
  G4String                    fGlobalTag;
  G4String                    fGlobalDescription;
  G4VisExtent                 fExtent;
  const G4ModelingParameters* fpMP;

private:

  // Private copy constructor and assigment operator - copying and
  // assignment not allowed.  Keeps CodeWizard happy.
  G4VModel (const G4VModel&);
  G4VModel& operator = (const G4VModel&);

  static const G4ModelingParameters* fpCurrentMP;
};

#include "G4VModel.icc"

#endif
