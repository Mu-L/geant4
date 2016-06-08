// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4OpenGLStoredXm.hh,v 1.4 1999/12/15 14:54:05 gunter Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
// 
// Andrew Walkden  10th February 1997
// OpenGL graphics system factory.

#if defined (G4VIS_BUILD_OPENGLXM_DRIVER) || defined (G4VIS_USE_OPENGLXM)

#ifndef G4OPENGLSTOREDXM_HH
#define G4OPENGLSTOREDXM_HH

#include "G4VGraphicsSystem.hh"

class G4OpenGLStoredXm: public G4VGraphicsSystem {
public:
  G4OpenGLStoredXm ();
  G4VSceneHandler* CreateSceneHandler (const G4String& name = "");
  G4VViewer*  CreateViewer  (G4VSceneHandler&, const G4String& name = "");
};

#endif

#endif
