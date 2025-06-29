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
//

#include "G4RayTracerQt.hh"
#include "G4RayTracerFeatures.hh"
#include "G4RayTracerSceneHandler.hh"
#include "G4RayTracerQtViewer.hh"

#define G4warn G4cerr

G4RayTracerQt::G4RayTracerQt():
  G4VGraphicsSystem("RayTracerQt",
		    "RTQt",
		    RAYTRACER_FEATURES,
		    G4VGraphicsSystem::threeD)
{}

G4RayTracerQt::~G4RayTracerQt()
{}

G4VSceneHandler* G4RayTracerQt::CreateSceneHandler (const G4String& name) {
  G4VSceneHandler* pSceneHandler = new G4RayTracerSceneHandler (*this, name);
  return pSceneHandler;
}

G4VViewer* G4RayTracerQt::CreateViewer (G4VSceneHandler& sceneHandler,
				       const G4String& name) {
  G4VViewer* pViewer = new G4RayTracerQtViewer (sceneHandler, name);
  if (pViewer) {
    if (pViewer->GetViewId() < 0) {
      G4warn <<
        "G4RayTracerQt::CreateViewer: ERROR flagged by negative"
        " view id in G4RayTracerQtViewer creation."
        "\n Destroying view and returning null pointer."
             << G4endl;
      delete pViewer;
      pViewer = 0;
    }
  }
  else {
    G4warn <<
      "G4RayTracerQt::CreateViewer: ERROR: null pointer on new G4RayTracerQtViewer."
           << G4endl;
  }
  return pViewer;
}
