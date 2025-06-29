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
// G4OpenGLWin32Viewer : Class to provide WindowsNT specific
//                       functionality for OpenGL in GEANT4

#ifndef G4OPENGLWIN32VIEWER_HH
#define G4OPENGLWIN32VIEWER_HH

#include "globals.hh"

#include "G4OpenGLViewer.hh"

class G4OpenGLSceneHandler;

class G4OpenGLWin32Viewer: virtual public G4OpenGLViewer {

public:
  G4OpenGLWin32Viewer (G4OpenGLSceneHandler& scene);
  virtual ~G4OpenGLWin32Viewer ();
  void SetView ();
  void ShowView ();
  void SwitchToMasterThread();

protected:
  void GetWin32Connection ();
  void CreateGLWin32Context ();
  virtual void CreateMainWindow ();
  HDC fHDC;

  G4bool fMouseHovered;
  G4bool fMousePressed;
  G4int fMousePressedX, fMousePressedY;

private:
  static LRESULT CALLBACK WindowProc(HWND,UINT,WPARAM,LPARAM);
  static G4bool SetWindowPixelFormat(HDC);

  void TrackMouse(G4int, G4int);
  void ReleaseMouse();
  void SetShift(G4int, G4int);
  void SetRotation(G4int, G4int);
  void SetZoom(G4int);

  HWND fWindow;
  HGLRC fHGLRC;
  G4bool fInCreateWindow;
};

#endif
