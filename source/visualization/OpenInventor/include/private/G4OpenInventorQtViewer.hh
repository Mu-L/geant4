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

// Frederick Jones TRIUMF 07 November 2017
// Open Inventor viewer using SoQt.

#ifndef G4OPENINVENTORQTVIEWER_HH
#define G4OPENINVENTORQTVIEWER_HH

// Inheritance :
#include "G4OpenInventorViewer.hh"

#include <Inventor/nodes/SoEventCallback.h>

class G4OpenInventorQtExaminerViewer;

class SoQtViewer;

#include <qobject.h>

class QMenuBar;
class QFont;
class QAction;

#if 0x060000 <= QT_VERSION
class G4OpenInventorQtDestroyCallback;
#endif

class G4OpenInventorQtViewer: public QObject,
                              public G4OpenInventorViewer {

  Q_OBJECT 

private Q_SLOTS :

   // File menu
   void FileWritePSCB();
   void FileWritePDFCB();
   void FileWriteIVCB();
   void FileEscapeCB();

   // Etc menu
   void EtcEraseDetectorCB();
   void EtcEraseEventCB();
   void EtcSetSolidCB();
   void EtcSetReducedWireframeCB();
   void EtcSetFullWireframeCB();
   void EtcVisibMInvisibDCB();
   void EtcVisibMVisibDCB();
   void EtcUpdateSceneCB();
   void EtcSceneGraphStatsCB();

private:

   // File menu
   QAction* FileWritePS{nullptr};
   QAction* FileWritePDF{nullptr};
   QAction* FileWriteIV{nullptr};
   QAction* FileEscape{nullptr};

   // Etc menu
   QAction* EtcEraseDetector{nullptr};
   QAction* EtcEraseEvent{nullptr};
   QAction* EtcSetSolid{nullptr};
   QAction* EtcSetReducedWireframe{nullptr};
   QAction* EtcSetFullWireframe{nullptr};
   QAction* EtcVisibMInvisibD{nullptr};
   QAction* EtcVisibMVisibD{nullptr};
   QAction* EtcUpdateScene{nullptr};
   QAction* EtcSceneGraphStats{nullptr};

public:

   G4OpenInventorQtViewer(G4OpenInventorSceneHandler& scene,
		         const G4String& name = "");
   virtual ~G4OpenInventorQtViewer();
   void Initialise();

public: //G4VViewer

   virtual void FinishView();
   virtual void SetView();

protected:

   virtual void ViewerRender();
   virtual SoCamera* GetCamera();
   // FWJ found insufficient to replace group and camera sensors
   // static void FinishCB(void*, SoQtViewer*);

protected:

   //SoQtExaminerViewer* fViewer;
  G4OpenInventorQtExaminerViewer* fViewer;

#if 0x060000 <= QT_VERSION
  G4OpenInventorQtDestroyCallback* fDestroyCallback;
#endif
};

//G.Barrand: I don't put the below class within "#if 0x060000 <= QT_VERSION"
//           since Qt6/moc does not handle it correctly! (With it, Qt6/moc
//           will anyway NOT handle this class. It looks to be a known Qt problem).
//#if 0x060000 <= QT_VERSION
class G4OpenInventorQtDestroyCallback : public QObject {
  Q_OBJECT
public:
  G4OpenInventorQtDestroyCallback(G4OpenInventorQtViewer* a_viewer):fViewer(a_viewer) {}
  virtual ~G4OpenInventorQtDestroyCallback() = default;
public slots:
  void execute(){
    delete fViewer;
  }
private:
  G4OpenInventorQtViewer* fViewer;
};
//#endif

#endif
