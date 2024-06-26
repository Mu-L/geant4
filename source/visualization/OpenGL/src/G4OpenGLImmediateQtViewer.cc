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
// Class G4OpenGLImmediateQtViewer : a class derived from G4OpenGLQtViewer and
//                                G4OpenGLImmediateViewer.

#include "G4OpenGLImmediateQtViewer.hh"
#include "G4OpenGLImmediateSceneHandler.hh"

#include "G4ios.hh"
#ifdef G4MULTITHREADED
#include "G4Threading.hh"
#endif
#include <qapplication.h>
#include <qtabwidget.h>


G4OpenGLImmediateQtViewer::G4OpenGLImmediateQtViewer
(G4OpenGLImmediateSceneHandler& sceneHandler,
 const G4String&  name):
  G4VViewer (sceneHandler, sceneHandler.IncrementViewCount (), name),
  G4OpenGLViewer (sceneHandler),
  G4OpenGLQtViewer (sceneHandler),
  G4OpenGLImmediateViewer (sceneHandler)
{
  fQGLWidgetInitialiseCompleted = false;

  setFocusPolicy(Qt::StrongFocus); // enable keybord events
  fHasToRepaint = false;
  fPaintEventLock = false;
  fUpdateGLLock = false;

  if (fViewId < 0) return;  // In case error in base class instantiation.
}

G4OpenGLImmediateQtViewer::~G4OpenGLImmediateQtViewer() {
  //  makeCurrent();  // Not sure why this - commented out 12-Apr-2021 JA
}

void G4OpenGLImmediateQtViewer::Initialise() {
#if QT_VERSION < 0x060000
  makeCurrent();
  
  fQGLWidgetInitialiseCompleted = false;
  CreateMainWindow (this,QString(GetName()));

  glDrawBuffer (GL_BACK);
  
  // set the good tab active
  if (G4QGLWidgetType::parentWidget()) {
    auto *parentTab = dynamic_cast<QTabWidget*> (G4QGLWidgetType::parentWidget()->parent()) ;
    if (parentTab) {
      parentTab->setCurrentIndex(parentTab->count()-1);
    }
  }
  
  fQGLWidgetInitialiseCompleted = true;
#else
  fQGLWidgetInitialiseCompleted = false;
  CreateMainWindow (this,QString(GetName()));
  if(G4QGLWidgetType::isValid()) { //G.Barrand: macOS: isValid() needed at startup.
    makeCurrent();
    glDrawBuffer (GL_BACK);
  }
  fQGLWidgetInitialiseCompleted = true;
#endif
}

void G4OpenGLImmediateQtViewer::initializeGL () {

  InitializeGLView ();

  // If a double buffer context has been forced upon us, ignore the
  // back buffer for this OpenGLImmediate view.
  //  glDrawBuffer (GL_FRONT); // FIXME : Ne marche pas avec cette ligne, mais affiche le run correctement...

  if (fSceneHandler.GetScene() == 0) {
    fHasToRepaint =false;
  } else {
    fHasToRepaint =true;
  }

  // Set the component visible
  
  // and update it immediatly before wait for SessionStart() (batch mode)
//  QCoreApplication::sendPostedEvents () ;

  // Set jpg as default export format for Qt viewer
  setExportImageFormat("jpg");
}


void  G4OpenGLImmediateQtViewer::DrawView() {
#if QT_VERSION < 0x060000
#else
  if(IsGettingPickInfos()) {
    paintGL();
    return;
  }
#endif
#ifdef G4MULTITHREADED
  if (G4Threading::G4GetThreadId() == G4Threading::MASTER_ID) {
    updateQWidget();
  }
#else
  updateQWidget();
#endif
}


void G4OpenGLImmediateQtViewer::ComputeView () {

  makeCurrent();
  // If a double buffer context has been forced upon us, ignore the
  // back buffer for this OpenGLImmediate view.
  //  glDrawBuffer (GL_FRONT);

  G4ViewParameters::DrawingStyle dstyle = GetViewParameters().GetDrawingStyle();

  if(dstyle!=G4ViewParameters::hlr &&
     haloing_enabled) {

    HaloingFirstPass ();
    NeedKernelVisit ();
    ProcessView ();
    FinishView();
    HaloingSecondPass ();

  }

  NeedKernelVisit ();  // Always need to visit G4 kernel.
  ProcessView ();

  if (isRecording()) {
    savePPMToTemp();
  }
   
  fHasToRepaint = true;
}

/**
   - Lors du resize de la fenetre, on doit non pas redessiner le detecteur, mais aussi les evenements
*/
void G4OpenGLImmediateQtViewer::resizeGL(
 int aWidth
,int aHeight)
{  
  if ((aWidth > 0) && (aHeight > 0)) {
#if QT_VERSION < 0x060000
    ResizeWindow(aWidth,aHeight);
#else
    ResizeWindow(devicePixelRatio()*aWidth,devicePixelRatio()*aHeight);
#endif
    fHasToRepaint = sizeHasChanged();
  }
}


void G4OpenGLImmediateQtViewer::paintGL()
{
#if QT_VERSION < 0x060000
  updateToolbarAndMouseContextMenu();
#else
  //G.Barrand: don't do any change in the GUI here, just "paint" this widget!
#endif

  if (fPaintEventLock) {
//    return ;
  }
  if (!fQGLWidgetInitialiseCompleted) {
    fPaintEventLock = false;
    return;
  }
  if ((getWinWidth() == 0) && (getWinHeight() == 0)) {
      return;
  }

#if QT_VERSION < 0x060000
  // DO NOT RESIZE IF SIZE HAS NOT CHANGE
  if ( !fHasToRepaint) {
    // L. Garnier : Trap to get the size with mac OSX 10.6 and Qt 4.6(devel)
    // Tested on Qt4.5 on mac, 4.4 on windows, 4.5 on unbuntu
    int sw = 0;
    int sh = 0;
    if (!isMaximized() && !isFullScreen()) {
      sw = normalGeometry().width();
      sh = normalGeometry().height();
    } else {
      sw = frameGeometry().width();
      sh = frameGeometry().height();
    }
    if ((getWinWidth() == (unsigned int)sw) &&(getWinHeight() == (unsigned int)sh)) {
      return;

    } else if ((sw == 0) && (sh == 0)) { // NOT A TOP LEVEL WIDGET
      if (((getWinWidth() == (unsigned int)width())) &&(getWinHeight() == (unsigned int) height())) { 
        return;
      }
    }
  }
#endif

#if QT_VERSION < 0x060000
#else
  InitializeGLView ();
#endif

  SetView();
   
  ClearView (); //ok, put the background correct
  ComputeView();

  fHasToRepaint = false; // could be set to false by ComputeView

  fPaintEventLock = false;
}

void G4OpenGLImmediateQtViewer::mousePressEvent(QMouseEvent *event)
{
  G4MousePressEvent(event);
}

void G4OpenGLImmediateQtViewer::keyPressEvent (QKeyEvent * event) 
{
  G4keyPressEvent(event);
}

void G4OpenGLImmediateQtViewer::keyReleaseEvent (QKeyEvent * event)
{
  G4keyReleaseEvent(event);
}

void G4OpenGLImmediateQtViewer::wheelEvent (QWheelEvent * event)
{
  G4wheelEvent(event);
}

void G4OpenGLImmediateQtViewer::showEvent (QShowEvent *) 
{
  if (fQGLWidgetInitialiseCompleted) {
    fHasToRepaint = true;
  }
}


/**
 * This function was build in order to make a zoom on double clic event.
 * It was think to build a rubberband on the zoom area, but never work fine
 */
void G4OpenGLImmediateQtViewer::mouseDoubleClickEvent(QMouseEvent *)
{
  G4MouseDoubleClickEvent();
}

void G4OpenGLImmediateQtViewer::mouseReleaseEvent(QMouseEvent *event)
{
  G4MouseReleaseEvent(event);
}

void G4OpenGLImmediateQtViewer::mouseMoveEvent(QMouseEvent *event)
{
  G4MouseMoveEvent(event);
}


void G4OpenGLImmediateQtViewer::contextMenuEvent(QContextMenuEvent *e)
{
  G4manageContextMenuEvent(e);
}

#if QT_VERSION < 0x060000
void G4OpenGLImmediateQtViewer::paintEvent(QPaintEvent *) {
  if (! fQGLWidgetInitialiseCompleted) {
    return;
  }
  // Force a repaint next time if the FRAMEBUFFER is not READY
  fHasToRepaint = isFramebufferReady();
  if ( fHasToRepaint) {
#if (QT_VERSION < QT_VERSION_CHECK(6, 0, 0))
    updateGL();
#else
    // Not sure this is correct....
    paintGL();
#endif
  }
}
#endif


void G4OpenGLImmediateQtViewer::updateQWidget() {
  if (fUpdateGLLock) {
    return;
  }
  
  if (! isCurrentWidget()){
    return;
  }
  
  fUpdateGLLock = true;
  fHasToRepaint= true;
#if QT_VERSION < 0x060000
  repaint();
  updateViewerPropertiesTableWidget();
  updateSceneTreeWidget();
#else
  //G.Barrand: don't do any change in the GUI here, just ask to "paint" this widget!
  update();
#endif
  fUpdateGLLock= false;
}


void G4OpenGLImmediateQtViewer::ShowView (
) 
//////////////////////////////////////////////////////////////////////////////
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
{
  fHasToRepaint = true;
  activateWindow();
}
