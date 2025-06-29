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
// John Allison  27th March 1996
// Abstract interface class for graphics views.

#include "G4VViewer.hh"

#include "G4PhysicalVolumeStore.hh"
#include "G4Scene.hh"
#include "G4Timer.hh"
#include "G4Transform3D.hh"
#include "G4UImanager.hh"
#include "G4UIsession.hh"
#include "G4VGraphicsSystem.hh"
#include "G4VInteractiveSession.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VSceneHandler.hh"
#include "G4VisManager.hh"
#include "G4ios.hh"

#include <sstream>

G4VViewer::G4VViewer(G4VSceneHandler& sceneHandler, G4int id, const G4String& name)
: fSceneHandler(sceneHandler)
, fViewId(id)
, fNeedKernelVisit(true)
, fTransientsNeedRedrawing(false)
{
  if (name == "") {
    std::ostringstream ost;
    ost << fSceneHandler.GetName() << '-' << fViewId;
    fName = ost.str();
  }
  else {
    fName = name;
  }
  fShortName = fName.substr(0, fName.find(' '));
  G4StrUtil::strip(fShortName);

  fVP = G4VisManager::GetInstance()->GetDefaultViewParameters();
  fDefaultVP = fVP;

  fSceneTree.SetType(G4SceneTreeItem::root);
  fSceneTree.SetDescription(fName);
}

G4VViewer::~G4VViewer()
{
  fSceneHandler.RemoveViewerFromList(this);
}

void G4VViewer::SetName(const G4String& name)
{
  fName = name;
  fShortName = fName.substr(0, fName.find(' '));
  G4StrUtil::strip(fShortName);
}

void G4VViewer::NeedKernelVisit()
{
  fNeedKernelVisit = true;

  // At one time I thought we'd better notify all viewers.  But I guess
  // each viewer can take care of itself, so the following code is
  // redundant (but keep it commented out for now).   (John Allison)
  // Notify all viewers that a kernel visit is required.
  // const G4ViewerList& viewerList = fSceneHandler.GetViewerList ();
  // G4ViewerListConstIterator i;
  // for (i = viewerList.begin(); i != viewerList.end(); i++) {
  //   (*i) -> SetNeedKernelVisit ();
  // }
  // ??...but, there's a problem in OpenGL Stored which seems to
  // require *all* viewers to revisit the kernel, so...
  //  const G4ViewerList& viewerList = fSceneHandler.GetViewerList ();
  //  G4ViewerListConstIterator i;
  //  for (i = viewerList.begin(); i != viewerList.end(); i++) {
  //    (*i) -> SetNeedKernelVisit (true);
  //  }
  // Feb 2005 - commented out.  Let's fix OpenGL if necessary.
}

void G4VViewer::FinishView() {}

void G4VViewer::ShowView() {}

void G4VViewer::ProcessView()
{
  // If the scene has changed, or if the concrete viewer has decided
  // that it necessary to visit the kernel, perhaps because the view
  // parameters have changed significantly (this should be done in the
  // concrete viewer's DrawView)...
  if (fNeedKernelVisit) {
    // Reset flag.  This must be done before ProcessScene to prevent
    // recursive calls when recomputing transients...
    G4Timer timer;
    timer.Start();
    fNeedKernelVisit = false;
    fSceneHandler.ClearStore();
    fSceneHandler.ProcessScene();
    UpdateGUISceneTree();
    timer.Stop();
    fKernelVisitElapsedTimeSeconds = timer.GetRealElapsed();
  }
}

void G4VViewer::ProcessTransients()
{
  // If transients change, e.g., a time window change...
  if (fTransientsNeedRedrawing) {
    // First, reset the flag - see comment above about recursive calls.
    fTransientsNeedRedrawing = false;
    fSceneHandler.ClearTransientStore();
    fSceneHandler.ProcessTransients();
  }
}

void G4VViewer::SetViewParameters(const G4ViewParameters& vp)
{
  fVP = vp;
}

void G4VViewer::SetTouchable(
  const std::vector<G4PhysicalVolumeModel::G4PhysicalVolumeNodeID>& fullPath)
{
  // Set the touchable for /vis/touchable/set/... commands.
  std::ostringstream oss;
  const auto& pvStore = G4PhysicalVolumeStore::GetInstance();
  for (const auto& pvNodeId : fullPath) {
    const auto& pv = pvNodeId.GetPhysicalVolume();
    auto iterator = find(pvStore->cbegin(), pvStore->cend(), pv);
    if (iterator == pvStore->cend()) {
      G4ExceptionDescription ed;
      ed << "Volume no longer in physical volume store.";
      G4Exception("G4VViewer::SetTouchable", "visman0401", JustWarning, ed);
    }
    else {
      oss << ' ' << pvNodeId.GetPhysicalVolume()->GetName() << ' ' << pvNodeId.GetCopyNo();
    }
  }
  G4UImanager::GetUIpointer()->ApplyCommand("/vis/set/touchable" + oss.str());
}

void G4VViewer::TouchableSetVisibility(
  const std::vector<G4PhysicalVolumeModel::G4PhysicalVolumeNodeID>& fullPath, G4bool visibiity)
{
  // Changes the Vis Attribute Modifiers and scene tree WITHOUT triggering a rebuild.

  // The following is equivalent to
  //  G4UImanager::GetUIpointer()->ApplyCommand("/vis/touchable/set/visibility ...");
  // (assuming the touchable has already been set), but avoids view rebuild.

  // Instantiate a working copy of a G4VisAttributes object...
  G4VisAttributes workingVisAtts;
  // and set the visibility.
  workingVisAtts.SetVisibility(visibiity);

  fVP.AddVisAttributesModifier(G4ModelingParameters::VisAttributesModifier(
    workingVisAtts, G4ModelingParameters::VASVisibility,
    G4PhysicalVolumeModel::GetPVNameCopyNoPath(fullPath)));
  // G4ModelingParameters::VASVisibility (VAS = Vis Attribute Signifier)
  // signifies that it is the visibility that should be picked out
  // and merged with the touchable's normal vis attributes.

  // Find scene tree item and set visibility
  // The scene tree works with strings
  G4String fullPathString = G4PhysicalVolumeModel::GetPVNamePathString(fullPath);
  std::list<G4SceneTreeItem>::iterator foundIter;
  if (fSceneTree.FindTouchableFromRoot(fullPathString, foundIter)) {
    foundIter->AccessVisAttributes().SetVisibility(visibiity);
    UpdateGUISceneTree();
  }
  else {
    G4ExceptionDescription ed;
    ed << "Touchable \"" << fullPath << "\" not found";
    G4Exception("G4VViewer::TouchableSetVisibility", "visman0402", JustWarning, ed);
  }
}

void G4VViewer::TouchableSetColour(
  const std::vector<G4PhysicalVolumeModel::G4PhysicalVolumeNodeID>& fullPath,
  const G4Colour& colour)
{
  // Changes the Vis Attribute Modifiers and scene tree WITHOUT triggering a rebuild.

  // The following is equivalent to
  //  G4UImanager::GetUIpointer()->ApplyCommand("/vis/touchable/set/colour ...");
  // (assuming the touchable has already been set), but avoids view rebuild.

  // Instantiate a working copy of a G4VisAttributes object...
  G4VisAttributes workingVisAtts;
  // and set the colour.
  workingVisAtts.SetColour(colour);

  fVP.AddVisAttributesModifier(G4ModelingParameters::VisAttributesModifier(
    workingVisAtts, G4ModelingParameters::VASColour,
    G4PhysicalVolumeModel::GetPVNameCopyNoPath(fullPath)));
  // G4ModelingParameters::VASColour (VAS = Vis Attribute Signifier)
  // signifies that it is the colour that should be picked out
  // and merged with the touchable's normal vis attributes.

  // Find scene tree item and set colour
  // The scene tree works with strings
  G4String fullPathString = G4PhysicalVolumeModel::GetPVNamePathString(fullPath);
  std::list<G4SceneTreeItem>::iterator foundIter;
  if (fSceneTree.FindTouchableFromRoot(fullPathString, foundIter)) {
    foundIter->AccessVisAttributes().SetColour(colour);
    UpdateGUISceneTree();
  }
  else {
    G4ExceptionDescription ed;
    ed << "Touchable \"" << fullPath << "\" not found";
    G4Exception("G4VViewer::TouchableSetColour", "visman0403", JustWarning, ed);
  }
}

void G4VViewer::UpdateGUISceneTree()
{
  G4UImanager* UI = G4UImanager::GetUIpointer();
  auto uiWindow = dynamic_cast<G4VInteractiveSession*>(UI->GetG4UIWindow());
  if (uiWindow) uiWindow->UpdateSceneTree(fSceneTree);
}

void G4VViewer::InsertModelInSceneTree(G4VModel* model)
{
  const auto& modelType = model->GetType();
  const auto& modelDescription = model->GetGlobalDescription();

  auto type = G4SceneTreeItem::model;
  auto pvModel = dynamic_cast<G4PhysicalVolumeModel*>(model);
  if (pvModel) type = G4SceneTreeItem::pvmodel;

  fCurtailDescent = false;  // This is used later in SceneTreeScene::ProcessVolume
  G4String furtherInfo;
  static G4bool firstWarning = true;
  G4bool warned = false;
  if (pvModel) {
    const auto& nAllTouchables = pvModel->GetTotalAllTouchables();
    if (nAllTouchables > fMaxAllTouchables) {
      std::ostringstream oss;
      oss << nAllTouchables << " touchables - too many for scene tree";
      furtherInfo = oss.str();
      if (firstWarning) {
        warned = true;
        if (G4VisManager::GetInstance()->GetVerbosity() >= G4VisManager::warnings) {
          G4ExceptionDescription ed;
          ed << pvModel->GetGlobalDescription() <<
          ":\n  Too many touchables (" << nAllTouchables
          << ") for scene tree. Scene tree for this model will be empty.";
          G4Exception("G4VViewer::InsertModelInSceneTree", "visman0404", JustWarning, ed);
        }
      }
      fCurtailDescent = true;  // This is used later in SceneTreeScene::ProcessVolume
    }
  }
  if (warned) firstWarning = false;

  // Find appropriate model
  auto& modelItems = fSceneTree.AccessChildren();
  auto modelIter = modelItems.begin();
  auto pvModelIter = modelItems.end();
  for (; modelIter != modelItems.end(); ++modelIter) {
    if (modelIter->GetType() == G4SceneTreeItem::pvmodel) {
      pvModelIter = modelIter;  // Last pre-existing PV model (if any)
    }
    if (modelIter->GetModelDescription() == modelDescription) break;
  }

  if (modelIter == modelItems.end()) {  // Model not seen before
    G4SceneTreeItem modelItem(type);
    modelItem.SetDescription("model");
    modelItem.SetModelType(modelType);
    modelItem.SetModelDescription(modelDescription);
    modelItem.SetFurtherInfo(furtherInfo);
    if (pvModelIter != modelItems.end() &&  // There was pre-existing PV Model...
        type == G4SceneTreeItem::pvmodel) {   // ...and the new model is also PV...
      fSceneTree.InsertChild(++pvModelIter, modelItem);  // ...insert after, else...
    } else {
      fSceneTree.InsertChild(modelIter, modelItem);  // ...insert at end
    }
  } else {  // Existing model - mark visible == active
    modelIter->AccessVisAttributes().SetVisibility(true);
  }
}

G4VViewer::SceneTreeScene::SceneTreeScene(G4VViewer* pViewer, G4PhysicalVolumeModel* pPVModel)
: fpViewer (pViewer)
, fpPVModel(pPVModel)
{
  if (fpPVModel == nullptr) {
    G4Exception("G4VViewer::SceneTreeScene::SceneTreeScene", "visman0405", FatalException,
                "G4PhysicalVolumeModel pointer is null");
    return;  // To keep Coverity happy
  }
  
  // Limit the expanded depth to limit the number expanded so as not to swamp the GUI
  G4int expanded = 0;
  for (const auto& dn : fpPVModel->GetMapOfDrawnTouchables()) {
    expanded += dn.second;
    if (fMaximumExpandedDepth < dn.first) fMaximumExpandedDepth = dn.first;
    if (expanded > fMaximumExpanded) break;
  }

  // Find appropriate model and its iterator
  const auto& modelID = fpPVModel->GetGlobalDescription();
  auto& modelItems = fpViewer->fSceneTree.AccessChildren();
  fModelIter = modelItems.begin();
  for (; fModelIter != modelItems.end(); ++fModelIter) {
    if (fModelIter->GetModelDescription() == modelID) break;
  }
  if (fModelIter == modelItems.end()) {
    G4Exception("G4VViewer::SceneTreeScene::SceneTreeScene", "visman0406", JustWarning,
                "Model not found");
  }
}

void G4VViewer::SceneTreeScene::ProcessVolume(const G4VSolid&)
{
  if (fpViewer->fCurtailDescent) {
    fpPVModel->CurtailDescent();
    return;
  }

  const auto& modelID = fpPVModel->GetGlobalDescription();

  std::ostringstream oss;
  oss << fpPVModel->GetFullPVPath();  // of this volume
  G4String fullPathString(oss.str());  // Has a leading space

  // Navigate scene tree and find or insert touchables one by one
  // Work down the path - "name id", then "name id name id", etc.
  const auto& nodeIDs = fpPVModel->GetFullPVPath();
  G4String partialPathString;
  auto currentIter = fModelIter;
  G4int depth = 0;
  for (const auto& nodeID : nodeIDs) {
    std::ostringstream oss1;
    oss1 << nodeID;
    partialPathString += ' ' + oss1.str();
    currentIter =
    FindOrInsertTouchable(modelID, *currentIter, ++depth, partialPathString, fullPathString);
  }
}

// clang-format off
std::list<G4SceneTreeItem>::iterator G4VViewer::SceneTreeScene::FindOrInsertTouchable
 (const G4String& modelID, G4SceneTreeItem& mother,
  G4int depth, const G4String& partialPathString, const G4String& fullPathString)
{
  auto& children = mother.AccessChildren();
  auto childIter = children.begin();
  for (; childIter != children.end(); ++childIter) {
    if (childIter->GetPVPath() == partialPathString) break;
  }
  if (childIter != children.end()) {

    // Item already exists

    if (childIter->GetType() == G4SceneTreeItem::ghost) {

      // Previously it was a ghost - but maybe this time it's real

      if (partialPathString == fullPathString) {
        // Partial path string refers to the actual volume so it's a touchable
        childIter->SetType(G4SceneTreeItem::touchable);
        // Populate with information
        childIter->SetDescription(fpPVModel->GetCurrentTag());
        childIter->SetModelType(fpPVModel->GetType());
        childIter->SetModelDescription(modelID);
        childIter->SetPVPath(partialPathString);
        if (fpVisAttributes) childIter->SetVisAttributes(*fpVisAttributes);
        childIter->SetAttDefs(fpPVModel->GetAttDefs());
        childIter->SetAttValues(fpPVModel->CreateCurrentAttValues());
      }  // Partial path string refers to an ancester - do nothing

    } else {

      // Already a pre-existing full touchable

      if (partialPathString == fullPathString) {
        // Partial path string refers to the actual volume
        // Replace vis attributes (if any) - they might have changed
        if (fpVisAttributes) childIter->SetVisAttributes(*fpVisAttributes);
      }  // Partial path string refers to an ancester - do nothing

    }

  } else {

    // Item does not yet exist

    if (partialPathString == fullPathString) {

      // Partial path string refers to the actual volume
      // Insert new touchable item
      G4SceneTreeItem touchable(G4SceneTreeItem::touchable);
      touchable.SetExpanded(depth > fMaximumExpandedDepth? false: true);
      touchable.SetDescription(fpPVModel->GetCurrentTag());
      touchable.SetModelType(fpPVModel->GetType());
      touchable.SetModelDescription(modelID);
      touchable.SetPVPath(partialPathString);
      if (fpVisAttributes) touchable.SetVisAttributes(*fpVisAttributes);
      touchable.SetAttDefs(fpPVModel->GetAttDefs());
      touchable.SetAttValues(fpPVModel->CreateCurrentAttValues());
      childIter = mother.InsertChild(childIter,touchable);

    } else {

      // Partial path string refers to an ancester - it's what we call a "ghost"
      G4SceneTreeItem ghost(G4SceneTreeItem::ghost);
      ghost.SetExpanded(depth > fMaximumExpandedDepth? false: true);
      // Create a tag from the partial path
      std::istringstream iss(partialPathString);
      G4String name, copyNo;
      while (iss >> name >> copyNo);
      std::ostringstream oss;
      oss << name << ':' << copyNo;
      ghost.SetDescription(oss.str());
      ghost.SetModelType(fpPVModel->GetType());
      ghost.SetModelDescription(modelID);
      ghost.SetPVPath(partialPathString);
      ghost.AccessVisAttributes().SetVisibility(false);
      childIter = mother.InsertChild(childIter,ghost);
    }
  }

  return childIter;
}
// clang-format on

std::ostream& operator<<(std::ostream& os, const G4VViewer& v)
{
  os << "View " << v.fName << ":\n";
  os << v.fVP;
  return os;
}
