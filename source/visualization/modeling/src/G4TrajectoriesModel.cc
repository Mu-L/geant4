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
// John Allison  26th August 1998.
// Model which knows how to draw GEANT4 trajectories.

#include "G4TrajectoriesModel.hh"

#include "G4ModelingParameters.hh"
#include "G4VGraphicsScene.hh"
#include "G4RunManager.hh"
#include "G4RunManagerFactory.hh"
#include "G4Run.hh"
#include "G4Event.hh"
#include "G4AttDefStore.hh"
#include "G4AttValue.hh"
#include "G4AttDef.hh"
#include "G4AttCheck.hh"
#include "G4UIcommand.hh"
#include "G4VVisManager.hh"
#include "G4Text.hh"
#include "G4SystemOfUnits.hh"

#include <sstream>

G4TrajectoriesModel::G4TrajectoriesModel ()
:fpCurrentTrajectory(0)
,fRunID(-1)
,fEventID(-1)
{
  fType = "G4TrajectoriesModel";
  fGlobalTag = "G4TrajectoriesModel for any type of trajectory";
  fGlobalDescription = fGlobalTag;
}

G4TrajectoriesModel::~G4TrajectoriesModel () {}

void G4TrajectoriesModelDebugG4AttValues(const G4VTrajectory*);

void G4TrajectoriesModel::DescribeYourselfTo (G4VGraphicsScene& sceneHandler)
{
  G4RunManager* runManager = G4RunManagerFactory::GetMasterRunManager();
  const G4Run* currentRun  = runManager->GetCurrentRun();
  if(currentRun)
  {
    fRunID = currentRun->GetRunID();
  }
  else
  {
    fRunID = -1;
    return;
  }

  const G4Event* event = fpMP->GetEvent();
  if (event) {
    fEventID = event->GetEventID();
  } else {
    fEventID = -1;
    return;
  }

  G4VVisManager* pVVisManager = G4VVisManager::GetConcreteInstance();
  if (!pVVisManager) return;

  G4TrajectoryContainer* TC = event -> GetTrajectoryContainer ();
  if (!TC) return;

  // Need to set the static "current" modeling parameters separately
  G4VModel::SetCurrentModelingParameters(fpMP);

  for (std::size_t iT = 0; iT < TC->entries(); ++iT) {
    fpCurrentTrajectory = (*TC) [iT];
    // Debug trajectory:
    // fpCurrentTrajectory->ShowTrajectory(); G4cout << G4endl;
    // Debug G4AttValues:
    // G4TrajectoriesModelDebugG4AttValues(fpCurrentTrajectory);
    if (fpCurrentTrajectory)
      sceneHandler.AddCompound (*fpCurrentTrajectory);
  }

  // If time windowing is active, display head time if requested
  const auto& timeParameters = fpMP->GetTimeParameters();
  if (timeParameters.fDisplayHeadTime && timeParameters.fStartTime >= 0.) {
    const auto& tp = timeParameters;  // Alias
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << timeParameters.fEndTime/ns << " ns";
    G4Text headTime(oss.str());
    headTime.SetLayout(G4Text::centre);
    headTime.SetScreenSize(tp.fDisplayHeadTimeSize);
    headTime.SetPosition(G4Point3D(tp.fDisplayHeadTimeX, tp.fDisplayHeadTimeY, 0.));
    G4VisAttributes headTimeVisAtts
    (G4Colour(tp.fDisplayHeadTimeRed, tp.fDisplayHeadTimeGreen, tp.fDisplayHeadTimeBlue));
    headTime.SetVisAttributes(&headTimeVisAtts);
    pVVisManager->Draw2D(headTime);
  }
}

const std::map<G4String,G4AttDef>* G4TrajectoriesModel::GetAttDefs() const
{
  G4bool isNew;
  std::map<G4String,G4AttDef>* store
  = G4AttDefStore::GetInstance("G4TrajectoriesModel", isNew);
  if (isNew) {
    (*store)["RunID"] =
    G4AttDef("RunID","Run ID","Physics","","G4int");
    (*store)["EventID"] =
    G4AttDef("EventID","Event ID","Physics","","G4int");
  }
  return store;
}

std::vector<G4AttValue>* G4TrajectoriesModel::CreateCurrentAttValues() const
{
  std::vector<G4AttValue>* values = new std::vector<G4AttValue>;
  values->push_back
  (G4AttValue("RunID",G4UIcommand::ConvertToString(fRunID),""));
  values->push_back
  (G4AttValue("EventID",G4UIcommand::ConvertToString(fEventID),""));
  return values;
}

// Debug material...

#include "G4VTrajectoryPoint.hh"

void G4TrajectoriesModelDebugG4AttValues(const G4VTrajectory* pTraj)
{
  // Trajectory attributes
  {  // Scope bracket - allows re-use of names without compiler warnings.
    std::vector<G4AttValue>* attValues = pTraj->CreateAttValues();
    if (attValues) {
      G4AttCheck attCheck(attValues, pTraj->GetAttDefs());
      G4cout << "\nProvided G4Atts:\n" << attCheck;
      if (attCheck.Check()) G4cout << "Error" << G4endl;
      else {
	std::vector<G4AttValue> standardValues;
	std::map<G4String,G4AttDef> standardDefinitions;
	attCheck.Standard(&standardValues, &standardDefinitions);
	G4cout << "\nStandard G4Atts:\n"
	       << G4AttCheck(&standardValues, &standardDefinitions);
      }
      delete attValues;
    }
  }
  // Trajectory point attributes
  for (G4int i = 0; i < pTraj->GetPointEntries(); ++i) {
    G4VTrajectoryPoint* aPoint = pTraj->GetPoint(i);
    std::vector<G4AttValue>* attValues = aPoint->CreateAttValues();
    if (attValues) {
      G4AttCheck attCheck(attValues, aPoint->GetAttDefs());
      G4cout << "\nProvided G4Atts:\n" << attCheck;
      if (attCheck.Check()) G4cout << "Error" << G4endl;
      else {
	std::vector<G4AttValue> standardValues;
	std::map<G4String,G4AttDef> standardDefinitions;
	attCheck.Standard(&standardValues, &standardDefinitions);
	G4cout << "\nStandard G4Atts:\n"
	       << G4AttCheck(&standardValues, &standardDefinitions);
      }
      delete attValues;
    }
  }
}
