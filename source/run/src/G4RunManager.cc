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
// G4RunManager implementation
//
// Original author: M.Asai, 1996
// --------------------------------------------------------------------

#include "G4RunManager.hh"

#include "G4ApplicationState.hh"
#include "G4AssemblyStore.hh"
#include "G4CopyRandomState.hh"
#include "G4GeometryManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4MTRunManagerKernel.hh"
#include "G4Material.hh"
#include "G4ParallelWorldProcess.hh"
#include "G4ParallelWorldProcessStore.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessTable.hh"
#include "G4ProductionCutsTable.hh"
#include "G4RegionStore.hh"
#include "G4Run.hh"
#include "G4RunManagerKernel.hh"
#include "G4RunMessenger.hh"
#include "G4SDManager.hh"
#include "G4ScoringBox.hh"
#include "G4ScoringCylinder.hh"
#include "G4ScoringManager.hh"
#include "G4SmartVoxelHeader.hh"
#include "G4SmartVoxelStat.hh"
#include "G4SolidStore.hh"
#include "G4StateManager.hh"
#include "G4Timer.hh"
#include "G4TransportationManager.hh"
#include "G4FieldBuilder.hh"
#include "G4UImanager.hh"
#include "G4UnitsTable.hh"
#include "G4UserRunAction.hh"
#include "G4UserWorkerInitialization.hh"
#include "G4UserWorkerThreadInitialization.hh"
#include "G4VHitsCollection.hh"
#include "G4VPersistencyManager.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VScoreNtupleWriter.hh"
#include "G4VScoringMesh.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPhysicsList.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4VVisManager.hh"
#include "G4WorkerRunManagerKernel.hh"
#include "G4ios.hh"
#include "Randomize.hh"

#include <sstream>

using namespace CLHEP;

// The following lines are needed since G4VUserPhysicsList
// uses a #define theParticleIterator
#ifdef theParticleIterator
#  undef theParticleIterator
#endif

G4ThreadLocal G4RunManager* G4RunManager::fRunManager = nullptr;

G4bool G4RunManager::fGeometryHasBeenDestroyed = false;

// --------------------------------------------------------------------
G4bool G4RunManager::IfGeometryHasBeenDestroyed()
{
  return fGeometryHasBeenDestroyed;
}

// --------------------------------------------------------------------
G4RunManager* G4RunManager::GetRunManager()
{
  return fRunManager;
}

// --------------------------------------------------------------------
G4RunManager::G4RunManager()
{
  if (fRunManager != nullptr) {
    G4Exception("G4RunManager::G4RunManager()", "Run0031", FatalException,
                "G4RunManager constructed twice.");
  }
  fRunManager = this;

  kernel = new G4RunManagerKernel();
  eventManager = kernel->GetEventManager();

  timer = new G4Timer();
  runMessenger = new G4RunMessenger(this);
  previousEvents = new std::list<G4Event*>;
  G4ParticleTable::GetParticleTable()->CreateMessenger();
  G4ProcessTable::GetProcessTable();
  std::ostringstream oss;
  G4Random::saveFullState(oss);
  randomNumberStatusForThisRun = oss.str();
  randomNumberStatusForThisEvent = oss.str();
  runManagerType = sequentialRM;
  G4UImanager::GetUIpointer()->SetAlias("RunMode sequential");
}

// --------------------------------------------------------------------
G4RunManager::G4RunManager(RMType rmType)
{
  // This version of the constructor should never be called in sequential mode!
#ifndef G4MULTITHREADED
  G4ExceptionDescription msg;
  msg << "Geant4 code is compiled without multi-threading support "
         "(-DG4MULTITHREADED is set to off).";
  msg << " This type of RunManager can only be used in mult-threaded "
         "applications.";
  G4Exception("G4RunManager::G4RunManager(G4bool)", "Run0107", FatalException, msg);
#endif

  if (fRunManager != nullptr) {
    G4Exception("G4RunManager::G4RunManager()", "Run0031", FatalException,
                "G4RunManager constructed twice.");
    return;
  }
  fRunManager = this;

  switch (rmType) {
    case masterRM:
      kernel = new G4MTRunManagerKernel();
      break;
    case workerRM:
      kernel = new G4WorkerRunManagerKernel();
      break;
    default:
      G4ExceptionDescription msgx;
      msgx << "This type of RunManager can only be used in multi-threaded "
              "applications.";
      G4Exception("G4RunManager::G4RunManager(G4bool)", "Run0108", FatalException, msgx);
      return;
  }
  runManagerType = rmType;

  eventManager = kernel->GetEventManager();

  timer = new G4Timer();
  runMessenger = new G4RunMessenger(this);
  previousEvents = new std::list<G4Event*>;
  G4ParticleTable::GetParticleTable()->CreateMessenger();
  G4ProcessTable::GetProcessTable();
  randomNumberStatusDir = "./";
  std::ostringstream oss;
  G4Random::saveFullState(oss);
  randomNumberStatusForThisRun = oss.str();
  randomNumberStatusForThisEvent = oss.str();
}

// --------------------------------------------------------------------
G4RunManager::~G4RunManager()
{
  G4StateManager* pStateManager = G4StateManager::GetStateManager();
  // set the application state to the quite state
  if (pStateManager->GetCurrentState() != G4State_Quit) {
    if (verboseLevel > 1) G4cout << "G4 kernel has come to Quit state." << G4endl;
    pStateManager->SetNewState(G4State_Quit);
  }

  CleanUpPreviousEvents();
  if (verboseLevel > 1 && currentRun!=nullptr) {
    G4cout << "Deleting G4Run (id:" << currentRun->GetRunID() << ") ";
    if(currentRun->GetEventVectorSize()>0) {
      G4cout << " that has " << currentRun->GetEventVectorSize() 
             << " events kept in eventVector";
    }
    G4cout << G4endl;
  }
  delete currentRun;
  delete timer;
  delete runMessenger;
  delete previousEvents;

  // The following will work for all RunManager types
  // if derived class does the correct thing in the derived
  // destructor that is: set to zero pointers of
  // user initialization objects for which have no
  // ownership
  DeleteUserInitializations();
  delete userRunAction;
  userRunAction = nullptr;
  if (verboseLevel > 1) G4cout << "UserRunAction deleted." << G4endl;

  delete userPrimaryGeneratorAction;
  userPrimaryGeneratorAction = nullptr;
  if (verboseLevel > 1) G4cout << "UserPrimaryGenerator deleted." << G4endl;

  if (verboseLevel > 1) G4cout << "RunManager is deleting RunManagerKernel." << G4endl;

  delete kernel;

  fRunManager = nullptr;

  if (verboseLevel > 1) G4cout << "RunManager is deleted." << G4endl;
}

// --------------------------------------------------------------------
void G4RunManager::DeleteUserInitializations()
{
  if (verboseLevel > 1) G4cout << "UserDetectorConstruction deleted " << userDetector << G4endl;
  delete userDetector;
  userDetector = nullptr;

  if (verboseLevel > 1) G4cout << "UserPhysicsList deleted " << physicsList << G4endl;
  delete physicsList;
  physicsList = nullptr;

  if (verboseLevel > 1) G4cout << "UserActionInitialization deleted " << userActionInitialization << G4endl;
  delete userActionInitialization;
  userActionInitialization = nullptr;

  if (verboseLevel > 1) G4cout << "UserWorkerInitialization deleted " << userWorkerInitialization << G4endl;
  delete userWorkerInitialization;
  userWorkerInitialization = nullptr;

  if (verboseLevel > 1) G4cout << "UserWorkerThreadInitialization deleted " << userWorkerThreadInitialization << G4endl;
  delete userWorkerThreadInitialization;
  userWorkerThreadInitialization = nullptr;
}

// --------------------------------------------------------------------
void G4RunManager::BeamOn(G4int n_event, const char* macroFile, G4int n_select)
{
  fakeRun = n_event <= 0;
  G4bool cond = ConfirmBeamOnCondition();
  if (cond) {
    numberOfEventToBeProcessed = n_event;
    numberOfEventProcessed = 0;
    ConstructScoringWorlds();
    RunInitialization();
    DoEventLoop(n_event, macroFile, n_select);
    RunTermination();
  }
  fakeRun = false;
}

// --------------------------------------------------------------------
G4bool G4RunManager::ConfirmBeamOnCondition()
{
  G4StateManager* stateManager = G4StateManager::GetStateManager();

  G4ApplicationState currentState = stateManager->GetCurrentState();
  if (currentState != G4State_PreInit && currentState != G4State_Idle) {
    G4cerr << "Illegal application state - BeamOn() ignored." << G4endl;
    return false;
  }

  if (!initializedAtLeastOnce) {
    G4cerr << " Geant4 kernel should be initialized" << G4endl;
    G4cerr << "before the first BeamOn(). - BeamOn ignored." << G4endl;
    return false;
  }

  if (!geometryInitialized || !physicsInitialized) {
    if (verboseLevel > 0) {
      G4cout << "Start re-initialization because " << G4endl;
      if (!geometryInitialized) G4cout << "  Geometry" << G4endl;
      if (!physicsInitialized) G4cout << "  Physics processes" << G4endl;
      G4cout << "has been modified since last Run." << G4endl;
    }
    Initialize();
  }
  return true;
}

// --------------------------------------------------------------------
void G4RunManager::RunInitialization()
{
  if (!(kernel->RunInitialization(fakeRun))) return;

  runAborted = false;
  numberOfEventProcessed = 0;

  CleanUpPreviousEvents();
  if (verboseLevel > 2 && currentRun!=nullptr) {
    G4cout << "Deleting G4Run (id:" << currentRun->GetRunID() << ") ";
    if(currentRun->GetEventVectorSize()>0) {
      G4cout << " that has " << currentRun->GetEventVectorSize()
             << " events kept in eventVector";
    }
    G4cout << G4endl;
  }
  delete currentRun;
  currentRun = nullptr;

  if (fakeRun) return;

  if (fGeometryHasBeenDestroyed) G4ParallelWorldProcessStore::GetInstance()->UpdateWorlds();

  if (userRunAction != nullptr) currentRun = userRunAction->GenerateRun();
  if (currentRun == nullptr) currentRun = new G4Run();

  currentRun->SetRunID(runIDCounter);
  currentRun->SetNumberOfEventToBeProcessed(numberOfEventToBeProcessed);

  currentRun->SetDCtable(DCtable);
  G4SDManager* fSDM = G4SDManager::GetSDMpointerIfExist();
  if (fSDM != nullptr) {
    currentRun->SetHCtable(fSDM->GetHCtable());
  }

  if (G4VScoreNtupleWriter::Instance() != nullptr) {
    auto hce = (fSDM != nullptr) ? fSDM->PrepareNewEvent() : nullptr;
    isScoreNtupleWriter = G4VScoreNtupleWriter::Instance()->Book(hce);
    delete hce;
  }

  std::ostringstream oss;
  G4Random::saveFullState(oss);
  randomNumberStatusForThisRun = oss.str();
  currentRun->SetRandomNumberStatus(randomNumberStatusForThisRun);

  for (G4int i_prev = 0; i_prev < n_perviousEventsToBeStored; ++i_prev) {
    previousEvents->push_back(nullptr);
  }

  if (printModulo >= 0 || verboseLevel > 0) {
    G4cout << "### Run " << currentRun->GetRunID() << " starts." << G4endl;
  }
  if (userRunAction != nullptr) userRunAction->BeginOfRunAction(currentRun);

  if (isScoreNtupleWriter) {
    G4VScoreNtupleWriter::Instance()->OpenFile();
  }

  if (storeRandomNumberStatus) {
    G4String fileN = "currentRun";
    if (rngStatusEventsFlag) {
      std::ostringstream os;
      os << "run" << currentRun->GetRunID();
      fileN = os.str();
    }
    StoreRNGStatus(fileN);
  }
}

// --------------------------------------------------------------------
void G4RunManager::DoEventLoop(G4int n_event, const char* macroFile, G4int n_select)
{
  InitializeEventLoop(n_event, macroFile, n_select);

  // Event loop
  for (G4int i_event = 0; i_event < n_event; ++i_event) {
    ProcessOneEvent(i_event);
    TerminateOneEvent();
    if (runAborted) break;
  }

  // For G4MTRunManager, TerminateEventLoop() is invoked after all threads are
  // finished.
  if (runManagerType == sequentialRM) TerminateEventLoop();
}

// --------------------------------------------------------------------
void G4RunManager::InitializeEventLoop(G4int n_event, const char* macroFile, G4int n_select)
{
  if (verboseLevel > 0) {
    timer->Start();
  }

  n_select_msg = n_select;
  if (macroFile != nullptr) {
    if (n_select_msg < 0) n_select_msg = n_event;
    msgText = "/control/execute ";
    msgText += macroFile;
    selectMacro = macroFile;
  }
  else {
    n_select_msg = -1;
    selectMacro = "";
  }
}

// --------------------------------------------------------------------
void G4RunManager::ProcessOneEvent(G4int i_event)
{
  currentEvent = GenerateEvent(i_event);
  eventManager->ProcessOneEvent(currentEvent);
  AnalyzeEvent(currentEvent);
  UpdateScoring();
  if (i_event < n_select_msg) G4UImanager::GetUIpointer()->ApplyCommand(msgText);
}

// --------------------------------------------------------------------
void G4RunManager::TerminateOneEvent()
{
  StackPreviousEvent(currentEvent);
  currentEvent = nullptr;
  ++numberOfEventProcessed;
}

// --------------------------------------------------------------------
void G4RunManager::TerminateEventLoop()
{
  if (verboseLevel > 0 && !fakeRun) {
    timer->Stop();
    G4cout << " Run terminated." << G4endl;
    G4cout << "Run Summary" << G4endl;
    if (runAborted) {
      G4cout << "  Run Aborted after " << numberOfEventProcessed << " events processed." << G4endl;
    }
    else {
      G4cout << "  Number of events processed : " << numberOfEventProcessed << G4endl;
    }
    G4cout << "  " << *timer << G4endl;
  }
  fGeometryHasBeenDestroyed = false;
}

// --------------------------------------------------------------------
G4Event* G4RunManager::GenerateEvent(G4int i_event)
{
  if (userPrimaryGeneratorAction == nullptr) {
    G4Exception("G4RunManager::GenerateEvent()", "Run0032", FatalException,
                "G4VUserPrimaryGeneratorAction is not defined!");
    return nullptr;
  }

  auto anEvent = new G4Event(i_event);

  if (storeRandomNumberStatusToG4Event == 1 || storeRandomNumberStatusToG4Event == 3) {
    std::ostringstream oss;
    G4Random::saveFullState(oss);
    randomNumberStatusForThisEvent = oss.str();
    anEvent->SetRandomNumberStatus(randomNumberStatusForThisEvent);
  }

  if (storeRandomNumberStatus) {
    G4String fileN = "currentEvent";
    if (rngStatusEventsFlag) {
      std::ostringstream os;
      os << "run" << currentRun->GetRunID() << "evt" << anEvent->GetEventID();
      fileN = os.str();
    }
    StoreRNGStatus(fileN);
  }

  if (printModulo > 0 && anEvent->GetEventID() % printModulo == 0) {
    G4cout << "--> Event " << anEvent->GetEventID() << " starts." << G4endl;
  }
  userPrimaryGeneratorAction->GeneratePrimaries(anEvent);
  return anEvent;
}

// --------------------------------------------------------------------
void G4RunManager::StoreRNGStatus(const G4String& fnpref)
{
  G4String fileN = randomNumberStatusDir + fnpref + ".rndm";
  G4Random::saveEngineStatus(fileN);
}

// --------------------------------------------------------------------
void G4RunManager::AnalyzeEvent(G4Event* anEvent)
{
  G4VPersistencyManager* fPersM = G4VPersistencyManager::GetPersistencyManager();
  if (fPersM != nullptr) fPersM->Store(anEvent);
  currentRun->RecordEvent(anEvent);
}

// --------------------------------------------------------------------
void G4RunManager::RunTermination()
{
  if (!fakeRun) {
    CleanUpUnnecessaryEvents(0);
//    if(G4int(previousEvents->size())>0) {
//      G4ExceptionDescription ed;
//      ed << "Run still has " << previousEvents->size() << " unfinished events!!";
//      G4Exception("G4RunManager::RunTermination()","RM09009",FatalException,ed);
//    }
    // tasking occasionally will call this function even
    // if there was not a current run
    if (currentRun != nullptr) {
      if (userRunAction != nullptr) userRunAction->EndOfRunAction(currentRun);
      G4VPersistencyManager* fPersM = G4VPersistencyManager::GetPersistencyManager();
      if (fPersM != nullptr) fPersM->Store(currentRun);
      // write & close analysis output
      if (isScoreNtupleWriter) {
        G4VScoreNtupleWriter::Instance()->Write();
      }
    }
    ++runIDCounter;
  }

  kernel->RunTermination();
}

// --------------------------------------------------------------------
void G4RunManager::CleanUpPreviousEvents()
{
  // Delete all events carried over from previous run.
  // This method is invoked at the beginning of the next run
  // or from the destructor of G4RunManager at the very end of
  // the program.
  // N.B. If ToBeKept() is true, the pointer of this event is
  // kept in G4Run of the previous run, and deleted along with
  // the deletion of G4Run.

  auto evItr = previousEvents->cbegin();
  while (evItr != previousEvents->cend()) {
    G4Event* evt = *evItr;
    if (evt != nullptr && !(evt->ToBeKept())) {
      ReportEventDeletion(evt);
      delete evt;
    }
    evItr = previousEvents->erase(evItr);
  }

}

// --------------------------------------------------------------------
void G4RunManager::CleanUpUnnecessaryEvents(G4int keepNEvents)
{
  // Delete events that are no longer necessary for post
  // processing such as visualization.
  // N.B. If ToBeKept() is true, the pointer of this event is
  // kept in G4Run of the previous run, and deleted along with
  // the deletion of G4Run.

  auto evItr = previousEvents->cbegin();
  while (evItr != previousEvents->cend()) {
    if (G4int(previousEvents->size()) <= keepNEvents) return;

    G4Event* evt = *evItr;
    if (evt != nullptr) {
      if (evt->GetNumberOfGrips() == 0) {
        if (!(evt->ToBeKept())) {
          ReportEventDeletion(evt);
          delete evt;
        }
        evItr = previousEvents->erase(evItr);
      }
      else {
        ++evItr;
      }
    }
    else {
      evItr = previousEvents->erase(evItr);
    }
  }
}

// --------------------------------------------------------------------
void G4RunManager::StackPreviousEvent(G4Event* anEvent)
{
  if (anEvent->ToBeKept()) currentRun->StoreEvent(anEvent);

  if (n_perviousEventsToBeStored == 0) {
    if (anEvent->GetNumberOfGrips() == 0) {
      if (!(anEvent->ToBeKept())) {
        ReportEventDeletion(anEvent);
        delete anEvent;
      }
    }
    else {
      previousEvents->push_back(anEvent);
    }
  }
  else {
    previousEvents->push_back(anEvent);
  }

  CleanUpUnnecessaryEvents(n_perviousEventsToBeStored);
}

// --------------------------------------------------------------------
void G4RunManager::ReportEventDeletion(const G4Event* evt)
{
  if(verboseLevel > 3) {
    G4cout << "deleting G4Event(" << evt << ") eventID = " << evt->GetEventID()
           << " -- grips = " << evt->GetNumberOfGrips()
           << " keepFlag = " << evt->KeepTheEventFlag()
           << " subEvt = " << evt->GetNumberOfRemainingSubEvents();
    if(evt->GetNumberOfCompletedSubEvent()>0) {
      G4cout << " -- contains " << evt->GetNumberOfCompletedSubEvent() << " completed sub-events";
    }
    G4cout << G4endl;
  }
}
// --------------------------------------------------------------------
void G4RunManager::Initialize()
{
  G4StateManager* stateManager = G4StateManager::GetStateManager();
  G4ApplicationState currentState = stateManager->GetCurrentState();
  if (currentState != G4State_PreInit && currentState != G4State_Idle) {
    G4cerr << "Illegal application state - "
           << "G4RunManager::Initialize() ignored." << G4endl;
    return;
  }

  stateManager->SetNewState(G4State_Init);
  if (!geometryInitialized) InitializeGeometry();
  if (!physicsInitialized) InitializePhysics();
  initializedAtLeastOnce = true;
  if (stateManager->GetCurrentState() != G4State_Idle) {
    stateManager->SetNewState(G4State_Idle);
  }
}

// --------------------------------------------------------------------
void G4RunManager::InitializeGeometry()
{
  if (userDetector == nullptr) {
    G4Exception("G4RunManager::InitializeGeometry", "Run0033", FatalException,
                "G4VUserDetectorConstruction is not defined!");
    return;
  }

  if (verboseLevel > 1) G4cout << "userDetector->Construct() start." << G4endl;

  G4StateManager* stateManager = G4StateManager::GetStateManager();
  G4ApplicationState currentState = stateManager->GetCurrentState();
  if (currentState == G4State_PreInit || currentState == G4State_Idle) {
    stateManager->SetNewState(G4State_Init);
  }
  if (!geometryDirectlyUpdated) {
    kernel->DefineWorldVolume(userDetector->Construct(), false);
    userDetector->ConstructSDandField();
    nParallelWorlds = userDetector->ConstructParallelGeometries();
    userDetector->ConstructParallelSD();
    kernel->SetNumberOfParallelWorld(nParallelWorlds);
  }
  // Notify the VisManager as well
  if (G4Threading::IsMasterThread()) {
    G4VVisManager* pVVisManager = G4VVisManager::GetConcreteInstance();
    if (pVVisManager != nullptr) pVVisManager->GeometryHasChanged();
  }

  geometryDirectlyUpdated = false;
  geometryInitialized = true;
  stateManager->SetNewState(currentState);
}

// --------------------------------------------------------------------
void G4RunManager::InitializePhysics()
{
  G4StateManager* stateManager = G4StateManager::GetStateManager();
  G4ApplicationState currentState = stateManager->GetCurrentState();
  if (currentState == G4State_PreInit || currentState == G4State_Idle) {
    stateManager->SetNewState(G4State_Init);
  }
  if (physicsList != nullptr) {
    kernel->InitializePhysics();
  }
  else {
    G4Exception("G4RunManager::InitializePhysics()", "Run0034", FatalException,
                "G4VUserPhysicsList is not defined!");
  }
  physicsInitialized = true;
  stateManager->SetNewState(currentState);
}

// --------------------------------------------------------------------
void G4RunManager::AbortRun(G4bool softAbort)
{
  // This method is valid only for GeomClosed or EventProc state
  G4ApplicationState currentState = G4StateManager::GetStateManager()->GetCurrentState();
  if (currentState == G4State_GeomClosed || currentState == G4State_EventProc) {
    runAborted = true;
    if (currentState == G4State_EventProc && !softAbort) {
      currentEvent->SetEventAborted();
      eventManager->AbortCurrentEvent();
    }
  }
  else {
    G4cerr << "Run is not in progress. AbortRun() ignored." << G4endl;
  }
}

// --------------------------------------------------------------------
void G4RunManager::AbortEvent()
{
  // This method is valid only for EventProc state
  G4ApplicationState currentState = G4StateManager::GetStateManager()->GetCurrentState();
  if (currentState == G4State_EventProc) {
    currentEvent->SetEventAborted();
    eventManager->AbortCurrentEvent();
  }
  else {
    G4cerr << "Event is not in progress. AbortEevnt() ignored." << G4endl;
  }
}

// --------------------------------------------------------------------
void G4RunManager::DefineWorldVolume(G4VPhysicalVolume* worldVol, G4bool topologyIsChanged)
{
  kernel->DefineWorldVolume(worldVol, topologyIsChanged);
}

// --------------------------------------------------------------------
void G4RunManager::rndmSaveThisRun()
{
  G4int runNumber = 0;
  if (currentRun != nullptr) runNumber = currentRun->GetRunID();
  if (!storeRandomNumberStatus) {
    G4cerr << "Warning from G4RunManager::rndmSaveThisRun():"
           << " Random number status was not stored prior to this run." << G4endl
           << "/random/setSavingFlag command must be issued. "
           << "Command ignored." << G4endl;
    return;
  }

  G4fs::path fileIn = randomNumberStatusDir + "currentRun.rndm";

  std::ostringstream os;
  os << "run" << runNumber << ".rndm" << '\0';
  G4fs::path fileOut = randomNumberStatusDir + os.str();

  if (G4CopyRandomState(fileIn, fileOut, "G4RunManager::rndmSaveThisRun()") && verboseLevel > 0)
  {
    G4cout << fileIn << " is copied to " << fileOut << G4endl;
  }
}

// --------------------------------------------------------------------
void G4RunManager::rndmSaveThisEvent()
{
  if (currentEvent == nullptr) {
    G4cerr << "Warning from G4RunManager::rndmSaveThisEvent():"
           << " there is no currentEvent available." << G4endl << "Command ignored." << G4endl;
    return;
  }

  if (!storeRandomNumberStatus) {
    G4cerr << "Warning from G4RunManager::rndmSaveThisEvent():"
           << " Random number engine status is not available." << G4endl
           << "/random/setSavingFlag command must be issued "
           << "prior to the start of the run. Command ignored." << G4endl;
    return;
  }

  G4fs::path fileIn = randomNumberStatusDir + "currentEvent.rndm";

  std::ostringstream os;
  os << "run" << currentRun->GetRunID() << "evt" << currentEvent->GetEventID() << ".rndm" << '\0';
  G4fs::path fileOut = randomNumberStatusDir + os.str();

  if (G4CopyRandomState(fileIn, fileOut, "G4RunManager::rndmSaveThisEvent()") && verboseLevel > 0)
  {
    G4cout << fileIn << " is copied to " << fileOut << G4endl;
  } 
}

// --------------------------------------------------------------------
void G4RunManager::RestoreRandomNumberStatus(const G4String& fileN)
{
  G4String fileNameWithDirectory;
  if (fileN.find('/') == std::string::npos) {
    fileNameWithDirectory = randomNumberStatusDir + fileN;
  }
  else {
    fileNameWithDirectory = fileN;
  }

  G4Random::restoreEngineStatus(fileNameWithDirectory);
  if (verboseLevel > 0)
    G4cout << "RandomNumberEngineStatus restored from file: " << fileNameWithDirectory << G4endl;
  G4Random::showEngineStatus();
}

// --------------------------------------------------------------------
void G4RunManager::DumpRegion(const G4String& rname) const
{
  kernel->DumpRegion(rname);
}

// --------------------------------------------------------------------
void G4RunManager::DumpRegion(G4Region* region) const
{
  kernel->DumpRegion(region);
}

// --------------------------------------------------------------------
void G4RunManager::ConstructScoringWorlds()
{
  using MeshShape = G4VScoringMesh::MeshShape;

  G4ScoringManager* ScM = G4ScoringManager::GetScoringManagerIfExist();
  if (ScM == nullptr) return;

  auto nPar = (G4int)ScM->GetNumberOfMesh();
  if (nPar < 1) return;

  auto theParticleIterator = G4ParticleTable::GetParticleTable()->GetIterator();
  for (G4int iw = 0; iw < nPar; ++iw) {
    G4VScoringMesh* mesh = ScM->GetMesh(iw);
    if (fGeometryHasBeenDestroyed) mesh->GeometryHasBeenDestroyed();
    G4VPhysicalVolume* pWorld = nullptr;
    if (mesh->GetShape() != MeshShape::realWorldLogVol) {
      pWorld =
        G4TransportationManager::GetTransportationManager()->IsWorldExisting(ScM->GetWorldName(iw));
      if (pWorld == nullptr) {
        pWorld = G4TransportationManager::GetTransportationManager()->GetParallelWorld(
          ScM->GetWorldName(iw));
        pWorld->SetName(ScM->GetWorldName(iw));

        G4ParallelWorldProcess* theParallelWorldProcess = mesh->GetParallelWorldProcess();
        if (theParallelWorldProcess != nullptr) {
          theParallelWorldProcess->SetParallelWorld(ScM->GetWorldName(iw));
        }
        else {
          theParallelWorldProcess = new G4ParallelWorldProcess(ScM->GetWorldName(iw));
          mesh->SetParallelWorldProcess(theParallelWorldProcess);
          theParallelWorldProcess->SetParallelWorld(ScM->GetWorldName(iw));

          theParticleIterator->reset();
          while ((*theParticleIterator)()) {
            G4ParticleDefinition* particle = theParticleIterator->value();
            G4ProcessManager* pmanager = particle->GetProcessManager();
            if (pmanager != nullptr) {
              pmanager->AddProcess(theParallelWorldProcess);
              if (theParallelWorldProcess->IsAtRestRequired(particle)) {
                pmanager->SetProcessOrdering(theParallelWorldProcess, idxAtRest, 9900);
              }
              pmanager->SetProcessOrderingToSecond(theParallelWorldProcess, idxAlongStep);
              pmanager->SetProcessOrdering(theParallelWorldProcess, idxPostStep, 9900);
            }
          }
        }
        theParallelWorldProcess->SetLayeredMaterialFlag(mesh->LayeredMassFlg());
      }
    }
    mesh->Construct(pWorld);
  }
  GeometryHasBeenModified();
}

// --------------------------------------------------------------------
void G4RunManager::UpdateScoring(const G4Event* evt)
{
  if(evt==nullptr) evt = currentEvent;
  
//MAMAMAMA need revisiting
  if(evt->ScoresAlreadyRecorded()) return;
//MAMAMAMA need revisiting

  if (isScoreNtupleWriter) {
    G4VScoreNtupleWriter::Instance()->Fill(evt->GetHCofThisEvent(), evt->GetEventID());
  }

  if(evt->ScoresAlreadyRecorded()) {
    G4Exception("G4RunManager::UpdateScoring()","RMSubEvt001",FatalException,
        "Double-counting!!!");
  }

  evt->ScoresRecorded();

  G4ScoringManager* ScM = G4ScoringManager::GetScoringManagerIfExist();
  if (ScM == nullptr) return;
  auto nPar = (G4int)ScM->GetNumberOfMesh();
  if (nPar < 1) return;

  G4HCofThisEvent* HCE = evt->GetHCofThisEvent();
  if (HCE == nullptr) return;
  auto nColl = (G4int)HCE->GetCapacity();
  for (G4int i = 0; i < nColl; ++i) {
    G4VHitsCollection* HC = HCE->GetHC(i);
    if (HC != nullptr) ScM->Accumulate(HC);
  }
}

// --------------------------------------------------------------------
void G4RunManager::ReOptimizeMotherOf(G4VPhysicalVolume* pPhys)
{
  G4LogicalVolume* pMotherL = pPhys->GetMotherLogical();
  if (pMotherL != nullptr) ReOptimize(pMotherL);
}

// --------------------------------------------------------------------
void G4RunManager::ReOptimize(G4LogicalVolume* pLog)
{
  G4Timer localtimer;
  if (verboseLevel > 1) {
    localtimer.Start();
  }
  G4SmartVoxelHeader* header = pLog->GetVoxelHeader();
  delete header;
  header = new G4SmartVoxelHeader(pLog);
  pLog->SetVoxelHeader(header);
  if (verboseLevel > 1) {
    localtimer.Stop();
    G4SmartVoxelStat stat(pLog, header, localtimer.GetSystemElapsed(), localtimer.GetUserElapsed());
    G4cout << G4endl << "Voxelisation of logical volume <" << pLog->GetName() << ">" << G4endl;
    G4cout << " heads : " << stat.GetNumberHeads() << " - nodes : " << stat.GetNumberNodes()
           << " - pointers : " << stat.GetNumberPointers() << G4endl;
    G4cout << " Memory used : " << (stat.GetMemoryUse() + 512) / 1024
           << "k - total time : " << stat.GetTotalTime() << " - system time : " << stat.GetSysTime()
           << G4endl;
  }
}

// --------------------------------------------------------------------
void G4RunManager::SetUserInitialization(G4VUserDetectorConstruction* userInit)
{
  userDetector = userInit;
}

// --------------------------------------------------------------------
void G4RunManager::SetUserInitialization(G4VUserPhysicsList* userInit)
{
  physicsList = userInit;
  kernel->SetPhysics(userInit);
}

// --------------------------------------------------------------------
void G4RunManager::SetUserInitialization(G4UserWorkerInitialization* /*userInit*/)
{
  G4Exception("G4RunManager::SetUserInitialization()", "Run3001", FatalException,
              "Base-class G4RunManager cannot take G4UserWorkerInitialization. Use "
              "G4MTRunManager.");
}

// --------------------------------------------------------------------
void G4RunManager::SetUserInitialization(G4UserWorkerThreadInitialization* /*userInit*/)
{
  G4Exception("G4RunManager::SetUserThreadInitialization()", "Run3001", FatalException,
              "Base-class G4RunManager cannot take G4UserWorkerThreadInitialization. "
              "Use G4MTRunManager.");
}

// --------------------------------------------------------------------
void G4RunManager::SetUserInitialization(G4VUserActionInitialization* userInit)
{
  userActionInitialization = userInit;
  userActionInitialization->Build();
}

// --------------------------------------------------------------------
void G4RunManager::SetUserAction(G4UserRunAction* userAction)
{
  userRunAction = userAction;
}

// --------------------------------------------------------------------
void G4RunManager::SetUserAction(G4VUserPrimaryGeneratorAction* userAction)
{
  userPrimaryGeneratorAction = userAction;
}

// --------------------------------------------------------------------
void G4RunManager::SetUserAction(G4UserEventAction* userAction)
{
  eventManager->SetUserAction(userAction);
  userEventAction = userAction;
}

// --------------------------------------------------------------------
void G4RunManager::SetUserAction(G4UserStackingAction* userAction)
{
  eventManager->SetUserAction(userAction);
  userStackingAction = userAction;
}

// --------------------------------------------------------------------
void G4RunManager::SetUserAction(G4UserTrackingAction* userAction)
{
  eventManager->SetUserAction(userAction);
  userTrackingAction = userAction;
}

// --------------------------------------------------------------------
void G4RunManager::SetUserAction(G4UserSteppingAction* userAction)
{
  eventManager->SetUserAction(userAction);
  userSteppingAction = userAction;
}

// --------------------------------------------------------------------
void G4RunManager::GeometryHasBeenModified(G4bool prop)
{
  if (prop) {
    G4UImanager::GetUIpointer()->ApplyCommand("/run/geometryModified");
  }
  else {
    kernel->GeometryHasBeenModified();
  }
}

// --------------------------------------------------------------------
void G4RunManager::ReinitializeGeometry(G4bool destroyFirst, G4bool prop)
{
  if (destroyFirst && G4Threading::IsMasterThread()) {
    if (verboseLevel > 0) {
      G4cout << "#### Assembly, Volume, Solid, and Surface Stores are being cleaned." << G4endl;
    }
    G4GeometryManager::GetInstance()->OpenGeometry();
    G4AssemblyStore::GetInstance()->Clean();
    G4PhysicalVolumeStore::GetInstance()->Clean();
    G4LogicalVolumeStore::GetInstance()->Clean();
    G4SolidStore::GetInstance()->Clean();
    G4LogicalSkinSurface::CleanSurfaceTable();
    G4LogicalBorderSurface::CleanSurfaceTable();

    // remove all logical volume pointers from regions
    // exception: world logical volume pointer must be kept
    G4RegionStore* regionStore = G4RegionStore::GetInstance();
    for (const auto& rItr : *regionStore) {
      if (rItr->GetName() == "DefaultRegionForTheWorld") continue;
      auto lvItr = rItr->GetRootLogicalVolumeIterator();
      for (std::size_t iRLV = 0; iRLV < rItr->GetNumberOfRootVolumes(); ++iRLV) {
        rItr->RemoveRootLogicalVolume(*lvItr, false);
        ++lvItr;
      }
      if (verboseLevel > 0) {
        G4cout << "#### Region <" << rItr->GetName() << "> is cleared." << G4endl;
      }
    }

    // clear transportation manager
    fGeometryHasBeenDestroyed = true;
    G4TransportationManager::GetTransportationManager()->ClearParallelWorlds();
  }
  if (prop) {
    G4UImanager::GetUIpointer()->ApplyCommand("/run/reinitializeGeometry");
  }
  else {
    kernel->GeometryHasBeenModified();
    geometryInitialized = false;
    // Notify the VisManager as well
    if (G4Threading::IsMasterThread()) {
      G4VVisManager* pVVisManager = G4VVisManager::GetConcreteInstance();
      if (pVVisManager != nullptr) pVVisManager->GeometryHasChanged();
    }
    // Reinitialize field builder
    if (G4FieldBuilder::IsInstance()) {
      G4FieldBuilder::Instance()->Reinitialize();
    }
  }
}

// --------------------------------------------------------------------
