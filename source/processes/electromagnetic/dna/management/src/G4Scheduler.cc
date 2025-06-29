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
// Author: Mathieu Karamitros (kara (AT) cenbg . in2p3 . fr)
//
// History:
// -----------
// 10 Oct 2011 M.Karamitros created
//
// -------------------------------------------------------------------

#include "G4ITGun.hh"
#include "G4ITModelHandler.hh"
#include "G4ITModelProcessor.hh"
#include "G4ITStepProcessor.hh"
#include "G4ITStepStatus.hh"
#include "G4ITTrackingInteractivity.hh"
#include "G4ITTrackingManager.hh"
#include "G4IosFlagsSaver.hh"
#include "G4StateManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Timer.hh"
#include "G4TrackingInformation.hh"
#include "G4UnitsTable.hh"
#include "G4UserTimeStepAction.hh"
#include "G4VITStepModel.hh"

#include <G4AllITFinder.hh>
#include <G4Scheduler.hh>
#include <G4SchedulerMessenger.hh>

#include <sstream>

#define DEBUG_MEM_STEPPING
#define DEBUG_MEM_DETAILED_STEPPING

#ifdef DEBUG_MEM
#  include "G4MemStat.hh"
using namespace G4MemStat;
using G4MemStat::MemStat;
#endif

#ifdef USE_COLOR
#  define RED "\033[0;31m"
#  define LIGHT_RED "\33[1;31m"
#  define GREEN "\033[32;40m"
#  define GREEN_ON_BLUE "\033[1;32;44m"
#  define RESET_COLOR "\033[0m"
#else
#  define RED ""
#  define LIGHT_RED ""
#  define GREEN ""
#  define GREEN_ON_BLUE ""
#  define RESET_COLOR ""
#endif

using namespace std;

G4ThreadLocal G4Scheduler* G4Scheduler::fgScheduler(nullptr);

template<typename T>
inline G4bool IsInf(T value)
{
  return std::numeric_limits<T>::has_infinity && value == std::numeric_limits<T>::infinity();
}
//_________________________________________________________________________

G4Scheduler* G4Scheduler::Instance()
{
  if (fgScheduler == nullptr) fgScheduler = new G4Scheduler();
  return fgScheduler;
}
//_________________________________________________________________________

G4bool G4Scheduler::Notify(G4ApplicationState requestedState)
{
  if (requestedState == G4State_Quit) {
    if (fVerbose >= 4) {
      G4cout << "G4Scheduler received G4State_Quit" << G4endl;
    }
    Clear();
  }
  return true;
}
//_________________________________________________________________________

void G4Scheduler::DeleteInstance()
{
  delete fgScheduler;
}
//_________________________________________________________________________

G4Scheduler::G4Scheduler() : fTrackContainer((G4ITTrackHolder&)*G4ITTrackHolder::Instance())
{
  Create();
}

void G4Scheduler::Create()
{
  fUseDefaultTimeSteps = true;
  fUserUpperTimeLimit = -1;
  fpGun = nullptr;
  fContinue = true;
  fpTrackingInteractivity = nullptr;
  fITStepStatus = eUndefined;
  fpUserTimeSteps = nullptr;
  fTimeStep = DBL_MAX;
  fTSTimeStep = DBL_MAX;
  fILTimeStep = DBL_MAX;
  fPreviousTimeStep = DBL_MAX;
  fZeroTimeCount = 0;
  fMaxNZeroTimeStepsAllowed = 10000;
  fStartTime = 0;
  fTimeTolerance = 1 * picosecond;
  fEndTime = 1 * microsecond;
  fGlobalTime = -1;
  fInteractionStep = true;
  fUsePreDefinedTimeSteps = false;

  fDefaultMinTimeStep = 1 * picosecond;
  fpStepProcessor = nullptr;
  fpModelProcessor = nullptr;
  fNbSteps = 0;
  fMaxSteps = -1;
  fRunning = false;
  fInitialized = false;
  fpUserTimeStepAction = nullptr;
  fpModelHandler = new G4ITModelHandler();
  fpTrackingManager = new G4ITTrackingManager();
  fVerbose = 0;
  fWhyDoYouStop = false;
  fDefinedMinTimeStep = -1.;
  fReachedUserTimeLimit = false;
  fStopTime = -1.;
  fTmpGlobalTime = -1.;
  fpMessenger = new G4SchedulerMessenger(this);
  fReactionSet = G4ITReactionSet::Instance();
  fMaxTimeStep = DBL_MAX;
  // hoang add
  fResetScavenger = true;  // true by default
  G4ITTypeManager::Instance()->ReserveRessource();
}

//_________________________________________________________________________

G4Scheduler::~G4Scheduler()
{
  if (fpMessenger != nullptr)  // is used as a flag to know whether the manager was cleared
  {
    Clear();
  }
  fgScheduler = nullptr;
}

void G4Scheduler::Clear()
{
  if (fpMessenger != nullptr) {
    delete fpMessenger;
    fpMessenger = nullptr;
  }
  if (fpStepProcessor != nullptr) {
    delete fpStepProcessor;
    fpStepProcessor = nullptr;
  }
  if (fpModelProcessor != nullptr) {
    delete fpModelProcessor;
    fpModelProcessor = nullptr;
  }

  G4ITTypeManager::Instance()->ReleaseRessource();
  ClearList();
  if (fpTrackingManager != nullptr) {
    delete fpTrackingManager;
    fpTrackingManager = nullptr;
  }

  if (fReactionSet != nullptr) {
    delete fReactionSet;
    fReactionSet = nullptr;
  }

  if (fpModelHandler != nullptr) {
    delete fpModelHandler;
    fpModelHandler = nullptr;
  }
}

//_________________________________________________________________________

void G4Scheduler::ClearList()
{
  fTrackContainer.Clear();
  G4AllITFinder::DeleteInstance();
}

//_________________________________________________________________________
void G4Scheduler::RegisterModel(G4VITStepModel* model, G4double time)
{
  fpModelHandler->RegisterModel(model, time);
}

//_________________________________________________________________________

void G4Scheduler::Initialize()
{
  delete fpStepProcessor;
  delete fpModelProcessor;

  fpModelProcessor = new G4ITModelProcessor();
  fpModelProcessor->SetModelHandler(fpModelHandler);
  fpModelProcessor->SetTrackingManager(fpTrackingManager);
  fpStepProcessor = new G4ITStepProcessor();
  fpStepProcessor->SetTrackingManager(fpTrackingManager);
  fpTrackingManager->SetInteractivity(fpTrackingInteractivity);
  if (fUsePreDefinedTimeSteps) {
    if (fpUserTimeSteps == nullptr)  // Extra checking, is it necessary ?
    {
      G4ExceptionDescription exceptionDescription;
      exceptionDescription << "You are asking to use user defined steps but you did not give any.";
      G4Exception("G4Scheduler::FindUserPreDefinedTimeStep", "Scheduler004", FatalErrorInArgument,
                  exceptionDescription);
      return;  // makes coverity happy
    }
  }

  fInitialized = true;
}

//_________________________________________________________________________

void G4Scheduler::Reset()
{
  fStartTime = 0;
  fUserUpperTimeLimit = -1;
  fTimeStep = DBL_MAX;
  fTSTimeStep = DBL_MAX;
  fILTimeStep = DBL_MAX;
  fPreviousTimeStep = DBL_MAX;
  fGlobalTime = -1;
  fInteractionStep = true;
  fITStepStatus = eUndefined;
  fZeroTimeCount = 0;

  fNbSteps = 0;
  fContinue = true;
  fReactionSet->CleanAllReaction();
}
//_________________________________________________________________________

void G4Scheduler::Process()
{
#ifdef G4VERBOSE
  if (fVerbose != 0) {
    G4cout << "*** G4Scheduler starts processing " << G4endl;
    if (fVerbose > 2)
      G4cout << "___________________________________________"
                "___________________________"
             << G4endl;
  }
#endif

  if (!fInitialized) {
    Initialize();
  }
  fpModelProcessor->Initialize();
  fpStepProcessor->Initialize();

  if (fpGun != nullptr) fpGun->DefineTracks();

  if (fpTrackingInteractivity != nullptr) fpTrackingInteractivity->Initialize();

  // ___________________
  fRunning = true;
  Reset();

  if (fResetScavenger) {
    if (fpUserScavenger != nullptr) {
      fpUserScavenger->Reset();
    }
  }

  if (fpUserTimeStepAction != nullptr) {
    fpUserTimeStepAction->StartProcessing();
  }

#ifdef G4VERBOSE
  G4bool trackFound = false;
  G4IosFlagsSaver iosfs(G4cout);
  G4cout.precision(5);
#endif

  //===========================================================================
  // By default, before the G4Scheduler is launched, the tracks are pushed to
  // the delayed lists
  //===========================================================================

  if (fTrackContainer.DelayListsNOTEmpty()) {
    fStartTime = fTrackContainer.GetNextTime();
#ifdef G4VERBOSE
    trackFound = true;
    G4Timer localtimer;
    if (fVerbose > 1) {
      localtimer.Start();
    }
#endif
    SynchronizeTracks();
#ifdef G4VERBOSE
    if (fVerbose > 1) {
      localtimer.Stop();
      G4cout << "G4Scheduler: process time= " << localtimer << G4endl;
    }
#endif
  }
#ifdef G4VERBOSE
  if (fVerbose != 0) {
    if (trackFound) {
      G4cout << "*** G4Scheduler ends at time : " << G4BestUnit(fGlobalTime, "Time") << G4endl;
      G4cout << "___________________________________" << G4endl;
    }
    else {
      G4cout << "*** G4Scheduler did not start because no "
                "track was found to be processed"
             << G4endl;
      G4cout << "___________________________________" << G4endl;
    }
  }
#endif

  fRunning = false;

  if (fpUserTimeStepAction != nullptr) {
    fpUserTimeStepAction->EndProcessing();
  }
  // ___________________
  EndTracking();
  ClearList();
  Reset();

  if (fpTrackingInteractivity != nullptr) {
    fpTrackingInteractivity->Finalize();
  }
}

//_________________________________________________________________________

G4double G4Scheduler::GetNextWatchedTime() const
{
  auto up = fWatchedTimes.upper_bound(fGlobalTime);
  if (up == fWatchedTimes.end()) {
    return DBL_MAX;
  }
  return *up;
}

//_________________________________________________________________________

void G4Scheduler::SynchronizeTracks()
{
  fTmpGlobalTime = fGlobalTime;
  fGlobalTime = fTrackContainer.GetNextTime();
  G4double tmpGlobalTime = fGlobalTime;
  G4double nextWatchedTime = -1;
  G4bool carryOn = true;
  while (fTrackContainer.MergeNextTimeToMainList(tmpGlobalTime) && carryOn) {
    if (tmpGlobalTime != fGlobalTime) {
      fGlobalTime = tmpGlobalTime;
    };
    fStopTime = min(fTrackContainer.GetNextTime(), fEndTime);
    while ((nextWatchedTime = GetNextWatchedTime()) < fTrackContainer.GetNextTime()
           && (carryOn = CanICarryOn()))
    {
      fStopTime = min(nextWatchedTime, fEndTime);
      DoProcess();
    }

    carryOn = CanICarryOn();

    if (nextWatchedTime > fEndTime && carryOn) {
      fStopTime = min(fTrackContainer.GetNextTime(), fEndTime);
      DoProcess();
    }
  }
}

//_________________________________________________________________________

G4bool G4Scheduler::CanICarryOn()
{
  return fGlobalTime < fEndTime && (fMaxSteps == -1 ? true : fNbSteps < fMaxSteps) && fContinue;
}

//_________________________________________________________________________

void G4Scheduler::PrintWhyDoYouStop()
{
#ifdef G4VERBOSE
  if (fWhyDoYouStop) {
    G4cout << "G4Scheduler has reached a stage: it might be"
              " a transition or the end"
           << G4endl;

    G4bool normalStop = false;

    if (fGlobalTime >= fStopTime) {
      G4cout << "== G4Scheduler: I stop because I reached the stop time : "
             << G4BestUnit(fStopTime, "Time") << " ==" << G4endl;
      normalStop = true;
    }
    if (!fTrackContainer.MainListsNOTEmpty())  // is empty
    {
      G4cout << "G4Scheduler: I stop because the current main list of tracks "
                "is empty"
             << G4endl;
      normalStop = true;
    }
    if (fMaxSteps == -1 ? false : fNbSteps >= fMaxSteps) {
      G4cout << "G4Scheduler: I stop because I reached the maximum allowed "
                "number of steps="
             << fMaxSteps << G4endl;
      normalStop = true;
    }
    if (fContinue && !normalStop) {
      G4cout << "G4Scheduler: It might be that I stop because "
                "I have been told so. You may check "
                "member fContinue and usage of the method G4Scheduler::Stop()."
             << G4endl;
    }
  }
#endif
}

//_________________________________________________________________________

void G4Scheduler::DoProcess()
{
  if (fpUserTimeStepAction != nullptr) fpUserTimeStepAction->NewStage();

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_STEPPING)
  MemStat mem_first, mem_second, mem_diff;
#endif

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_STEPPING)
  mem_first = MemoryUsage();
#endif

  while (fGlobalTime < fStopTime && fTrackContainer.MainListsNOTEmpty()
         && (fMaxSteps == -1 ? true : fNbSteps < fMaxSteps) && fContinue)
  {
    Stepping();

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_STEPPING)
    mem_second = MemoryUsage();
    mem_diff = mem_second - mem_first;
    G4cout << "\t || MEM || After step " << fNbSteps << ", diff is : " << mem_diff << G4endl;
#endif
  }

  PrintWhyDoYouStop();

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_STEPPING)
  mem_second = MemoryUsage();
  mem_diff = mem_second - mem_first;
  G4cout << "\t || MEM || After stepping, diff is : " << mem_diff << G4endl;
#endif

#ifdef G4VERBOSE
  if (fVerbose > 2)
    G4cout << "*** G4Scheduler has finished processing a track list at time : "
           << G4BestUnit(fGlobalTime, "Time") << G4endl;
#endif
}
//_________________________________________________________________________

void G4Scheduler::Stepping()
{
  fTimeStep = fMaxTimeStep;

  fTSTimeStep = DBL_MAX;
  fILTimeStep = DBL_MAX;

  //fInteractionStep = false;
  fReachedUserTimeLimit = false;

  fITStepStatus = eUndefined;

  // Start of step
#ifdef G4VERBOSE
  if (fVerbose > 2) {
#  ifdef USE_COLOR
    G4cout << LIGHT_RED;
#  endif
    G4cout << "*** Start Of Step N°" << fNbSteps + 1 << " species number : " << GetNTracks()
           << " ***" << G4endl;
    G4cout << "Current Global time : " << G4BestUnit(fGlobalTime, "Time") << G4endl;
#  ifdef USE_COLOR
    G4cout << RESET_COLOR;
#  endif
  }
#endif

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_DETAILED_STEPPING)
  MemStat mem_first, mem_second, mem_diff;
#endif

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_DETAILED_STEPPING)
  mem_first = MemoryUsage();
#endif

  fDefinedMinTimeStep = GetLimitingTimeStep();

  if (fUsePreDefinedTimeSteps) {
#ifdef G4VERBOSE
    if (fVerbose > 2) {
#  ifdef USE_COLOR
      G4cout << LIGHT_RED;
#  endif
      G4cout << "*** At time : " << G4BestUnit(fGlobalTime, "Time")
             << " the chosen user time step is : " << G4BestUnit(fDefinedMinTimeStep, "Time")
             << " ***" << G4endl;
#  ifdef USE_COLOR
      G4cout << RESET_COLOR;
#  endif
    }
#endif
  }

  if (fpModelProcessor->GetComputeTimeStep())  // fComputeTimeStep)
  {
    fTSTimeStep = fpModelProcessor->CalculateMinTimeStep(fGlobalTime, fDefinedMinTimeStep);
    // => at least N (N = nb of tracks) loops
  }
  else if (fUseDefaultTimeSteps) {
    fTSTimeStep = fDefinedMinTimeStep;
  }

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_DETAILED_STEPPING)
  mem_second = MemoryUsage();
  mem_diff = mem_second - mem_first;
  G4cout << "|| MEM || After computing TS, diff is : " << mem_diff << G4endl;
#endif

#ifdef G4VERBOSE
  if (fVerbose > 2) {
#  ifdef USE_COLOR
    G4cout << LIGHT_RED;
#  endif
    G4cout << "*** Time stepper returned : " << G4BestUnit(fTSTimeStep, "Time") << " ***" << G4endl;
#  ifdef USE_COLOR
    G4cout << RESET_COLOR;
#  endif
  }
#endif

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_DETAILED_STEPPING)
  mem_first = MemoryUsage();
#endif

  // Call IL even if fTSTimeStep == 0
  // if fILTimeStep == 0 give the priority to DoIt processes

  fILTimeStep = fpStepProcessor->ComputeInteractionLength(fPreviousTimeStep);
  // => at least N loops
  // All process returns the physical step of interaction
  // The transportation process calculates the corresponding
  // time step

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_DETAILED_STEPPING)
  mem_second = MemoryUsage();
  mem_diff = mem_second - mem_first;
  G4cout << "|| MEM || After IL, diff is : " << mem_diff << G4endl;
#endif

#ifdef G4VERBOSE
  if (fVerbose > 2) {
#  ifdef USE_COLOR
    G4cout << LIGHT_RED;
#  endif
    G4cout << "*** The minimum time returned by the processes is : "
           << G4BestUnit(fILTimeStep, "Time") << " ***" << G4endl;
#  ifdef USE_COLOR
    G4cout << RESET_COLOR;
#  endif
  }
#endif

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_DETAILED_STEPPING)
  mem_first = MemoryUsage();
#endif

  if (fILTimeStep <= fTSTimeStep)
  // Give the priority to the IL
  {
    fInteractionStep = true;
    //fReactionSet->CleanAllReaction();
    fTimeStep = fILTimeStep;
    fITStepStatus = eInteractionWithMedium;
    fpStepProcessor->PrepareLeadingTracks();
  }
  else {
    fInteractionStep = false;
    fpStepProcessor->ResetLeadingTracks();
    fTimeStep = fTSTimeStep;
    fITStepStatus = eCollisionBetweenTracks;
  }

  if (fGlobalTime + fTimeStep > fStopTime)
  // This check is done at every time step
  {
    fTimeStep = fStopTime - fGlobalTime;
    fITStepStatus = eInteractionWithMedium;  // ie: transportation
    fInteractionStep = true;
    fReactionSet->CleanAllReaction();
    fpStepProcessor->ResetLeadingTracks();
  }

  if (fTimeStep == 0)  // < fTimeTolerance)
  {
    ++fZeroTimeCount;
    if (fZeroTimeCount >= fMaxNZeroTimeStepsAllowed) {
      G4ExceptionDescription exceptionDescription;

      exceptionDescription << "Too many zero time steps were detected. ";
      exceptionDescription << "The simulation is probably stuck. ";
      exceptionDescription << "The maximum number of zero time steps is currently : "
                           << fMaxNZeroTimeStepsAllowed;
      exceptionDescription << ".";

      G4Exception("G4Scheduler::Stepping", "SchedulerNullTimeSteps", FatalErrorInArgument,
                  exceptionDescription);
    }
  }
  else {
    fZeroTimeCount = 0;
  }

  fReachedUserTimeLimit = (fTimeStep <= fDefinedMinTimeStep)
                          || ((fTimeStep > fDefinedMinTimeStep)
                              && fabs(fTimeStep - fDefinedMinTimeStep) < fTimeTolerance);

  if (fpUserTimeStepAction != nullptr) fpUserTimeStepAction->UserPreTimeStepAction();
    // TODO: pre/post

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_DETAILED_STEPPING)
  mem_second = MemoryUsage();
  mem_diff = mem_second - mem_first;
  G4cout << "|| MEM || After LeadingTracks and UserPreTimeStepAction: " << mem_diff << G4endl;
#endif

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_DETAILED_STEPPING)
  mem_first = MemoryUsage();
#endif

  fGlobalTime += fTimeStep;

  // if fTSTimeStep > 0 => still need to call the transportation process
  // if fILTimeStep < fTSTimeStep => call only DoIt processes, no reactions
  // if fILTimeStep == fTSTimeStep => give the priority to the DoIt processes
  if (fTSTimeStep > 0 || fILTimeStep <= fTSTimeStep) {
    fpStepProcessor->DoIt(fTimeStep);
  }
#if defined(DEBUG_MEM) && defined(DEBUG_MEM_DETAILED_STEPPING)
  mem_second = MemoryUsage();
  mem_diff = mem_second - mem_first;
  G4cout << "|| MEM || After DoIT, diff is : " << mem_diff << G4endl;
#endif

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_DETAILED_STEPPING)
  mem_first = MemoryUsage();
#endif

  fpModelProcessor->ComputeTrackReaction(fITStepStatus, fGlobalTime, fTimeStep, fPreviousTimeStep,
                                         fReachedUserTimeLimit, fTimeTolerance,
                                         fpUserTimeStepAction, fVerbose);

  ++fNbSteps;

  if (fpUserTimeStepAction != nullptr) {
    fpUserTimeStepAction->UserPostTimeStepAction();
  }

  fPreviousTimeStep = fTimeStep;

#if defined(DEBUG_MEM) && defined(DEBUG_MEM_DETAILED_STEPPING)
  mem_second = MemoryUsage();
  mem_diff = mem_second - mem_first;
  G4cout << "|| MEM || After computing reactions + UserPostTimeStepAction, "
            "diff is : "
         << mem_diff << G4endl;
#endif

  // End of step
#ifdef G4VERBOSE
  if (fVerbose >= 2) {
#  ifdef USE_COLOR
    G4cout << LIGHT_RED;
#  endif

    G4String interactionType;
    GetCollisionType(interactionType);

    std::stringstream finalOutput;

    finalOutput << "*** End of step N°" << fNbSteps
                << "\t T_i= " << G4BestUnit(fGlobalTime - fTimeStep, "Time")
                << "\t dt= " << G4BestUnit(fTimeStep, "Time")
                << "\t T_f= " << G4BestUnit(fGlobalTime, "Time") << "\t " << interactionType
                << G4endl;

    if (fVerbose > 2) {
      if (fReachedUserTimeLimit) {
        finalOutput << "It has also reached the user time limit" << G4endl;
      }
      finalOutput << "_______________________________________________________________"
                     "_______"
                  << G4endl;
    }

    G4cout << finalOutput.str();

#  ifdef USE_COLOR
    G4cout << RESET_COLOR;
#  endif
  }
#endif
}
//_________________________________________________________________________

G4double G4Scheduler::GetLimitingTimeStep() const
{
  if (fpUserTimeSteps == nullptr) return fDefaultMinTimeStep;
  if (fabs(fGlobalTime - fUserUpperTimeLimit) < fTimeTolerance) return fDefinedMinTimeStep;

  auto it_fpUserTimeSteps_i = fpUserTimeSteps->upper_bound(fGlobalTime);
  auto it_fpUserTimeSteps_low = fpUserTimeSteps->lower_bound(fGlobalTime);

  if (it_fpUserTimeSteps_i == fpUserTimeSteps->end()) {
    it_fpUserTimeSteps_i--;
    fUserUpperTimeLimit = fStopTime;
  }
  else if (fabs(fGlobalTime - it_fpUserTimeSteps_low->first) < fTimeTolerance) {
    it_fpUserTimeSteps_i = it_fpUserTimeSteps_low;
    auto tmp_it = it_fpUserTimeSteps_low;
    ++tmp_it;
    if (tmp_it == fpUserTimeSteps->end()) {
      fUserUpperTimeLimit = fStopTime;
    }
    else {
      fUserUpperTimeLimit = tmp_it->first;
    }
  }
  else if (it_fpUserTimeSteps_i == it_fpUserTimeSteps_low) {
    fUserUpperTimeLimit = it_fpUserTimeSteps_i->first;
    if (it_fpUserTimeSteps_i != fpUserTimeSteps->begin()) it_fpUserTimeSteps_i--;
  }
  else {
    fUserUpperTimeLimit = it_fpUserTimeSteps_i->first;
    it_fpUserTimeSteps_i = it_fpUserTimeSteps_low;
  }
  return it_fpUserTimeSteps_i->second;
}

//_________________________________________________________________________

void G4Scheduler::FindUserPreDefinedTimeStep()
{
  if (fpUserTimeSteps == nullptr) {
    G4ExceptionDescription exceptionDescription;
    exceptionDescription << "You are asking to use user defined steps but you did not give any.";
    G4Exception("G4Scheduler::FindUserPreDefinedTimeStep", "Scheduler004", FatalErrorInArgument,
                exceptionDescription);
    return;  // makes coverity happy
  }
  auto fpUserTimeSteps_i = fpUserTimeSteps->upper_bound(fGlobalTime);
  auto fpUserTimeSteps_low = fpUserTimeSteps->lower_bound(fGlobalTime);

  if (fpUserTimeSteps_i == fpUserTimeSteps->end()) {
    fpUserTimeSteps_i--;
  }
  else if (fabs(fGlobalTime - fpUserTimeSteps_low->first) < fTimeTolerance) {
    // Case : fGlobalTime = X picosecond
    // and fpUserTimeSteps_low->first = X picosecond
    // but the precision is not good enough
    fpUserTimeSteps_i = fpUserTimeSteps_low;
  }
  else if (fpUserTimeSteps_i == fpUserTimeSteps_low) {
    // "Normal" cases
    fpUserTimeSteps_i--;
  }
  else {
    fpUserTimeSteps_i = fpUserTimeSteps_low;
  }

  fDefinedMinTimeStep = fpUserTimeSteps_i->second;
}

//_________________________________________________________________________

void G4Scheduler::EndTracking()
{
  if (fRunning) {
    G4ExceptionDescription exceptionDescription;
    exceptionDescription << "End tracking is called while G4Scheduler is still running." << G4endl;

    G4Exception("G4Scheduler::EndTracking", "Scheduler017", FatalErrorInArgument,
                exceptionDescription);
  }

  while (fTrackContainer.DelayListsNOTEmpty()) {
    auto nextTime = fTrackContainer.GetNextTime();
    fTrackContainer.MergeNextTimeToMainList(nextTime);
  }

  fTrackContainer.MergeSecondariesWithMainList();

  if (fTrackContainer.MainListsNOTEmpty()) {
    G4TrackManyList* mainList = fTrackContainer.GetMainList();
    G4TrackManyList::iterator it = mainList->begin();
    G4TrackManyList::iterator end = mainList->end();
    for (; it != end; ++it) {
      fpTrackingManager->EndTrackingWOKill(*it);
    }
  }

  if (fTrackContainer.SecondaryListsNOTEmpty())  // should be empty
  {
    G4TrackManyList* secondaries = fTrackContainer.GetSecondariesList();
    G4TrackManyList::iterator it = secondaries->begin();
    G4TrackManyList::iterator end = secondaries->end();

    for (; it != end; ++it) {
      fpTrackingManager->EndTrackingWOKill(*it);
    }
  }
}

//_________________________________________________________________________
void G4Scheduler::SetInteractivity(G4ITTrackingInteractivity* interactivity)
{
  fpTrackingInteractivity = interactivity;
  if (fpTrackingManager != nullptr) {
    fpTrackingManager->SetInteractivity(fpTrackingInteractivity);
  }
}

//_________________________________________________________________________
void G4Scheduler::ForceReinitialization()
{
  fInitialized = false;
  Initialize();
}

size_t G4Scheduler::GetNTracks()
{
  return fTrackContainer.GetNTracks();
}
//_________________________________________________________________________

void G4Scheduler::GetCollisionType(G4String& interactionType)
{
  switch (fITStepStatus) {
    case eInteractionWithMedium:
      interactionType = "eInteractionWithMedium";
      break;
    case eCollisionBetweenTracks:
      interactionType = "eCollisionBetweenTracks";
      break;
    default:
      interactionType = "eCollisionBetweenTracks";
      break;
  }
}
