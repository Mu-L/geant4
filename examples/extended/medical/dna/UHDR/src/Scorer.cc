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

#include "Scorer.hh"

#include "InterPulseAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "PulseAction.hh"
#include "TimeStepAction.hh"

#include "G4AnalysisManager.hh"
#include "G4DNAEventScheduler.hh"
#include "G4DNAScavengerMaterial.hh"
#include "G4Event.hh"
#include "G4PhysicalConstants.hh"
#include "G4RunManager.hh"
#include "G4Scheduler.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4VChemistryWorld.hh"

#include <G4EventManager.hh>
#include <G4MolecularConfiguration.hh>
#include <G4SystemOfUnits.hh>
#include <globals.hh>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// Dose
Dose::Dose()
  : G4UImessenger(),
    fpDoseDir(new G4UIdirectory("/scorer/Dose/")),
    fpAddDoseCutOff(new G4UIcmdWithADoubleAndUnit("/scorer/Dose/cutoff", this)),
    fpAddDoseToAbort(new G4UIcmdWithADoubleAndUnit("/scorer/Dose/abortedDose", this))
{
  fpDoseDir->SetGuidance("Dose scorer commands");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Dose::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == fpAddDoseCutOff.get()) {
    fDosesCutOff = G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue);
  }
  if (command == fpAddDoseToAbort.get()) {
    fDosesToAbort = G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

template<>
void Scorer<Dose>::SetChemistryWorld(G4VChemistryWorld* pChemistryWorld)
{
  fpChemistryWorld = pChemistryWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

template<>
G4VChemistryWorld* Scorer<Dose>::GetChemistryWorld() const
{
  return fpChemistryWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

template<>
void Scorer<Dose>::clear()
{
  fpScorer->fCumulatedDose = 0.;
  fpScorer->fPulseMax = 0;
}

template<>
void Scorer<Dose>::Initialize(G4HCofThisEvent* HCE)
{
  clear();
  fpEvtMap = new G4THitsMap<G4double>(GetMultiFunctionalDetector()->GetName(), GetName());
  if (fHCID < 0) {
    fHCID = GetCollectionID(0);
  }
  HCE->AddHitsCollection(fHCID, (G4VHitsCollection*)fpEvtMap);

  fPulseActionInfo =
    dynamic_cast<const InterPulseAction*>(G4RunManager::GetRunManager()->GetUserTrackingAction());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

template<>
void Scorer<Dose>::EndOfEvent(G4HCofThisEvent*)
{
  if (!G4RunManager::GetRunManager()->GetCurrentEvent()->IsAborted()) {
    fpEvtMap->add(0, fpScorer->fDosesCutOff);
  }
  fpScorer->fCumulatedDose = 0.;
  fpScorer->fPulseMax = 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

template<>
G4bool Scorer<Dose>::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
  auto currentEvent = G4EventManager::GetEventManager();
  const G4Track* track = aStep->GetTrack();
  auto boundingBox = fpChemistryWorld->GetChemistryBoundary();

  G4double V = boundingBox->Volume() / cm3;
  G4double edep = aStep->GetTotalEnergyDeposit();
  if (edep == 0.) {
    return false;
  }
  G4double eToGray = (0.001 * V * 6.242e+18);
  if (eToGray != 0) {
    (fpScorer->fCumulatedDose) += ((edep) / eV) / eToGray;
  }
  if (track->GetParentID() == 0 && aStep->IsFirstStepInVolume()) {
    auto pulseInfo = dynamic_cast<PulseInfo*>(track->GetUserInformation());
    if (pulseInfo != nullptr) {
      auto delayedTime = pulseInfo->GetDelayedTime();
      if (fpScorer->fPulseMax < delayedTime) {
        fpScorer->fPulseMax = delayedTime;
      }
    }
    G4double DoseInGray = fpScorer->fCumulatedDose;
    if (DoseInGray > fpScorer->fDosesCutOff / gray) {
      G4cout << "_____________________________________________________________________________"
             << G4endl;
      auto name = currentEvent->GetConstCurrentEvent()
                    ->GetPrimaryVertex()
                    ->GetPrimary()
                    ->GetParticleDefinition()
                    ->GetParticleName();
      auto energy =
        currentEvent->GetConstCurrentEvent()->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy();

      G4cout << "Beam line            : " << "(" << name << ", " << energy << " MeV)" << G4endl;
      G4cout << "Cut-off dose         : " << fpScorer->fDosesCutOff / gray << " Gy" << G4endl;
      G4cout << "Stop at actual dose  : " << DoseInGray << " Gy" << G4endl;
      if (fPulseActionInfo != nullptr) {
        auto numberOfPulse = fPulseActionInfo->GetNumberOfPulse();
        auto DIT = fPulseActionInfo->GetPulsePeriod();
        G4cout << "DIT                  : " << DIT / CLHEP::ms << " ms" << G4endl;
        G4cout << "Pulse number         : " << numberOfPulse << G4endl;
      }

      const auto generatorAction = static_cast<const PrimaryGeneratorAction*>(
        G4RunManager::GetRunManager()->GetUserPrimaryGeneratorAction());
      const auto particleGun = generatorAction->GetSPGun();
      auto NumberOfParticlesGeneratedinOneEvent = particleGun->GetNumberOfParticles();

      G4cout << "Beam duration        : " << fpScorer->fPulseMax / second << " s" << G4endl;
      if(fpScorer->fPulseMax != 0){
        G4cout << "Actual dose rate     : " << DoseInGray / (fpScorer->fPulseMax / second) << " Gy/s"
               << G4endl;
      }else{
        G4cout << "Actual dose rate     : " << "infinite"
               << G4endl;
      }

      G4cout << "Track number         : "
             << NumberOfParticlesGeneratedinOneEvent
                  - currentEvent->GetStackManager()->GetNUrgentTrack() - 1
             << " tracks" << G4endl;
      G4cout << "Irradiated volume    : " << V * 1e+12  // convert cm3 to um3
             << " um3 (" << 2 * boundingBox->halfSideLengthInX() / um << " x "
             << 2 * boundingBox->halfSideLengthInY() / um << " x "
             << 2 * boundingBox->halfSideLengthInZ() / um << ")" << G4endl;
      auto ChemComponent = fpChemistryWorld->GetChemicalComponent();
      for (const auto& it : ChemComponent) {
        if (fH2O == it.first) continue;
        if (fOHm == it.first) continue;
        if (it.first == fH3Op) {
          G4cout << "pH                   : " << -std::log10(it.second * (mole * liter)) << G4endl;
          continue;
        }
        G4cout << it.first->GetName() << "                : " << it.second * (mole * liter) << " M "
               << G4endl;
      }

      G4cout << "Total deposit energy : " << fpScorer->fCumulatedDose * eToGray << " eV" << G4endl;
      G4double DoseAbort;
      if (fpScorer->fDosesToAbort == 0) {
        DoseAbort = fpScorer->fDosesCutOff / gray + 0.5 * fpScorer->fDosesCutOff / gray;
      }
      else {
        DoseAbort = fpScorer->fDosesToAbort / gray;
      }

      G4cout << "Dose to abort        : " << DoseAbort << " Gy" << G4endl;
      if (DoseInGray > DoseAbort) {
        G4cout << "Abort this beam line (" << name << ", " << energy
               << " MeV) at actual dose: " << DoseInGray << " Gy" << G4endl;
        G4RunManager::GetRunManager()->AbortEvent();
      }
      G4cout << "_____________________________________________________________________________"
             << G4endl;
      auto myTrack = ((G4Track*)track);
      myTrack->SetTrackStatus(fStopAndKill);
      auto secondaries = track->GetStep()->GetSecondaryInCurrentStep();
      if (!secondaries->empty()) {
        for (auto it : *(secondaries)) {
          if (it != nullptr) {
            ((G4Track*)it)->SetTrackStatus(fStopAndKill);
          }
        }
      }
      currentEvent->GetStackManager()->ClearUrgentStack();
    }
  }
  return true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// Gvalues
Gvalues::Gvalues()
  : G4UImessenger(),
    fTimeLimit(G4Scheduler::Instance()->GetEndTime()),
    fSpeciesdir(new G4UIdirectory("/scorer/Gvalues/")),
    fTimeBincmd(new G4UIcmdWithAnInteger("/scorer/Gvalues/nOfTimeBins", this)),
    fAddTimeToRecordcmd(new G4UIcmdWithADoubleAndUnit("/scorer/Gvalues/addTimeToRecord", this))
{
  fSpeciesdir->SetGuidance("ScoreSpecies commands");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void Gvalues::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == fAddTimeToRecordcmd.get()) {
    G4double cmdTime = fAddTimeToRecordcmd->GetNewDoubleValue(newValue);
    if (fTimeLimit >= cmdTime) {
      AddTimeToRecord(cmdTime);
    }
    else {
      AddTimeToRecord(fTimeLimit);
    }
  }
  if (command == fTimeBincmd.get()) {
    G4int cmdBins = fTimeBincmd->GetNewIntValue(newValue);
    G4double timeMin = 1 * ps;
    G4double timeMax = G4Scheduler::Instance()->GetEndTime() - timeMin;
    G4double timeLogMin = std::log10(timeMin);
    G4double timeLogMax = std::log10(timeMax);
    for (int i = 0; i <= cmdBins; i++) {
      AddTimeToRecord(std::pow(10, timeLogMin + i * (timeLogMax - timeLogMin) / cmdBins));
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void Gvalues::WriteInfo(G4VAnalysisManager* analysisManager, const std::string& out)
{
  G4int NtupleID = analysisManager->CreateNtuple("info", "Simulation");
  analysisManager->CreateNtupleDColumn(NtupleID, "Dose");
  analysisManager->FinishNtuple(NtupleID);
  analysisManager->FillNtupleDColumn(NtupleID, 0, std::stod(out));
  analysisManager->AddNtupleRow(NtupleID);
}

void Gvalues::WriteGvalues(G4VAnalysisManager* analysisManager)
{
  G4int NtupleID = analysisManager->CreateNtuple("Gvalue", "Gvalue");
  analysisManager->CreateNtupleIColumn(NtupleID, "speciesID");
  analysisManager->CreateNtupleIColumn(NtupleID, "number");
  analysisManager->CreateNtupleIColumn(NtupleID, "nEvent");
  analysisManager->CreateNtupleSColumn(NtupleID, "speciesName");
  analysisManager->CreateNtupleDColumn(NtupleID, "time");
  analysisManager->CreateNtupleDColumn(NtupleID, "sumG");
  analysisManager->CreateNtupleDColumn(NtupleID, "sumG2");
  analysisManager->CreateNtupleDColumn(NtupleID, "TotalDose");
  analysisManager->CreateNtupleDColumn(NtupleID, "TotalDose2");
  analysisManager->CreateNtupleDColumn(NtupleID, "TotalDoseRate");
  analysisManager->CreateNtupleDColumn(NtupleID, "TotalDoseRate2");
  analysisManager->FinishNtuple(NtupleID);

  for (const auto& it_map1 : fSpeciesInfoPerTime) {
    const InnerSpeciesMap& map2 = it_map1.second;
    for (auto it_map2 : map2) {
      double time = it_map1.first;
      auto species = it_map2.first;
      const G4String& name = species->GetName();
      int molID = it_map2.first->GetMoleculeID();
      auto number = it_map2.second.fNumber;
      double G = it_map2.second.fG;
      double G2 = it_map2.second.fG2;

      analysisManager->FillNtupleIColumn(NtupleID, 0, molID);  // MolID
      analysisManager->FillNtupleIColumn(NtupleID, 1, number);  // Number
      analysisManager->FillNtupleIColumn(NtupleID, 2, fNEvent);  // Total nb events
      analysisManager->FillNtupleSColumn(NtupleID, 3, name);  // molName
      analysisManager->FillNtupleDColumn(NtupleID, 4, time);  // time
      analysisManager->FillNtupleDColumn(NtupleID, 5, G);  // G
      analysisManager->FillNtupleDColumn(NtupleID, 6, G2);  // G2
      analysisManager->FillNtupleDColumn(NtupleID, 7, fTotalDose);
      analysisManager->FillNtupleDColumn(NtupleID, 8, fTotalDose2);
      analysisManager->FillNtupleDColumn(NtupleID, 9, fTotalDoseRate);
      analysisManager->FillNtupleDColumn(NtupleID, 10, fTotalDoseRate2);
      analysisManager->AddNtupleRow(NtupleID);
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....
void Gvalues::WriteWithAnalysisManager(G4VAnalysisManager* analysisManager, const std::string& out)
{
  WriteInfo(analysisManager, out);
  WriteGvalues(analysisManager);
  // fRunID++;
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

template<>
void Scorer<Gvalues>::clear()
{
  fpEvtMap->clear();
  fpScorer->fNEvent = 0;
  fpScorer->fEdep = 0;
  fpScorer->fSpeciesInfoPerTime.clear();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

template<>
void Scorer<Gvalues>::Initialize(G4HCofThisEvent* HCE)
{
  fpEvtMap = new G4THitsMap<G4double>(GetMultiFunctionalDetector()->GetName(), GetName());
  if (fHCID < 0) {
    fHCID = GetCollectionID(0);
  }
  HCE->AddHitsCollection(fHCID, (G4VHitsCollection*)fpEvtMap);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

template<>
void Scorer<Gvalues>::SetChemistryWorld(G4VChemistryWorld* pChemistryWorld)
{
  fpChemistryWorld = pChemistryWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

template<>
G4VChemistryWorld* Scorer<Gvalues>::GetChemistryWorld() const
{
  return fpChemistryWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

template<>
G4bool Scorer<Gvalues>::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
  const G4Track* track = aStep->GetTrack();
  auto pulseInfo = dynamic_cast<PulseInfo*>(track->GetUserInformation());
  if (pulseInfo != nullptr) {
    auto delayedTime = pulseInfo->GetDelayedTime();
    if (fpScorer->fPulseMax < delayedTime) {
      fpScorer->fPulseMax = delayedTime;
    }
  }

  G4double edep = aStep->GetTotalEnergyDeposit();
  if (edep == 0.) {
    return FALSE;
  }
  edep *= aStep->GetPreStepPoint()->GetWeight();  // (Particle Weight)
  G4int index = GetIndex(aStep);
  fpEvtMap->add(index, edep);
  (fpScorer->fEdep) += edep;
  return TRUE;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

template<>
void Scorer<Gvalues>::SaveScavengerChange()
{
  auto pScavengerMaterial =
    dynamic_cast<G4DNAScavengerMaterial*>(G4Scheduler::Instance()->GetScavengerMaterial());
  if (pScavengerMaterial == nullptr) {
    G4ExceptionDescription errMsg;
    errMsg << "pScavengerMaterial == nullptr";
    G4Exception("Scorer<Gvalues>::SaveScavengerChange()", "SaveScavengerChange",
                FatalErrorInArgument, errMsg);
  }
  auto scavengerList = pScavengerMaterial->GetScavengerList();
  auto V = fpChemistryWorld->GetChemistryBoundary()->Volume();
  auto iniC = fpChemistryWorld->GetChemicalComponent();

  G4double eToGray = (0.001 * V * 6.242e+18);
  if (eToGray != 0) {
    G4double dose = 1000 * (fpScorer->fEdep / eV) / eToGray;  // in Gy
    for (const auto& it : scavengerList) {
      if (it == fH2O || fH3Op == it || fOHm == it) {
        continue;
      }
      for (auto time_mol : fpScorer->fTimeToRecord) {
        int64_t n_mol = pScavengerMaterial->GetNMoleculesAtTime(it, time_mol);
        if (n_mol < 0) {
          G4ExceptionDescription errMsg;
          errMsg << "SaveScavengerChange()::N molecules not valid < 0 : " << it->GetName()
                 << " N : " << n_mol << G4endl;
          G4Exception("", "N<0", FatalException, errMsg);
        }

        Gvalues::SpeciesInfo& molInfo = fpScorer->fSpeciesInfoPerTime[time_mol][it];
        molInfo.fNumber += n_mol;
        if (V > 0) {
          auto concentration = n_mol / (Avogadro * V /*mm3 to L*/);
          // auto percentage = 10 * 100 * (iniC[it]-concentration) * (mole * liter) / 0.0013; //in %
          // air/10 Gy
          auto inuM = (iniC[it] - concentration) * (mole * liter) * 1e6;  // in uM

          if (dose > 0) {
            G4double gValue = inuM / dose;  // in uM
            molInfo.fG += gValue;
            molInfo.fG2 += gValue * gValue;
          }
        }
      }
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

template<>
void Scorer<Gvalues>::SaveMoleculeCounter()
{
  if (fpEventScheduler == nullptr) {
    G4Exception("fpEventScheduler == nullptr", "Scorer<Gvalues>::SaveMoleculeCounter()",
                FatalException, "fpEventScheduler == nullptr");
  }
  else {
    auto counterMap = fpEventScheduler->GetCounterMap();
    if (counterMap.empty()) {
      G4Exception("No counter", "Scorer<Gvalues>::SaveMoleculeCounter()", JustWarning,
                  "CounterMap is not used");
      return;
    }
    for (const auto& map_mol : counterMap) {
      auto time_mol = map_mol.first;
      for (auto it_mol : map_mol.second) {
        auto molecule = it_mol.first;
        if (molecule == fO2) {
          continue;
        }
        int n_mol = it_mol.second;

        if (n_mol < 0) {
          G4ExceptionDescription errMsg;
          errMsg << "N molecules not valid < 0 " << " molecule : " << it_mol.first->GetName()
                 << " N : " << n_mol << G4endl;
          G4Exception("", "N<0", FatalException, errMsg);
        }

        if (fpScorer->fEdep > 0) {
          Gvalues::SpeciesInfo& molInfo = fpScorer->fSpeciesInfoPerTime[time_mol][molecule];
          molInfo.fNumber += n_mol;
          G4double gValue = (n_mol / (fpScorer->fEdep / eV)) * 100.;
          // G4double gValue = n_mol;
          molInfo.fG += gValue;
          molInfo.fG2 += gValue * gValue;
        }
      }
    }

    auto boundingBox = fpChemistryWorld->GetChemistryBoundary();
    G4double V = boundingBox->Volume() / cm3;
    G4double eToGray = (0.001 * V * 6.242e+18);
    if (eToGray != 0) {
      G4double dose = (fpScorer->fEdep / eV) / eToGray;  // in Gy
      if (fpScorer->fPulseMax != 0) {
        G4double doseRate = dose / (fpScorer->fPulseMax / second);
        fpScorer->fTotalDose += dose;
        fpScorer->fTotalDose2 += dose * dose;
        fpScorer->fTotalDoseRate += doseRate;
        fpScorer->fTotalDoseRate2 += doseRate * doseRate;
      }
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

template<>
void Scorer<Gvalues>::EndOfEvent(G4HCofThisEvent*)
{
  if (G4EventManager::GetEventManager()->GetConstCurrentEvent()->IsAborted()) {
    fpScorer->fEdep = 0.;
    fpEventScheduler->ResetCounter();
    return;
  }

  SaveScavengerChange();
  SaveMoleculeCounter();

  ++(fpScorer->fNEvent);
  fpScorer->fEdep = 0.;

  fpEventScheduler->ResetCounter();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

template<>
void Scorer<Gvalues>::AbsorbResultsFromWorkerScorer(G4VPrimitiveScorer* workerScorer)
{
  auto right = dynamic_cast<Scorer<Gvalues>*>(dynamic_cast<G4VPrimitiveScorer*>(workerScorer));

  if (right == nullptr) {
    return;
  }
  if (right == this) {
    return;
  }
  auto it_map1 = right->fpScorer->fSpeciesInfoPerTime.begin();
  auto end_map1 = right->fpScorer->fSpeciesInfoPerTime.end();

  for (; it_map1 != end_map1; ++it_map1) {
    Gvalues::InnerSpeciesMap& map2 = it_map1->second;
    auto it_map2 = map2.begin();
    auto end_map2 = map2.end();

    for (; it_map2 != end_map2; ++it_map2) {
      Gvalues::SpeciesInfo& molInfo = fpScorer->fSpeciesInfoPerTime[it_map1->first][it_map2->first];
      molInfo.fNumber += it_map2->second.fNumber;
      molInfo.fG += it_map2->second.fG;
      molInfo.fG2 += it_map2->second.fG2;
    }
  }

  // dose
  fpScorer->fTotalDose += right->fpScorer->fTotalDose;
  fpScorer->fTotalDose2 += right->fpScorer->fTotalDose2;
  right->fpScorer->fTotalDose = 0;
  right->fpScorer->fTotalDose2 = 0.;

  // dose rate
  fpScorer->fTotalDoseRate += right->fpScorer->fTotalDoseRate;
  fpScorer->fTotalDoseRate2 += right->fpScorer->fTotalDoseRate2;
  right->fpScorer->fTotalDoseRate = 0;
  right->fpScorer->fTotalDoseRate2 = 0.;

  right->fpScorer->fSpeciesInfoPerTime.clear();
  fpScorer->fNEvent += right->fpScorer->fNEvent;
  right->fpScorer->fNEvent = 0;
  right->fpScorer->fEdep = 0.;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

template<>
void Scorer<Gvalues>::OutputAndClear(const std::string& dose)
{
  if (G4Threading::IsWorkerThread()) {
    return;
  }
  G4VAnalysisManager* analysisManager = G4AnalysisManager::Instance();
  if (analysisManager != nullptr) {
    this->fpScorer->WriteWithAnalysisManager(analysisManager, dose);
  }
  fpScorer->fNEvent = 0;
  fpScorer->fSpeciesInfoPerTime.clear();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....
