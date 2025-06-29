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
#include "G4DNAScavengerMaterial.hh"

#include "G4DNABoundingBox.hh"
#include "G4DNAMolecularMaterial.hh"
#include "G4MolecularConfiguration.hh"
#include "G4PhysicalConstants.hh"
#include "G4Scheduler.hh"
#include "G4StateManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4VChemistryWorld.hh"

#include <memory>

using namespace std;

//------------------------------------------------------------------------------

G4DNAScavengerMaterial::G4DNAScavengerMaterial(G4VChemistryWorld* pChemistryInfo)
  : fpChemistryInfo(pChemistryInfo), fIsInitialized(false), fCounterAgainstTime(false), fVerbose(0)
{
  Initialize();
}

//------------------------------------------------------------------------------

void G4DNAScavengerMaterial::Initialize()
{
  if (fIsInitialized) {
    return;
  }

  if (fpChemistryInfo->size() == 0) {
    G4cout << "G4DNAScavengerMaterial existed but empty" << G4endl;
  }
  Reset();

  fEquilibriumProcesses.emplace(
    std::make_pair(6, std::make_unique<G4ChemEquilibrium>(6, 10 * CLHEP::us)));//reactionType6 and 10 * us
  fEquilibriumProcesses.emplace(
    std::make_pair(7, std::make_unique<G4ChemEquilibrium>(7, 10 * CLHEP::us)));//reactionType6 and 10 * us
  fEquilibriumProcesses.emplace(
    std::make_pair(8, std::make_unique<G4ChemEquilibrium>(8, 10 * CLHEP::us)));//reactionType6 and 10 * us
  for(auto& it : fEquilibriumProcesses)
  {
    it.second->Initialize();
    it.second->SetVerbose(fVerbose);
  }

  fIsInitialized = true;
}

G4double
G4DNAScavengerMaterial::GetNumberMoleculePerVolumeUnitForMaterialConf(MolType matConf) const
{
  // no change these molecules
  if (fH2O == matConf) {
    return 0;
  }

  auto iter = fScavengerTable.find(matConf);
  if (iter == fScavengerTable.end()) {
    return 0;
  }

  if (iter->second >= 1) {
    return (floor)(iter->second);
  }

  return 0;
}

void G4DNAScavengerMaterial::ReduceNumberMoleculePerVolumeUnitForMaterialConf(MolType matConf,
                                                                              G4double time)
{
  // no change these molecules
  if (fH2O == matConf || fH3Op == matConf ||  // suppose that pH is not changed during simu
      fHOm == matConf)
  {
    // G4cout<<"moletype : "<<matConf->GetName()<<G4endl;
    // kobs is already counted these molecule concentrations
    return;
  }
  if (!find(matConf))  // matConf must greater than 0
  {
    return;
  }
  fScavengerTable[matConf]--;
  if (fScavengerTable[matConf] < 0)  // projection
  {
    assert(false);
  }

  if (fCounterAgainstTime) {
    RemoveAMoleculeAtTime(matConf, time);
  }
}

void G4DNAScavengerMaterial::AddNumberMoleculePerVolumeUnitForMaterialConf(MolType matConf,
                                                                           G4double time)
{
  // no change these molecules

  if (fH2O == matConf || fH3Op == matConf ||  // pH has no change
      fHOm == matConf)
  {
    // G4cout<<"moletype : "<<matConf->GetName()<<G4endl;
    // kobs is already counted these molecule concentrations
    return;
  }

  auto it = fScavengerTable.find(matConf);
  if (it == fScavengerTable.end())  // matConf must be in fScavengerTable
  {
    return;
  }
  fScavengerTable[matConf]++;

  if (fCounterAgainstTime) {
    AddAMoleculeAtTime(matConf, time);
  }
}

void G4DNAScavengerMaterial::PrintInfo()
{
  auto pConfinedBox = fpChemistryInfo->GetChemistryBoundary();
  auto iter = fpChemistryInfo->begin();
  G4cout << "**************************************************************" << G4endl;
  for (; iter != fpChemistryInfo->end(); iter++) {
    auto containedConf = iter->first;
    // auto concentration = iter->second;
    auto concentration = fScavengerTable[containedConf] / (Avogadro * pConfinedBox->Volume());
    G4cout << "Scavenger:" << containedConf->GetName() << "  : "
           << concentration / 1.0e-6 /*mm3 to L*/ << " (M)  with : "
           << fScavengerTable[containedConf] << " (molecules)"
           << "in: " << pConfinedBox->Volume() / (um * um * um) << " (um3)" << G4endl;
    if (fScavengerTable[containedConf] < 1) {
      G4cout << "!!!!!!!!!!!!! this molecule has less one molecule for "
                "considered volume"
             << G4endl;
      // assert(false);
    }
    if (fVerbose != 0) {
      Dump();
    }
  }
  G4cout << "**************************************************************" << G4endl;
}

void G4DNAScavengerMaterial::Reset()
{
  if (fpChemistryInfo == nullptr) {
    return;
  }

  if (fpChemistryInfo->size() == 0) {
    return;
  }

  ResetEquilibrium();

  fScavengerTable.clear();
  fCounterMap.clear();
  fpLastSearch.reset(nullptr);

  auto pConfinedBox = fpChemistryInfo->GetChemistryBoundary();
  auto iter = fpChemistryInfo->begin();
  for (; iter != fpChemistryInfo->end(); iter++) {
    auto containedConf = iter->first;
    auto concentration = iter->second;
    fScavengerTable[containedConf] = floor(Avogadro * concentration * pConfinedBox->Volume());
    fCounterMap[containedConf][1 * picosecond] =
      floor(Avogadro * concentration * pConfinedBox->Volume());
  }
  if (fVerbose != 0) {
    PrintInfo();
  }
}

//------------------------------------------------------------------------------

void G4DNAScavengerMaterial::AddAMoleculeAtTime(MolType molecule, G4double time,
                                                const G4ThreeVector* /*position*/, int number)
{
  if (fVerbose != 0) {
    G4cout << "G4DNAScavengerMaterial::AddAMoleculeAtTime : " << molecule->GetName()
           << " at time : " << G4BestUnit(time, "Time") << G4endl;
  }

  auto counterMap_i = fCounterMap.find(molecule);

  if (counterMap_i == fCounterMap.end()) {
    fCounterMap[molecule][time] = number;
  }
  else if (counterMap_i->second.empty()) {
    counterMap_i->second[time] = number;
  }
  else {
    auto end = counterMap_i->second.rbegin();

    if (end->first <= time
        || fabs(end->first - time) <= G4::MoleculeCounter::FixedTimeComparer::fPrecision) {
      G4double newValue = end->second + number;
      counterMap_i->second[time] = newValue;
      if (newValue != (floor)(fScavengerTable[molecule]))  // protection
      {
        G4String errMsg = "You are trying to add wrong molecule : ";
                        G4cout<< " newValue : "<<newValue<<" " << molecule->GetName()
                          << " at time : " << G4BestUnit(time, "Time")
                          << " with number : " << number
               <<" (floor)(fScavengerTable[molecule]) : "<<(floor)(fScavengerTable[molecule])
                          << " and the final number is not valid." << G4endl;
        G4Exception("AddAMoleculeAtTime", "", FatalErrorInArgument, errMsg);
      }
    }
  }
}

//------------------------------------------------------------------------------

void G4DNAScavengerMaterial::RemoveAMoleculeAtTime(MolType pMolecule, G4double time,
                                                   const G4ThreeVector* /*position*/, int number)
{
  NbMoleculeInTime& nbMolPerTime = fCounterMap[pMolecule];

  if (fVerbose != 0) {
    auto it_ = nbMolPerTime.rbegin();
    G4cout << "G4DNAScavengerMaterial::RemoveAMoleculeAtTime : " << pMolecule->GetName()
           << " at time : " << G4BestUnit(time, "Time")

           << " form : " << it_->second << G4endl;
  }

  if (nbMolPerTime.empty()) {
    Dump();
    G4String errMsg = "You are trying to remove molecule " +
                      pMolecule->GetName() +
                      " from the counter while this kind of molecules has not "
                      "been registered yet";
    G4Exception("G4DNAScavengerMaterial::RemoveAMoleculeAtTime", "", FatalErrorInArgument, errMsg);

    return;
  }

  auto it = nbMolPerTime.rbegin();

  if (it == nbMolPerTime.rend()) {
    it--;

    G4String errMsg = "There was no " + pMolecule->GetName()
                      + " recorded at the time or even before the time asked";
    G4Exception("G4DNAScavengerMaterial::RemoveAMoleculeAtTime", "", FatalErrorInArgument, errMsg);
  }

  G4double finalN = it->second - number;
  if (finalN < 0) {
    Dump();

    G4cout << "fScavengerTable : " << pMolecule->GetName() << " : " << (fScavengerTable[pMolecule])
           << G4endl;

    G4ExceptionDescription errMsg;
    errMsg << "After removal of " << number << " species of "
           << " " << it->second << " " << pMolecule->GetName() << " the final number at time "
           << G4BestUnit(time, "Time") << " is less than zero and so not valid." << G4endl;
    G4cout << " Global time is " << G4BestUnit(G4Scheduler::Instance()->GetGlobalTime(), "Time")
           << ". Previous selected time is " << G4BestUnit(it->first, "Time") << G4endl;
    G4Exception("G4DNAScavengerMaterial::RemoveAMoleculeAtTime", "N_INF_0", FatalException, errMsg);
  }
  nbMolPerTime[time] = finalN;

  if (finalN != (floor)(fScavengerTable[pMolecule]))  // protection
  {
    assert(false);
  }
}

void G4DNAScavengerMaterial::Dump()
{
  auto pConfinedBox = fpChemistryInfo->GetChemistryBoundary();
  auto V = pConfinedBox->Volume();
  for (const auto& it : fCounterMap) {
    auto pReactant = it.first;

    G4cout << " --- > For " << pReactant->GetName() << G4endl;

    for (const auto& it2 : it.second) {
      G4cout << " " << G4BestUnit(it2.first, "Time") << "    "
             << it2.second / (Avogadro * V * 1.0e-6 /*mm3 to L*/) << G4endl;
    }
  }
}

int64_t G4DNAScavengerMaterial::GetNMoleculesAtTime(MolType molecule, G4double time)
{
  if (!fCounterAgainstTime) {
    G4cout << "fCounterAgainstTime == false" << G4endl;
    assert(false);
  }

  G4bool sameTypeOfMolecule = SearchTimeMap(molecule);
  auto output = SearchUpperBoundTime(time, sameTypeOfMolecule);
  if (output < 0) {
    G4ExceptionDescription errMsg;
    errMsg << "N molecules not valid < 0 : " << molecule->GetName() << " N : " << output << G4endl;
    G4Exception("G4DNAScavengerMaterial::GetNMoleculesAtTime", "", FatalErrorInArgument, errMsg);
  }
  return output;
}

G4bool G4DNAScavengerMaterial::SearchTimeMap(MolType molecule)
{
  if (fpLastSearch == nullptr) {
    fpLastSearch = std::make_unique<Search>();
  }
  else {
    if (fpLastSearch->fLowerBoundSet && fpLastSearch->fLastMoleculeSearched->first == molecule) {
      return true;
    }
  }

  auto mol_it = fCounterMap.find(molecule);
  fpLastSearch->fLastMoleculeSearched = mol_it;

  if (mol_it != fCounterMap.end()) {
    fpLastSearch->fLowerBoundTime = fpLastSearch->fLastMoleculeSearched->second.end();
    fpLastSearch->fLowerBoundSet = true;
  }
  else {
    fpLastSearch->fLowerBoundSet = false;
  }

  return false;
}

//------------------------------------------------------------------------------

int64_t G4DNAScavengerMaterial::SearchUpperBoundTime(G4double time, G4bool sameTypeOfMolecule)
{
  auto mol_it = fpLastSearch->fLastMoleculeSearched;
  if (mol_it == fCounterMap.end()) {
    return 0;
  }

  NbMoleculeInTime& timeMap = mol_it->second;
  if (timeMap.empty()) {
    return 0;
  }

  if (sameTypeOfMolecule) {
    if (fpLastSearch->fLowerBoundSet && fpLastSearch->fLowerBoundTime != timeMap.end()) {
      if (fpLastSearch->fLowerBoundTime->first < time) {
        auto upperToLast = fpLastSearch->fLowerBoundTime;
        upperToLast++;

        if (upperToLast == timeMap.end()) {
          return fpLastSearch->fLowerBoundTime->second;
        }

        if (upperToLast->first > time) {
          return fpLastSearch->fLowerBoundTime->second;
        }
      }
    }
  }

  auto up_time_it = timeMap.upper_bound(time);

  if (up_time_it == timeMap.end()) {
    auto last_time = timeMap.rbegin();
    return last_time->second;
  }
  if (up_time_it == timeMap.begin()) {
    return 0;
  }

  up_time_it--;

  fpLastSearch->fLowerBoundTime = up_time_it;
  fpLastSearch->fLowerBoundSet = true;

  return fpLastSearch->fLowerBoundTime->second;
}

void G4DNAScavengerMaterial::WaterEquilibrium()
{
  auto convertFactor = Avogadro * fpChemistryInfo->GetChemistryBoundary()->Volume() / liter;
  G4double kw = 1.01e-14;
  fScavengerTable[fHOm] = (kw / ((G4double)fScavengerTable[fH3Op] / convertFactor)) * convertFactor;
  G4cout << "pH : " << GetpH() << G4endl;
  return;
}

void G4DNAScavengerMaterial::SetpH(const G4int& ph)
{
  auto volume = fpChemistryInfo->GetChemistryBoundary()->Volume();
  fScavengerTable[fH3Op] = floor(Avogadro * std::pow(10, -ph) * volume / liter);
  fScavengerTable[fHOm] = floor(Avogadro * std::pow(10, -(14 - ph)) * volume / liter);
}

G4double G4DNAScavengerMaterial::GetpH()
{
  G4double volumeInLiter = fpChemistryInfo->GetChemistryBoundary()->Volume() / liter;
  G4double Cion = (G4double)fScavengerTable[fH3Op] / (Avogadro * volumeInLiter);
  G4double pH = std::log10(Cion);
  // G4cout<<"OH- : "<<fScavengerTable[fHOm]<<"   H3O+ : "<<fScavengerTable[fH3Op]<<"   pH :
  // "<<-pH<<G4endl;
  if (fScavengerTable[fH3Op] < 0)  // protect me
  {
    G4Exception("G4DNAScavengerMaterial::GetpH()", "G4DNAScavengerMaterial001", JustWarning,
                "H3O+ < 0");
    fScavengerTable[fH3Op] = 0;
  }
  if (fScavengerTable[fHOm] < 0)  // protect me
  {
    G4Exception("G4DNAScavengerMaterial::GetpH()", "G4DNAScavengerMaterial001", JustWarning,
                "HO- < 0");
    fScavengerTable[fHOm] = 0;
  }
  return -pH;
}

G4bool G4DNAScavengerMaterial::SetEquilibrium(const G4DNAMolecularReactionData* pReaction,
                                              G4double time)
{
  for(auto& it : fEquilibriumProcesses)
  {
    it.second->SetGlobalTime(time);
    it.second->SetEquilibrium(pReaction);
    if(it.second->IsStatusChanged()) return true;
  }
  return false;
}

void G4DNAScavengerMaterial::ResetEquilibrium()
{
  for(auto& it : fEquilibriumProcesses)
  {
    it.second->Reset();
  }
}

G4bool G4DNAScavengerMaterial::IsEquilibrium(const G4int& reactionType) const
{
  auto reaction = fEquilibriumProcesses.find(reactionType);
  if(reaction == fEquilibriumProcesses.end())
  {
    return true;
  }else
  {
    return (reaction->second->GetEquilibriumStatus());
  }

}