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

// -------------------------------------------------------------------
//
// GEANT4 Class file
//
//
// File name:     G4NistManager
//
// Author:        Vladimir Ivanchenko
//
// Creation date: 23.12.2004
//
//
// -------------------------------------------------------------------
//
// Class Description:
//
// Element data from the NIST DB on Atomic Weights and Isotope Compositions
// http://physics.nist.gov/PhysRefData/Compositions/index.html

#include "G4NistManager.hh"
#include "G4AutoLock.hh"
#include "G4NistMessenger.hh"


namespace
{
G4Mutex nistManagerMutex = G4MUTEX_INITIALIZER;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

G4NistManager* G4NistManager::Instance()
{
  static G4NistManager manager;
  return &manager;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4NistManager::~G4NistManager()
{
  delete messenger;
  delete matBuilder;
  delete elmBuilder;
  delete fICRU90;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4Material* G4NistManager::BuildMaterialWithNewDensity(const G4String& name,
  const G4String& basename, G4double density, G4double temperature, G4double pressure)
{
  G4Material* bmat = FindOrBuildMaterial(name);
  if (bmat != nullptr) {
    G4cout << "G4NistManager::BuildMaterialWithNewDensity ERROR: " << G4endl;
    G4cout << " New material <" << name << "> cannot be built because material"
           << " with the same name already exists." << G4endl;
    G4Exception("G4NistManager::BuildMaterialWithNewDensity()", "mat101", FatalException,
      "Wrong material name");
    return nullptr;
  }
  bmat = FindOrBuildMaterial(basename);
  if (bmat == nullptr) {
    G4cout << "G4NistManager::BuildMaterialWithNewDensity ERROR: " << G4endl;
    G4cout << " New material <" << name << "> cannot be built because " << G4endl;
    G4cout << " base material <" << basename << "> does not exist." << G4endl;
    G4Exception("G4NistManager::BuildMaterialWithNewDensity()", "mat102", FatalException,
      "Wrong material name");
    return nullptr;
  }
  G4double dens = density;
  G4double temp = temperature;
  G4double pres = pressure;
  if (dens == 0.0) {
    dens = bmat->GetDensity();
    temp = bmat->GetTemperature();
    pres = bmat->GetPressure();
  }
  auto mat = new G4Material(name, dens, bmat, bmat->GetState(), temp, pres);
  return mat;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4NistManager::PrintElement(const G4String& symbol) const
{
  if (symbol == "all") {
    elmBuilder->PrintElement(0);
  }
  else {
    elmBuilder->PrintElement(elmBuilder->GetZ(symbol));
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4NistManager::PrintG4Element(const G4String& name) const
{
  for (auto const & elm : *G4Element::GetElementTable()) {
    if (name == elm->GetName() || "all" == name) {
      G4cout << *elm << G4endl;
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4NistManager::PrintG4Material(const G4String& name) const
{
  for (auto const & mat : *G4Material::GetMaterialTable()) {
    if (name == mat->GetName() || "all" == name) {
      G4cout << *mat << G4endl;
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4NistManager::SetVerbose(G4int val)
{
  verbose = val;
  elmBuilder->SetVerbose(val);
  matBuilder->SetVerbose(val);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4NistManager::G4NistManager()
{
  nElements = 0;
  nMaterials = 0;
  verbose = 0;

  theMaterialTable = G4Material::GetMaterialTable();
  theElementTable = G4Element::GetElementTable();
  theIsotopeTable = G4Isotope::GetIsotopeTable();

  elmBuilder = new G4NistElementBuilder(verbose);
  matBuilder = new G4NistMaterialBuilder(elmBuilder, verbose);

  messenger = new G4NistMessenger(this);
  g4pow = G4Pow::GetInstance();

  // compute frequently used values for mean atomic numbers
  for (G4int j = 1; j < 101; ++j) {
    G4double A = elmBuilder->GetAtomicMassAmu(j);
    POWERA27[j] = std::pow(A, 0.27);
    LOGAZ[j] = std::log(A);
  }
  POWERA27[0] = 1.0;
  LOGAZ[0] = 0.0;
  fICRU90 = nullptr;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4ICRU90StoppingData* G4NistManager::GetICRU90StoppingData()
{
  if (fICRU90 == nullptr) {
    G4AutoLock l(&nistManagerMutex);
    if (fICRU90 == nullptr) {
      fICRU90 = new G4ICRU90StoppingData();
    }
    l.unlock();
  }
  return fICRU90;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4NistManager::SetDensityEffectCalculatorFlag(const G4String& mname, G4bool val)
{
  if (mname == "all") {
    for (auto mat : materials) {
      SetDensityEffectCalculatorFlag(mat, val);
    }
  }
  else {
    G4Material* mat = FindMaterial(mname);
    SetDensityEffectCalculatorFlag(mat, val);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4NistManager::SetDensityEffectCalculatorFlag(G4Material* mat, G4bool val)
{
  if (mat != nullptr) {
    mat->ComputeDensityEffectOnFly(val);
  }
}
