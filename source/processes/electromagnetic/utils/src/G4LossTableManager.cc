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
// -------------------------------------------------------------------
//
// GEANT4 Class file
//
//
// File name:     G4LossTableManager
//
// Author:        Vladimir Ivanchenko
//
// Creation date: 03.01.2002
//
// Modifications: by V.Ivanchenko
//
//
// Class Description:
//
// -------------------------------------------------------------------
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#include "G4LossTableManager.hh"
#include "G4SystemOfUnits.hh"

#include "G4VMultipleScattering.hh"
#include "G4VEmProcess.hh"
#include "G4VXRayModel.hh"

#include "G4EmParameters.hh"
#include "G4EmSaturation.hh"
#include "G4EmConfigurator.hh"
#include "G4ElectronIonPair.hh"
#include "G4NIELCalculator.hh"
#include "G4EmCorrections.hh"
#include "G4LossTableBuilder.hh"
#include "G4VAtomDeexcitation.hh"
#include "G4VSubCutProducer.hh"
#include "G4VXRayModel.hh"

#include "G4PhysicsTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ProcessManager.hh"
#include "G4Electron.hh"
#include "G4Proton.hh"
#include "G4ProductionCutsTable.hh"
#include "G4PhysicsTableHelper.hh"
#include "G4EmTableType.hh"
#include "G4Region.hh"
#include "G4PhysicalConstants.hh"

#include "G4Gamma.hh"
#include "G4Positron.hh"
#include "G4OpticalPhoton.hh"
#include "G4Neutron.hh"
#include "G4MuonPlus.hh"
#include "G4MuonMinus.hh"
#include "G4GenericIon.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

static std::once_flag applyOnce;
G4ThreadLocal G4LossTableManager* G4LossTableManager::instance = nullptr;

G4LossTableManager* G4LossTableManager::Instance()
{
  if(nullptr == instance) {
    static G4ThreadLocalSingleton<G4LossTableManager> inst;
    instance = inst.Instance();
  }
  return instance;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

G4LossTableManager::~G4LossTableManager()
{
  for (auto const & p : loss_vector) { delete p; }
  for (auto const & p : msc_vector) { delete p; }
  for (auto const & p : emp_vector) { delete p; }
  for (auto const & p : p_vector) { delete p; }
  for (auto const & p : xray_vector) { delete p; }

  std::size_t mod = mod_vector.size();
  std::size_t fmod = fmod_vector.size();
  for (std::size_t a=0; a<mod; ++a) {
    if( nullptr != mod_vector[a] ) { 
      for (std::size_t b=0; b<fmod; ++b) {
        if((G4VEmModel*)(fmod_vector[b]) == mod_vector[a]) {
          fmod_vector[b] = nullptr;
        }
      }
      delete mod_vector[a]; 
      mod_vector[a] = nullptr;
    }
  }
  for (auto const & p : fmod_vector) { delete p; }

  Clear();
  delete tableBuilder;
  delete emCorrections;
  delete emConfigurator;
  delete emElectronIonPair;
  delete nielCalculator;
  delete atomDeexcitation;
  delete subcutProducer;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

G4LossTableManager::G4LossTableManager()
{
  theParameters = G4EmParameters::Instance();
  theElectron = G4Electron::Electron();
  
  // only one thread is the master 
  std::call_once(applyOnce, [this]() { isMaster = true; });
  verbose = isMaster ? theParameters->Verbose() : theParameters->WorkerVerbose();

  tableBuilder = new G4LossTableBuilder(isMaster);
  emCorrections = new G4EmCorrections(verbose);

  std::size_t n = 70;
  loss_vector.reserve(n);
  part_vector.reserve(n);
  base_part_vector.reserve(n);
  dedx_vector.reserve(n);
  range_vector.reserve(n);
  inv_range_vector.reserve(n);
  tables_are_built.reserve(n);
  isActive.reserve(n);
  msc_vector.reserve(10);
  emp_vector.reserve(16);
  mod_vector.reserve(150);
  fmod_vector.reserve(60);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::Clear()
{
  all_tables_are_built = false;
  currentLoss = nullptr;
  currentParticle = nullptr;
  if(n_loss) {
    dedx_vector.clear();
    range_vector.clear();
    inv_range_vector.clear();
    loss_map.clear();
    loss_vector.clear();
    part_vector.clear();
    base_part_vector.clear();
    tables_are_built.clear();
    isActive.clear();
    n_loss = 0;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::Register(G4VEnergyLossProcess* p)
{
  if (nullptr == p) { return; }
  for (G4int i=0; i<n_loss; ++i) {
    if(loss_vector[i] == p) { return; }
  }
  if(verbose > 1) {
    G4cout << "G4LossTableManager::Register G4VEnergyLossProcess : " 
           << p->GetProcessName() << "  idx= " << n_loss << G4endl;
  }
  ++n_loss;
  loss_vector.push_back(p);
  part_vector.push_back(nullptr);
  base_part_vector.push_back(nullptr);
  dedx_vector.push_back(nullptr);
  range_vector.push_back(nullptr);
  inv_range_vector.push_back(nullptr);
  tables_are_built.push_back(false);
  isActive.push_back(true);
  all_tables_are_built = false;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::ResetParameters()
{
  // initialisation once per run
  if (!resetParam) { return; }
  resetParam = false;
  startInitialisation = true;
  verbose = theParameters->Verbose();
  if(!isMaster) {
    verbose = theParameters->WorkerVerbose();
  } else {
    if(verbose > 0) { theParameters->Dump(); }
  }

  tableBuilder->InitialiseBaseMaterials(); 
  if (nullptr != nielCalculator) { nielCalculator->Initialise(); } 

  emCorrections->SetVerbose(verbose); 
  if(nullptr != emConfigurator) { emConfigurator->SetVerbose(verbose); };
  if(nullptr != emElectronIonPair) { emElectronIonPair->SetVerbose(verbose); };
  if(nullptr != atomDeexcitation) {
    atomDeexcitation->SetVerboseLevel(verbose);
    atomDeexcitation->InitialiseAtomicDeexcitation();
  }
  if (1 < verbose) {
    G4cout << "====== G4LossTableManager::ResetParameters " 
           << " Nloss=" << loss_vector.size()
	   << " run=" << run << " master=" << isMaster
	   << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::DeRegister(G4VEnergyLossProcess* p)
{
  if (nullptr == p) { return; }
  for (G4int i=0; i<n_loss; ++i) {
    if(loss_vector[i] == p) { 
      loss_vector[i] = nullptr;
      break;
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::Register(G4VMultipleScattering* p)
{
  if (nullptr == p) { return; }
  std::size_t n = msc_vector.size();
  for (std::size_t i=0; i<n; ++i) {
    if(msc_vector[i] == p) { return; }
  }
  if(verbose > 1) {
    G4cout << "G4LossTableManager::Register G4VMultipleScattering : " 
           << p->GetProcessName() << "  idx= " << msc_vector.size() << G4endl;
  }
  msc_vector.push_back(p);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::DeRegister(G4VMultipleScattering* p)
{
  if (nullptr == p) { return; }
  std::size_t msc = msc_vector.size();
  for (std::size_t i=0; i<msc; ++i) {
    if(msc_vector[i] == p) { 
      msc_vector[i] = nullptr;
      break;
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::Register(G4VEmProcess* p)
{
  if (nullptr == p) { return; }
  std::size_t n = emp_vector.size();
  for (std::size_t i=0; i<n; ++i) {
    if(emp_vector[i] == p) { return; }
  }
  if(verbose > 1) {
    G4cout << "G4LossTableManager::Register G4VEmProcess : " 
           << p->GetProcessName() << "  idx= " << emp_vector.size() << G4endl;
  }
  emp_vector.push_back(p);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::DeRegister(G4VEmProcess* p)
{
  if (nullptr == p) { return; }
  std::size_t emp = emp_vector.size();
  for (std::size_t i=0; i<emp; ++i) {
    if(emp_vector[i] == p) { 
      emp_vector[i] = nullptr; 
      break;
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::Register(G4VProcess* p)
{
  if (nullptr == p) { return; }
  std::size_t n = p_vector.size();
  for (std::size_t i=0; i<n; ++i) {
    if(p_vector[i] == p) { return; }
  }
  if(verbose > 1) {
    G4cout << "G4LossTableManager::Register G4VProcess : " 
           << p->GetProcessName() << "  idx= " << p_vector.size() << G4endl;
  }
  p_vector.push_back(p);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::DeRegister(G4VProcess* p)
{
  if (nullptr == p) { return; }
  std::size_t emp = p_vector.size();
  for (std::size_t i=0; i<emp; ++i) {
    if(p_vector[i] == p) { 
      p_vector[i] = nullptr;
      break;
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::Register(G4VEmModel* p)
{
  if (nullptr == p) { return; }
  std::size_t n = mod_vector.size();
  for (std::size_t i=0; i<n; ++i) { if (mod_vector[i] == p) { return; } }
  mod_vector.push_back(p);
  if(verbose > 1) {
    G4cout << "G4LossTableManager::Register G4VEmModel : " 
           << p->GetName() << "  " << p << "  " << mod_vector.size() << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::DeRegister(G4VEmModel* p)
{
  std::size_t n = mod_vector.size();
  for (std::size_t i=0; i<n; ++i) {
    if(mod_vector[i] == p) { 
      mod_vector[i] = nullptr; 
      break;
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::Register(G4VEmFluctuationModel* p)
{
  if (nullptr == p) { return; }
  std::size_t n = fmod_vector.size();
  for (std::size_t i=0; i<n; ++i) { if (fmod_vector[i] == p) { return; } }
  fmod_vector.push_back(p);
  if(verbose > 1) {
    G4cout << "G4LossTableManager::Register G4VEmFluctuationModel : " 
           << p->GetName() << "  " << fmod_vector.size() << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::DeRegister(G4VEmFluctuationModel* p)
{
  std::size_t n = fmod_vector.size();
  for (std::size_t i=0; i<n; ++i) {
    if(fmod_vector[i] == p) {
      fmod_vector[i] = nullptr;
      break;
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::Register(G4VXRayModel* p)
{
  if (nullptr == p) { return; }
  std::size_t n = xray_vector.size();
  for (std::size_t i=0; i<n; ++i) { if (xray_vector[i] == p) { return; } }
  xray_vector.push_back(p);
  if (verbose > 1) {
    G4cout << "G4LossTableManager::Register G4VXRayModel : " 
           << p->GetName() << "  " << xray_vector.size() << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::DeRegister(G4VXRayModel* p)
{
  std::size_t n = xray_vector.size();
  for (std::size_t i=0; i<n; ++i) {
    if (xray_vector[i] == p) {
      xray_vector[i] = nullptr;
      break;
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::RegisterExtraParticle(
     const G4ParticleDefinition* part,
     G4VEnergyLossProcess* p)
{ 
  if (nullptr == p || nullptr == part) { return; }
  for (G4int i=0; i<n_loss; ++i) {
    if(loss_vector[i] == p) { return; }
  }
  if(verbose > 1) {
    G4cout << "G4LossTableManager::RegisterExtraParticle "
           << part->GetParticleName() << "  G4VEnergyLossProcess : " 
           << p->GetProcessName() << "  idx= " << n_loss << G4endl;
  }
  ++n_loss;
  loss_vector.push_back(p);
  part_vector.push_back(part);
  base_part_vector.push_back(p->BaseParticle());
  dedx_vector.push_back(nullptr);
  range_vector.push_back(nullptr);
  inv_range_vector.push_back(nullptr);
  tables_are_built.push_back(false);
  all_tables_are_built = false;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VEnergyLossProcess* 
G4LossTableManager::GetEnergyLossProcess(const G4ParticleDefinition *aParticle)
{
  if(aParticle != currentParticle) {
    currentParticle = aParticle;
    std::map<PD,G4VEnergyLossProcess*,std::less<PD> >::const_iterator pos;
    if ((pos = loss_map.find(aParticle)) != loss_map.end()) {
      currentLoss = (*pos).second;
    } else {
      currentLoss = nullptr;
      if(0.0 != aParticle->GetPDGCharge() && 
	 (pos = loss_map.find(theGenericIon)) != loss_map.end()) {
	currentLoss = (*pos).second;
      }
    }
  }
  return currentLoss;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void 
G4LossTableManager::PreparePhysicsTable(const G4ParticleDefinition* particle,
                                        G4VEnergyLossProcess* p)
{
  if (1 < verbose) {
    G4cout << "G4LossTableManager::PreparePhysicsTable for " 
           << particle->GetParticleName() 
           << " and " << p->GetProcessName() << " run= " << run 
           << "   loss_vector " << loss_vector.size()
	   << " run=" << run << " master=" << isMaster
	   << G4endl;
  }

  // start initialisation for the first run
  if ( -1 == run ) {
    if (nullptr != emConfigurator) {
      emConfigurator->PrepareModels(particle, p);
    }

    // initialise particles for given process
    for (G4int j=0; j<n_loss; ++j) {
      if (p == loss_vector[j] && nullptr == part_vector[j]) { 
        part_vector[j] = particle;
        if (particle->GetParticleName() == "GenericIon") {
          theGenericIon = particle;
        }
      }
    }
  }
  ResetParameters();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void 
G4LossTableManager::PreparePhysicsTable(const G4ParticleDefinition* particle,
                                        G4VEmProcess* p)
{
  if (1 < verbose) {
    G4cout << "G4LossTableManager::PreparePhysicsTable for " 
           << particle->GetParticleName() 
           << " and " << p->GetProcessName()
	   << " run=" << run << " master=" << isMaster
	   << G4endl;
  }

  // start initialisation for the first run
  if( -1 == run ) {
    if (nullptr != emConfigurator) { emConfigurator->PrepareModels(particle, p); }
  }

  ResetParameters();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void 
G4LossTableManager::PreparePhysicsTable(const G4ParticleDefinition* particle,
                                        G4VMultipleScattering* p)
{
  if (1 < verbose) {
    G4cout << "G4LossTableManager::PreparePhysicsTable for " 
           << particle->GetParticleName() 
           << " and " << p->GetProcessName()
	   << " run=" << run << " master=" << isMaster
	   << G4endl;
  }

  // start initialisation for the first run
  if ( -1 == run ) {
    if (nullptr != emConfigurator) {
      emConfigurator->PrepareModels(particle, p);
    }
  } 
  
  ResetParameters();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void 
G4LossTableManager::BuildPhysicsTable(const G4ParticleDefinition*)
{
  if(-1 == run && startInitialisation) {
    if (nullptr != emConfigurator) { emConfigurator->Clear(); }
  }
  if (startInitialisation) { resetParam = true; }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::LocalPhysicsTables(
     const G4ParticleDefinition* aParticle,
     G4VEnergyLossProcess* p)
{
  if (1 < verbose) {
    G4cout << "### G4LossTableManager::LocalPhysicsTable() for "
           << aParticle->GetParticleName()
           << " and process " << p->GetProcessName()
           << G4endl;
  }

  if(-1 == run && startInitialisation) {
    if (nullptr != emConfigurator) { emConfigurator->Clear(); }
    firstParticle = aParticle; 
  }

  if (startInitialisation) {
    ++run;
    if (1 < verbose) {
      G4cout << "===== G4LossTableManager::LocalPhysicsTable() for run "
             << run << " =====" << G4endl;
    }
    currentParticle = nullptr;
    startInitialisation = false;
    resetParam = true;
    for (G4int i=0; i<n_loss; ++i) {
      if (nullptr != loss_vector[i]) {
        tables_are_built[i] = false;
      } else {
        tables_are_built[i] = true;
        part_vector[i] = nullptr;
      }
    }
  }

  all_tables_are_built= true;
  for (G4int i=0; i<n_loss; ++i) {
    if(p == loss_vector[i]) {
      tables_are_built[i] = true;
      isActive[i] = true;
      part_vector[i] = p->Particle(); 
      base_part_vector[i] = p->BaseParticle(); 
      dedx_vector[i] = p->DEDXTable();
      range_vector[i] = p->RangeTableForLoss();
      inv_range_vector[i] = p->InverseRangeTable();
      if(0 == run && p->IsIonisationProcess()) {
        loss_map[part_vector[i]] = p;
      }

      if(1 < verbose) { 
        G4cout << i <<".   "<< p->GetProcessName(); 
        if(part_vector[i]) {
          G4cout << "  for "  << part_vector[i]->GetParticleName();
        }
        G4cout << "  active= " << isActive[i]
               << "  table= " << tables_are_built[i]
               << "  isIonisation= " << p->IsIonisationProcess()
               << G4endl;
      }
      break;
    } else if(!tables_are_built[i]) {
      all_tables_are_built = false;
    }
  }

  if(1 < verbose) {
    G4cout << "### G4LossTableManager::LocalPhysicsTable end"
           << G4endl;
  }
  if(all_tables_are_built) { 
    if(1 < verbose) {
      G4cout << "%%%%% All dEdx and Range tables for worker are ready for run "
             << run << " %%%%%" << G4endl;
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::BuildPhysicsTable(
     const G4ParticleDefinition* aParticle,
     G4VEnergyLossProcess* p)
{
  if(1 < verbose) {
    G4cout << "### G4LossTableManager::BuildPhysicsTable() for "
           << aParticle->GetParticleName()
           << " and process " << p->GetProcessName() << G4endl;
  }
  // clear configurator
  if(-1 == run && startInitialisation) {
    if( nullptr != emConfigurator) { emConfigurator->Clear(); }
    firstParticle = aParticle; 
  }
  if(startInitialisation) {
    ++run;
    resetParam = true;
    startInitialisation = false;
    if(1 < verbose) {
      G4cout << "===== G4LossTableManager::BuildPhysicsTable() for run "
             << run << " ===== " << atomDeexcitation << G4endl;
    }
    currentParticle = nullptr;
    all_tables_are_built = false;

    for (G4int i=0; i<n_loss; ++i) {
      G4VEnergyLossProcess* el = loss_vector[i];

      if(nullptr != el) {
        isActive[i] = true;
        part_vector[i] = el->Particle(); 
        base_part_vector[i] = el->BaseParticle(); 
        tables_are_built[i] = false;  
        if(1 < verbose) { 
          G4cout << i <<".   "<< el->GetProcessName();
          if(el->Particle()) {
            G4cout << "  for "  << el->Particle()->GetParticleName();
          }
          G4cout << "  active= " << isActive[i]
                 << "  table= " << tables_are_built[i]
                 << "  isIonisation= " << el->IsIonisationProcess();
          if(base_part_vector[i]) { 
            G4cout << "  base particle " 
                   << base_part_vector[i]->GetParticleName();
          }
          G4cout << G4endl;
        }
      } else {
        tables_are_built[i] = true;
        part_vector[i] = nullptr;
        isActive[i] = false;
      }
    }
  }

  if (all_tables_are_built) { 
    theParameters->SetIsPrintedFlag(true);
    return; 
  }

  // Build tables for given particle
  all_tables_are_built = true;

  for(G4int i=0; i<n_loss; ++i) {
    if(p == loss_vector[i] && !tables_are_built[i] && nullptr == base_part_vector[i]) {
      const G4ParticleDefinition* curr_part = part_vector[i];
      if(1 < verbose) {
        G4cout << "### Build Table for " << p->GetProcessName()
               << " and " << curr_part->GetParticleName() 
               << "  " << tables_are_built[i] << "  " << base_part_vector[i] 
               << G4endl;
      }
      G4VEnergyLossProcess* curr_proc = BuildTables(curr_part);
      if(curr_proc) { 
        CopyTables(curr_part, curr_proc); 
        if(p == curr_proc && 0 == run && p->IsIonisationProcess()) { 
          loss_map[aParticle] = p; 
          //G4cout << "G4LossTableManager::BuildPhysicsTable: " 
          //     << aParticle->GetParticleName()
          //         << " added to map " << p <<  G4endl;
        }
      }
    }
    if ( !tables_are_built[i] ) { all_tables_are_built = false; }
  }
  if(1 < verbose) {
    G4cout << "### G4LossTableManager::BuildPhysicsTable end: "
           << "all_tables_are_built= " << all_tables_are_built << " "
           << aParticle->GetParticleName() << " proc: " << p << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::CopyTables(const G4ParticleDefinition* part,
                                    G4VEnergyLossProcess* base_proc)
{
  for (G4int j=0; j<n_loss; ++j) {

    G4VEnergyLossProcess* proc = loss_vector[j];

    if (!tables_are_built[j] && part == base_part_vector[j]) {
      tables_are_built[j] = true;
      // for base particle approach only ionisation table should be used
      proc->SetDEDXTable(base_proc->IonisationTable(),fRestricted);
      proc->SetDEDXTable(base_proc->DEDXunRestrictedTable(),fTotal);
      proc->SetCSDARangeTable(base_proc->CSDARangeTable());
      proc->SetRangeTableForLoss(base_proc->RangeTableForLoss());
      proc->SetInverseRangeTable(base_proc->InverseRangeTable());
      proc->SetLambdaTable(base_proc->LambdaTable());
      if(proc->IsIonisationProcess()) { 
        range_vector[j] = base_proc->RangeTableForLoss();
        inv_range_vector[j] = base_proc->InverseRangeTable();
        loss_map[part_vector[j]] = proc; 
        //G4cout << "G4LossTableManager::CopyTable " 
        //       << part_vector[j]->GetParticleName()
        //       << " added to map " << proc << G4endl;
      }
      if (1 < verbose) {
         G4cout << "   CopyTables for " << proc->GetProcessName()
                << " for " << part_vector[j]->GetParticleName()
                << " base_part= " << part->GetParticleName()
                << " tables are assigned"
                << G4endl;
      }
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

G4VEnergyLossProcess* G4LossTableManager::BuildTables(
                      const G4ParticleDefinition* aParticle)
{
  if(1 < verbose) {
    G4cout << "   G4LossTableManager::BuildTables(part) for "
           << aParticle->GetParticleName() << G4endl;
  }

  std::vector<G4PhysicsTable*> t_list;  
  std::vector<G4VEnergyLossProcess*> loss_list;
  std::vector<G4bool> build_flags;
  G4VEnergyLossProcess* em = nullptr;
  G4VEnergyLossProcess* p = nullptr;
  G4int iem = 0;
  G4PhysicsTable* dedx = nullptr;
  G4int i;

  G4ProcessVector* pvec = 
    aParticle->GetProcessManager()->GetProcessList();
  G4int nvec = (G4int)pvec->size();

  for (i=0; i<n_loss; ++i) {
    p = loss_vector[i];
    if (nullptr != p) {
      G4bool yes = (aParticle == part_vector[i]);

      // possible case of process sharing between particle/anti-particle
      if(!yes) {
        auto ptr = static_cast<G4VProcess*>(p);
        for(G4int j=0; j<nvec; ++j) {
          //G4cout << "j= " << j << " " << (*pvec)[j] << " " << ptr << G4endl;
          if(ptr == (*pvec)[j]) {
            yes = true;
            break;
          }
        }
      }      
      // process belong to this particle
      if(yes && isActive[i]) {
        if (p->IsIonisationProcess() || !em) {
          em = p;
          iem= i;
        }
        // tables may be shared between particle/anti-particle
        G4bool val = false;
        if (!tables_are_built[i]) {
          val = true;
          dedx = p->BuildDEDXTable(fRestricted);
          //G4cout << "===Build DEDX table for " << p->GetProcessName()
          // << " idx= " << i << " dedx:" << dedx << " " << dedx->length() << G4endl;
          p->SetDEDXTable(dedx,fRestricted);
          tables_are_built[i] = true;
        } else {
          dedx = p->DEDXTable();
        }
        t_list.push_back(dedx);
        loss_list.push_back(p);
        build_flags.push_back(val);
      }
    }
  }

  G4int n_dedx = (G4int)t_list.size();
  if (0 == n_dedx || !em) {
    G4cout << "G4LossTableManager WARNING: no DEDX processes for " 
           << aParticle->GetParticleName() << G4endl;
    return nullptr;
  }
  G4int nSubRegions = em->NumberOfSubCutoffRegions();

  if (1 < verbose) {
    G4cout << "     Start to build the sum of " << n_dedx << " processes"
           << " iem= " << iem << " em= " << em->GetProcessName()
           << " buildCSDARange= " << theParameters->BuildCSDARange()
           << " nSubRegions= " << nSubRegions;
    if(subcutProducer) { 
      G4cout << " SubCutProducer " << subcutProducer->GetName(); 
    }
    G4cout << G4endl;
  }
  // do not build tables if producer class is defined
  if(subcutProducer) { nSubRegions = 0; }

  dedx = em->DEDXTable();
  em->SetDEDXTable(dedx, fIsIonisation);

  if (1 < n_dedx) {
    dedx = nullptr;
    dedx = G4PhysicsTableHelper::PreparePhysicsTable(dedx);
    tableBuilder->BuildDEDXTable(dedx, t_list);
    em->SetDEDXTable(dedx, fRestricted);
  }

  dedx_vector[iem] = dedx;

  G4PhysicsTable* range = em->RangeTableForLoss();
  if(!range) range  = G4PhysicsTableHelper::PreparePhysicsTable(range);
  range_vector[iem] = range;

  G4PhysicsTable* invrange = em->InverseRangeTable();
  if(!invrange) invrange = G4PhysicsTableHelper::PreparePhysicsTable(invrange);
  inv_range_vector[iem]  = invrange;

  tableBuilder->BuildRangeTable(dedx, range);
  tableBuilder->BuildInverseRangeTable(range, invrange);

  em->SetRangeTableForLoss(range);
  em->SetInverseRangeTable(invrange);

  std::vector<G4PhysicsTable*> listCSDA;

  for (i=0; i<n_dedx; ++i) {
    p = loss_list[i];
    if(build_flags[i]) {
      p->SetLambdaTable(p->BuildLambdaTable(fRestricted));
    }
    if(theParameters->BuildCSDARange()) { 
      dedx = p->BuildDEDXTable(fTotal);
      p->SetDEDXTable(dedx,fTotal);
      listCSDA.push_back(dedx); 
    }     
  }

  if(theParameters->BuildCSDARange()) {
    G4PhysicsTable* dedxCSDA = em->DEDXunRestrictedTable();
    if (1 < n_dedx) {
      dedxCSDA = G4PhysicsTableHelper::PreparePhysicsTable(dedxCSDA);
      tableBuilder->BuildDEDXTable(dedxCSDA, listCSDA);
      em->SetDEDXTable(dedxCSDA,fTotal);
    }
    G4PhysicsTable* rCSDA = em->CSDARangeTable();
    if(!rCSDA) { rCSDA = G4PhysicsTableHelper::PreparePhysicsTable(rCSDA); }
    tableBuilder->BuildRangeTable(dedxCSDA, rCSDA);
    em->SetCSDARangeTable(rCSDA);
  }

  if (1 < verbose) {
    G4cout << "G4LossTableManager::BuildTables: Tables are built for "
           << aParticle->GetParticleName()
           << "; ionisation process: " << em->GetProcessName()
           << "  " << em
           << G4endl;
  }
  return em;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::ParticleHaveNoLoss(
     const G4ParticleDefinition* aParticle)
{
  G4ExceptionDescription ed;
  ed << "Energy loss process not found for " << aParticle->GetParticleName() 
     << " !";
  G4Exception("G4LossTableManager::ParticleHaveNoLoss", "em0001",
              FatalException, ed);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::SetVerbose(G4int val)
{
  verbose = val;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

const std::vector<G4VEnergyLossProcess*>& 
G4LossTableManager::GetEnergyLossProcessVector()
{
  return loss_vector;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

const std::vector<G4VEmProcess*>& G4LossTableManager::GetEmProcessVector()
{
  return emp_vector;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

const std::vector<G4VMultipleScattering*>& 
G4LossTableManager::GetMultipleScatteringVector()
{
  return msc_vector;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4EmSaturation* G4LossTableManager::EmSaturation()
{
  return theParameters->GetEmSaturation();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4EmConfigurator* G4LossTableManager::EmConfigurator()
{
  if(!emConfigurator) {
    emConfigurator = new G4EmConfigurator(verbose); 
  }
  return emConfigurator;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4ElectronIonPair* G4LossTableManager::ElectronIonPair()
{
  if(!emElectronIonPair) { 
    emElectronIonPair = new G4ElectronIonPair(verbose);
  }
  return emElectronIonPair;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4LossTableManager::SetNIELCalculator(G4NIELCalculator* ptr)
{
  if(nullptr != ptr && ptr != nielCalculator) {
    delete nielCalculator;
    nielCalculator = ptr;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4NIELCalculator* G4LossTableManager::NIELCalculator()
{
  if(!nielCalculator) { 
    nielCalculator = new G4NIELCalculator(nullptr, verbose); 
  }
  return nielCalculator;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
 
void G4LossTableManager::SetAtomDeexcitation(G4VAtomDeexcitation* p)
{
  if(atomDeexcitation != p) {
    delete atomDeexcitation;
    atomDeexcitation = p;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

void G4LossTableManager::SetSubCutProducer(G4VSubCutProducer* p) 
{
  if(subcutProducer != p) {
    delete subcutProducer;
    subcutProducer = p;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4LossTableManager::PrintEWarning(G4String tit, G4double /*val*/)
{
  G4String ss = "G4LossTableManager::" + tit; 
  G4ExceptionDescription ed;
  /*
  ed << "Parameter is out of range: " << val 
     << " it will have no effect!\n" << " ## " 
     << " nbins= " << nbinsLambda 
     << " nbinsPerDecade= " << nbinsPerDecade 
     << " Emin(keV)= " << minKinEnergy/keV 
     << " Emax(GeV)= " << maxKinEnergy/GeV;
  */
  G4Exception(ss, "em0044", JustWarning, ed);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4LossTableManager::DumpHtml()
{
  // Automatic generation of html documentation page for physics lists
  // List processes and models for the most important
  // particles in descending order of importance
  // NB. for model names with length > 18 characters the .rst file needs
  //  to be edited by hand. Or modify G4EmModelManager::DumpModelList

  char* dirName = std::getenv("G4PhysListDocDir");
  char* physList = std::getenv("G4PhysListName");
  if (dirName && physList) {
    G4String physListName = G4String(physList);
    G4String pathName = G4String(dirName) + "/" + physListName + ".rst";

    std::ofstream outFile;
    outFile.open(pathName);
   
    outFile << physListName << G4endl;
    outFile << std::string(physListName.length(), '=') << G4endl;

    std::vector<G4ParticleDefinition*> particles {
        G4Gamma::Gamma(),
        G4Electron::Electron(),
        G4Positron::Positron(),
        G4Proton::ProtonDefinition(),
        G4MuonPlus::MuonPlusDefinition(),
        G4MuonMinus::MuonMinusDefinition(),
      };
   
    std::vector<G4VEmProcess*> emproc_vector = GetEmProcessVector();
    std::vector<G4VEnergyLossProcess*> enloss_vector = 
      GetEnergyLossProcessVector();
    std::vector<G4VMultipleScattering*> mscat_vector =
      GetMultipleScatteringVector();
    
    for (auto theParticle : particles) {
      outFile << G4endl << "**" << theParticle->GetParticleName()
              << "**" << G4endl << G4endl << " .. code-block:: none" << G4endl;

      G4ProcessManager* pm = theParticle->GetProcessManager();
      G4ProcessVector*  pv = pm->GetProcessList();
      G4int plen = pm->GetProcessListLength();

      for (auto emproc : emproc_vector) {
        for (G4int i = 0; i < plen; ++i) {
          G4VProcess* proc = (*pv)[i];
          if (proc == emproc) {
            outFile << G4endl;
            proc->ProcessDescription(outFile);
            break;
          }
        }
      }

      for (auto mscproc : mscat_vector) {
        for (G4int i = 0; i < plen; ++i) {
          G4VProcess* proc = (*pv)[i];
          if (proc == mscproc) {
            outFile << G4endl;
            proc->ProcessDescription(outFile);
            break;
          }
        }
      }

      for (auto enlossproc : enloss_vector) {
        for (G4int i = 0; i < plen; ++i) {
          G4VProcess* proc = (*pv)[i];
          if (proc == enlossproc) {
            outFile << G4endl;
            proc->ProcessDescription(outFile);
            break;
          }
        }
      }
    }
    outFile.close();
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

