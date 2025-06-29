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
// GEANT4 Class header file
//
//
// File name:     G4LossTableManager
//
// Author:        Vladimir Ivanchenko on base of G4LossTables class
//                and Maria Grazia Pia ideas
//
// Creation date: 03.01.2002
//
// Modifications by V.Ivanchenko  
//
// Class Description:
//
// A utility static class, responsable for the energy loss tables
// for each particle
//
// Energy loss processes have to register their tables with this
// class. The responsibility of creating and deleting the tables
// remains with the energy loss classes.

// -------------------------------------------------------------------
//

#ifndef G4LossTableManager_h
#define G4LossTableManager_h 1

#include <map>
#include <vector>
#include "globals.hh"
#include "G4ThreadLocalSingleton.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4EmParameters.hh"

class G4PhysicsTable;
class G4MaterialCutsCouple;
class G4ParticleDefinition;
class G4Region;
class G4EmSaturation;
class G4EmConfigurator;
class G4ElectronIonPair;
class G4NIELCalculator;
class G4VMultipleScattering;
class G4VEmProcess;
class G4EmCorrections;
class G4LossTableBuilder;
class G4VAtomDeexcitation;
class G4VSubCutProducer;
class G4VXRayModel;

class G4LossTableManager
{

friend class G4ThreadLocalSingleton<G4LossTableManager>;

public:

  static G4LossTableManager* Instance();

  ~G4LossTableManager();

  //-------------------------------------------------
  // initialisation before a new run
  //-------------------------------------------------

  void PreparePhysicsTable(const G4ParticleDefinition* aParticle,
                           G4VEnergyLossProcess* p);

  void PreparePhysicsTable(const G4ParticleDefinition* aParticle,
                           G4VEmProcess* p);

  void PreparePhysicsTable(const G4ParticleDefinition* aParticle,
                           G4VMultipleScattering* p);

  void BuildPhysicsTable(const G4ParticleDefinition* aParticle);

  void BuildPhysicsTable(const G4ParticleDefinition* aParticle, 
                         G4VEnergyLossProcess* p);

  void LocalPhysicsTables(const G4ParticleDefinition* aParticle, 
                          G4VEnergyLossProcess* p);
 
  void DumpHtml();

  //-------------------------------------------------
  // Run time access to DEDX, range, energy for a given particle, 
  // energy, and G4MaterialCutsCouple
  //-------------------------------------------------

  inline G4double GetDEDX(
    const G4ParticleDefinition *aParticle,
    G4double kineticEnergy,
    const G4MaterialCutsCouple *couple);

  inline G4double GetRange(
    const G4ParticleDefinition *aParticle,
    G4double kineticEnergy,
    const G4MaterialCutsCouple *couple);

  inline G4double GetCSDARange(
    const G4ParticleDefinition *aParticle,
    G4double kineticEnergy,
    const G4MaterialCutsCouple *couple);

  inline G4double GetRangeFromRestricteDEDX(
    const G4ParticleDefinition *aParticle,
    G4double kineticEnergy,
    const G4MaterialCutsCouple *couple);

  inline G4double GetEnergy(
    const G4ParticleDefinition *aParticle,
    G4double range,
    const G4MaterialCutsCouple *couple);

  inline G4double GetDEDXDispersion(
    const G4MaterialCutsCouple *couple,
    const G4DynamicParticle* dp,
          G4double& length);

  //-------------------------------------------------
  // Methods to be called only at initialisation
  // and at the end of the job
  //-------------------------------------------------

  void Register(G4VEnergyLossProcess* p);

  void DeRegister(G4VEnergyLossProcess* p);

  void Register(G4VMultipleScattering* p);

  void DeRegister(G4VMultipleScattering* p);

  void Register(G4VEmProcess* p);

  void DeRegister(G4VEmProcess* p);

  void Register(G4VProcess* p);

  void DeRegister(G4VProcess* p);

  void Register(G4VEmModel* p);

  void DeRegister(G4VEmModel* p);

  void Register(G4VEmFluctuationModel* p);

  void DeRegister(G4VEmFluctuationModel* p);

  void Register(G4VXRayModel* p);

  void DeRegister(G4VXRayModel* p);

  void RegisterExtraParticle(const G4ParticleDefinition* aParticle, 
                             G4VEnergyLossProcess* p);

  void SetVerbose(G4int val);

  void ResetParameters();

  void SetAtomDeexcitation(G4VAtomDeexcitation*);

  void SetSubCutProducer(G4VSubCutProducer*);

  void SetNIELCalculator(G4NIELCalculator*);

  //-------------------------------------------------
  // Access methods
  //-------------------------------------------------

  inline G4bool IsMaster() const;

  G4VEnergyLossProcess* GetEnergyLossProcess(const G4ParticleDefinition*);
 
  const std::vector<G4VEnergyLossProcess*>& GetEnergyLossProcessVector();

  const std::vector<G4VEmProcess*>& GetEmProcessVector();

  const std::vector<G4VMultipleScattering*>& GetMultipleScatteringVector();

  G4EmSaturation* EmSaturation();

  G4EmConfigurator* EmConfigurator();

  G4ElectronIonPair* ElectronIonPair();

  G4NIELCalculator* NIELCalculator();

  inline G4EmCorrections* EmCorrections();

  inline G4VAtomDeexcitation* AtomDeexcitation();

  inline G4VSubCutProducer* SubCutProducer();

  inline G4LossTableBuilder* GetTableBuilder();

  inline void SetGammaGeneralProcess(G4VEmProcess*); 

  inline G4VEmProcess* GetGammaGeneralProcess(); 

  inline void SetElectronGeneralProcess(G4VEmProcess*); 

  inline G4VEmProcess* GetElectronGeneralProcess(); 

  inline void SetPositronGeneralProcess(G4VEmProcess*); 

  inline G4VEmProcess* GetPositronGeneralProcess(); 

  G4LossTableManager(G4LossTableManager &) = delete;
  G4LossTableManager & operator=(const G4LossTableManager &right) = delete;

private:

  //-------------------------------------------------
  // Private methods and members
  //-------------------------------------------------

  G4LossTableManager();

  void Clear();

  G4VEnergyLossProcess* BuildTables(const G4ParticleDefinition* aParticle);

  void CopyTables(const G4ParticleDefinition* aParticle, 
                  G4VEnergyLossProcess*);

  void ParticleHaveNoLoss(const G4ParticleDefinition* aParticle);

  void CopyDEDXTables();

  void PrintEWarning(G4String, G4double);

  static G4ThreadLocal G4LossTableManager* instance;

  typedef const G4ParticleDefinition* PD;

  // cache
  G4VEnergyLossProcess* currentLoss{nullptr};
  PD currentParticle{nullptr};
  PD theElectron;
  PD theGenericIon{nullptr};
  PD firstParticle{nullptr};

  G4LossTableBuilder* tableBuilder;
  G4EmCorrections* emCorrections;
  G4EmConfigurator* emConfigurator{nullptr};
  G4ElectronIonPair* emElectronIonPair{nullptr};
  G4NIELCalculator* nielCalculator{nullptr};
  G4VAtomDeexcitation* atomDeexcitation{nullptr};
  G4VSubCutProducer* subcutProducer{nullptr};

  G4EmParameters* theParameters;
  G4VEmProcess* gGeneral{nullptr};
  G4VEmProcess* eGeneral{nullptr};
  G4VEmProcess* pGeneral{nullptr};

  G4int verbose;
  G4int n_loss{0};
  G4int run{-1};

  G4bool all_tables_are_built{false};
  G4bool startInitialisation{false};
  G4bool resetParam{true};
  G4bool isMaster{false};

  std::vector<G4VEnergyLossProcess*> loss_vector;
  std::vector<PD> part_vector;
  std::vector<PD> base_part_vector;
  std::vector<G4PhysicsTable*> dedx_vector;
  std::vector<G4PhysicsTable*> range_vector;
  std::vector<G4PhysicsTable*> inv_range_vector;
  std::vector<G4bool> tables_are_built;
  std::vector<G4bool> isActive;
  std::vector<G4VMultipleScattering*> msc_vector;
  std::vector<G4VEmProcess*> emp_vector;
  std::vector<G4VEmModel*> mod_vector;
  std::vector<G4VEmFluctuationModel*> fmod_vector;
  std::vector<G4VXRayModel*> xray_vector;
  std::vector<G4VProcess*> p_vector;

  std::map<PD,G4VEnergyLossProcess*,std::less<PD> > loss_map;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline
G4double G4LossTableManager::GetDEDX(const G4ParticleDefinition *aParticle,
                                     G4double kineticEnergy,
                                     const G4MaterialCutsCouple *couple)
{
  if(aParticle != currentParticle) { GetEnergyLossProcess(aParticle); }
  return currentLoss ? currentLoss->GetDEDX(kineticEnergy, couple) : 0.0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

inline
G4double G4LossTableManager::GetCSDARange(const G4ParticleDefinition *aParticle,
                                          G4double kineticEnergy,
                                          const G4MaterialCutsCouple *couple)
{
  if(aParticle != currentParticle) { GetEnergyLossProcess(aParticle); }
  return currentLoss ? currentLoss->GetCSDARange(kineticEnergy, couple) : DBL_MAX;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

inline
G4double G4LossTableManager::GetRangeFromRestricteDEDX(
                             const G4ParticleDefinition *aParticle,
                             G4double kineticEnergy,
                             const G4MaterialCutsCouple *couple)
{
  if(aParticle != currentParticle) { GetEnergyLossProcess(aParticle); }
  return currentLoss ? currentLoss->GetRange(kineticEnergy, couple) : DBL_MAX;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

inline
G4double G4LossTableManager::GetRange(const G4ParticleDefinition *aParticle,
                                      G4double kineticEnergy,
                                      const G4MaterialCutsCouple *couple)
{
  if(aParticle != currentParticle) { GetEnergyLossProcess(aParticle); }
  return currentLoss ? currentLoss->GetRange(kineticEnergy, couple) : DBL_MAX;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline
G4double G4LossTableManager::GetEnergy(const G4ParticleDefinition *aParticle,
                                       G4double range,
                                       const G4MaterialCutsCouple *couple)
{
  if(aParticle != currentParticle) { GetEnergyLossProcess(aParticle); }
  return currentLoss ? currentLoss->GetKineticEnergy(range, couple) : 0.0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline
G4double G4LossTableManager::GetDEDXDispersion(
                             const G4MaterialCutsCouple *couple,
                             const G4DynamicParticle* dp,
                             G4double& length)
{
  const G4ParticleDefinition* aParticle = dp->GetParticleDefinition();
  if(aParticle != currentParticle) { GetEnergyLossProcess(aParticle); }
  return currentLoss ? currentLoss->GetDEDXDispersion(couple, dp, length) : 0.0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

inline G4bool G4LossTableManager::IsMaster() const
{
  return isMaster;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline G4EmCorrections* G4LossTableManager::EmCorrections() 
{
  return emCorrections;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

inline G4VAtomDeexcitation* G4LossTableManager::AtomDeexcitation()
{
  return atomDeexcitation;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....

inline G4VSubCutProducer* G4LossTableManager::SubCutProducer()
{ 
  return subcutProducer;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

inline G4LossTableBuilder* G4LossTableManager::GetTableBuilder()
{
  return tableBuilder;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

inline void G4LossTableManager::SetGammaGeneralProcess(G4VEmProcess* ptr)
{
  gGeneral = ptr;
} 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

inline G4VEmProcess* G4LossTableManager::GetGammaGeneralProcess()
{
  return gGeneral;
} 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

inline void G4LossTableManager::SetElectronGeneralProcess(G4VEmProcess* ptr)
{
  eGeneral = ptr;
} 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

inline G4VEmProcess* G4LossTableManager::GetElectronGeneralProcess()
{
  return eGeneral;
} 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

inline void G4LossTableManager::SetPositronGeneralProcess(G4VEmProcess* ptr)
{
  pGeneral = ptr;
} 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

inline G4VEmProcess* G4LossTableManager::GetPositronGeneralProcess()
{
  return pGeneral;
} 
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#endif
