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

#ifndef G4DNABornIonisationModel1_h
#define G4DNABornIonisationModel1_h 1

#include "G4VEmModel.hh"
#include "G4ParticleChangeForGamma.hh"
#include "G4ProductionCutsTable.hh"

#include "G4DNACrossSectionDataSet.hh"
#include "G4Electron.hh"
#include "G4Proton.hh"
#include "G4DNAGenericIonsManager.hh"

#include "G4LogLogInterpolation.hh"

#include "G4DNAWaterIonisationStructure.hh"
#include "G4VAtomDeexcitation.hh"
#include "G4NistManager.hh"

class G4DNAChemistryManager;

class G4DNABornIonisationModel1 : public G4VEmModel
{

public:

  G4DNABornIonisationModel1(const G4ParticleDefinition* p = nullptr,
		           const G4String& nam = "DNABornIonisationModel");

  ~G4DNABornIonisationModel1() override;
   
  G4DNABornIonisationModel1 & operator=(const  G4DNABornIonisationModel1 &right) = delete;
  G4DNABornIonisationModel1(const  G4DNABornIonisationModel1&) = delete;

  void Initialise(const G4ParticleDefinition*, const G4DataVector& = *(new G4DataVector())) override;

  G4double CrossSectionPerVolume(  const G4Material* material,
					   const G4ParticleDefinition* p,
					   G4double ekin,
					   G4double emin,
					   G4double emax) override;

  void SampleSecondaries(std::vector<G4DynamicParticle*>*,
				 const G4MaterialCutsCouple*,
				 const G4DynamicParticle*,
				 G4double tmin,
				 G4double maxEnergy) override;

  G4double GetPartialCrossSection(const G4Material*,
                                          G4int /*level*/,
                                          const G4ParticleDefinition*,
                                          G4double /*kineticEnergy*/) override;
  void StartTracking(G4Track*) override;

  G4double DifferentialCrossSection(G4ParticleDefinition * aParticleDefinition, G4double k, G4double energyTransfer, G4int shell);

  G4double TransferedEnergy(G4ParticleDefinition * aParticleDefinition,
                            G4double incomingParticleEnergy, G4int shell, G4double random);

  inline void SelectFasterComputation(G4bool input); 

  inline void SelectStationary(G4bool input); 

  inline void SelectSPScaling(G4bool input); 

protected:

  G4ParticleChangeForGamma* fParticleChangeForGamma;

private:

  G4bool fasterCode{false};
  G4bool statCode{false};
  G4bool spScaling{true};

  // Water density table
  const std::vector<G4double>* fpMolWaterDensity;

  // Deexcitation manager to produce fluo photons and e-
  G4VAtomDeexcitation* fAtomDeexcitation;

  const G4Track* fTrack{nullptr};
  G4DNAChemistryManager* fChemistry{nullptr};
  

  std::map<G4String,G4double,std::less<G4String> > lowEnergyLimit;
  std::map<G4String,G4double,std::less<G4String> > highEnergyLimit;

  G4bool isInitialised{false};
  G4int verboseLevel;
  
  // Cross section
  using MapFile = std::map<G4String, G4String, std::less<G4String>>;
  MapFile tableFile; // useful ?

  using MapData = std::map<G4String, G4DNACrossSectionDataSet *, std::less<G4String>>;
  MapData tableData;
  
  // Final state
  
  G4DNAWaterIonisationStructure waterStructure;

  G4double RandomizeEjectedElectronEnergy(G4ParticleDefinition * aParticleDefinition, G4double incomingParticleEnergy, G4int shell) ;

  G4double RandomizeEjectedElectronEnergyFromCumulatedDcs(G4ParticleDefinition * aParticleDefinition, G4double incomingParticleEnergy, G4int shell) ;
   
  G4double Interpolate(G4double e1, G4double e2, G4double e, G4double xs1, G4double xs2);
   
  G4double QuadInterpolator( G4double e11, 
			     G4double e12, 
			     G4double e21, 
			     G4double e22, 
			     G4double x11,
			     G4double x12, 
			     G4double x21, 
			     G4double x22, 
			     G4double t1, 
			     G4double t2, 
			     G4double t, 
			     G4double e);

  using TriDimensionMap = std::map<G4double, std::map<G4double, G4double>>;
  
  TriDimensionMap eDiffCrossSectionData[6];
  TriDimensionMap eNrjTransfData[6]; // for cumulated dcs
  
  TriDimensionMap pDiffCrossSectionData[6];
  TriDimensionMap pNrjTransfData[6]; // for cumulated dcs
  
  std::vector<G4double> eTdummyVec;
  std::vector<G4double> pTdummyVec;

  using VecMap = std::map<G4double, std::vector<G4double>>;
  
  VecMap eVecm;
  VecMap pVecm;
  
  VecMap eProbaShellMap[6]; // for cumulated dcs
  VecMap pProbaShellMap[6]; // for cumulated dcs
  
  // Partial cross section  
  G4int RandomSelect(G4double energy,const G4String& particle );
};

inline void G4DNABornIonisationModel1::SelectFasterComputation (G4bool input)
{ 
  fasterCode = input;
}		 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

inline void G4DNABornIonisationModel1::SelectStationary (G4bool input)
{ 
  statCode = input;
}		 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

inline void G4DNABornIonisationModel1::SelectSPScaling (G4bool input)
{ 
  spScaling = input; 
}		 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#endif
